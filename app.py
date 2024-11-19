# python3.11 -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import time
import json
from datetime import datetime
import os
from pathlib import Path
import logging
import shutil

from zones import zones, validate_category, validate_subcategory, get_coordinate_range
from embedding_utils import embedder
from adaptive_zones import adaptive_zones_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Q-verse Coordinate Service")

# Constants
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COORDINATES_FILE = DATA_DIR / "qverse_coordinates.json"
VECTOR_CACHE_FILE = DATA_DIR / "vector_cache.json"
UPLOAD_DIR = DATA_DIR / "uploads"

def ensure_directories():
    """Ensure all required directories exist."""
    try:
        for directory in [DATA_DIR, UPLOAD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory verified: {directory}")

        # Initialize files if they don't exist
        if not COORDINATES_FILE.exists():
            COORDINATES_FILE.write_text("[]")
            logger.info(f"Initialized coordinates file: {COORDINATES_FILE}")

        if not VECTOR_CACHE_FILE.exists():
            VECTOR_CACHE_FILE.write_text("{}")
            logger.info(f"Initialized vector cache file: {VECTOR_CACHE_FILE}")

    except Exception as e:
        logger.error(f"Error initializing directory structure: {e}")
        raise RuntimeError(f"Failed to initialize directories: {e}")

# Ensure directories exist on startup
ensure_directories()

class ContentRequest(BaseModel):
    text: Optional[str] = None
    category: str
    subcategory: Optional[str] = None
    sub_subcategory: Optional[str] = None
    detail: Optional[str] = None
    metadata: Optional[Dict] = None

class VectorData(BaseModel):
    vector: List[float]
    category: str
    subcategory: Optional[str] = None
    sub_subcategory: Optional[str] = None
    detail: Optional[str] = None
    metadata: Optional[Dict] = None

class CoordinateResponse(BaseModel):
    x: float
    y: float
    z: float
    timestamp: float
    category: str
    subcategory: Optional[str]
    sub_subcategory: Optional[str]
    detail: Optional[str]
    nearest_neighbors: Optional[List[Dict]] = None
    vector: List[float]
    source_type: str

class VectorCache:
    def __init__(self):
        self.vectors: Dict[str, List[List[float]]] = {}
        self.load_cache()
    
    def load_cache(self):
        """Load existing vectors from cache file."""
        try:
            if VECTOR_CACHE_FILE.exists():
                with open(VECTOR_CACHE_FILE, 'r') as f:
                    self.vectors = json.load(f)
            logger.info("Vector cache loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector cache: {e}")
            self.vectors = {}
    
    def save_cache(self):
        """Save vectors to cache file."""
        try:
            with open(VECTOR_CACHE_FILE, 'w') as f:
                json.dump(self.vectors, f)
            logger.info("Vector cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector cache: {e}")
    
    def add_vector(self, category: str, vector: List[float]):
        """Add a vector to the cache for a specific category."""
        if category not in self.vectors:
            self.vectors[category] = []
        self.vectors[category].append(vector)
        self.save_cache()
    
    def get_vectors(self, category: str) -> np.ndarray:
        """Get all vectors for a category."""
        return np.array(self.vectors.get(category, []))

# Initialize vector cache
vector_cache = VectorCache()

def load_coordinate_history() -> List[Dict]:
    """Load existing coordinate history from JSON file."""
    try:
        if COORDINATES_FILE.exists():
            with open(COORDINATES_FILE, 'r') as f:
                return json.load(f)
        logger.info("Coordinate history loaded successfully")
    except Exception as e:
        logger.error(f"Error loading coordinate history: {e}")
    return []

def save_coordinate_history(history: List[Dict]):
    """Save coordinate history to JSON file."""
    try:
        with open(COORDINATES_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Verify save
        if COORDINATES_FILE.exists():
            logger.info(f"Coordinate history saved successfully. Size: {COORDINATES_FILE.stat().st_size} bytes")
        else:
            logger.error("File not found after save attempt")
            
    except Exception as e:
        logger.error(f"Error saving coordinate history: {e}")

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector

# no more than 6 decmial places 
def round_coordinates(x: float, y: float, z: float, decimal_places: int = 12) -> Tuple[float, float, float]:
    """Round coordinates to specified decimal places."""
    return (
        round(float(x), decimal_places),
        round(float(y), decimal_places),
        round(float(z), decimal_places)
    )

# Calculate coordiantes using Nearest Neighbor algorithm
def calculate_coordinates_with_nn(
    vector: List[float],
    category: str,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    z_range: Tuple[int, int]
) -> Tuple[Tuple[float, float, float], Optional[List[Dict]]]:
    """Calculate coordinates using nearest neighbors when possible."""
    try:
        vector_np = np.array(vector, dtype=np.float32)
        if len(vector_np.shape) == 1:
            vector_np = vector_np.reshape(1, -1)
        
        # Get existing vectors for this category
        category_vectors = vector_cache.get_vectors(category)
        
        if len(category_vectors) < 5:  # Not enough vectors for meaningful NN
            # When not enough neighbors, use vector components to determine position within range
            # Ensure we use modulo to stay within range even if vector components are large
            vector_sum = np.sum(vector_np) if len(vector_np) > 0 else 0
            
            # Generate pseudo-random but deterministic coordinates within the ranges
            x_span = x_range[1] - x_range[0]
            y_span = y_range[1] - y_range[0]
            z_span = z_range[1] - z_range[0]
            
            # Use different components of the vector for each dimension
            x = x_range[0] + (abs(vector_sum * 1.23456) % x_span)  # Use different multipliers
            y = y_range[0] + (abs(vector_sum * 2.34567) % y_span)  # to get different positions
            z = z_range[0] + (abs(vector_sum * 3.45678) % z_span)  # for each dimension
            
            # Ensure coordinates are within bounds
            x = np.clip(x, x_range[0], x_range[1])
            y = np.clip(y, y_range[0], y_range[1])
            z = np.clip(z, z_range[0], z_range[1])
            
            return round_coordinates(float(x), float(y), float(z)), None
        
        # If we have enough vectors, use nearest neighbors
        vectors_array = np.array(category_vectors)
        n_neighbors = min(5, len(category_vectors))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(vectors_array)
        
        distances, indices = nn.kneighbors(vector_np)
        
        # Load history to get neighbor details
        history = load_coordinate_history()
        neighbors_info = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(history):
                neighbor = history[idx]
                neighbors_info.append({
                    "distance": float(distances[0][i]),
                    "coordinates": neighbor["coordinates"],
                    "category": neighbor["input_data"]["category"],
                    "subcategory": neighbor["input_data"].get("subcategory"),
                    "metadata": neighbor["input_data"].get("metadata")
                })
        
        if not neighbors_info:
            # If no valid neighbors found, use the same random-but-deterministic approach
            vector_sum = np.sum(vector_np)
            x = x_range[0] + (abs(vector_sum * 1.23456) % (x_range[1] - x_range[0]))
            y = y_range[0] + (abs(vector_sum * 2.34567) % (y_range[1] - y_range[0]))
            z = z_range[0] + (abs(vector_sum * 3.45678) % (z_range[1] - z_range[0]))
            return round_coordinates(float(x), float(y), float(z)), None
        
        # Calculate weighted average of neighbor coordinates
        weights = 1 / (distances[0] + 1e-6)
        weights = weights / weights.sum()
        
        weighted_coords = np.zeros(3)
        for i, neighbor in enumerate(neighbors_info):
            coords = [
                neighbor["coordinates"]["x"],
                neighbor["coordinates"]["y"],
                neighbor["coordinates"]["z"]
            ]
            weighted_coords += weights[i] * np.array(coords)
        
        # Ensure coordinates are within the specified ranges
        x = np.clip(weighted_coords[0], x_range[0], x_range[1])
        y = np.clip(weighted_coords[1], y_range[0], y_range[1])
        z = np.clip(weighted_coords[2], z_range[0], z_range[1])
        
        return round_coordinates(float(x), float(y), float(z)), neighbors_info
        
    except Exception as e:
        logger.error(f"Error in coordinate calculation: {e}")
        # Even on error, generate valid coordinates instead of using center point
        vector_sum = np.sum(vector_np) if len(vector_np) > 0 else 0
        x = x_range[0] + (abs(vector_sum * 1.23456) % (x_range[1] - x_range[0]))
        y = y_range[0] + (abs(vector_sum * 2.34567) % (y_range[1] - y_range[0]))
        z = z_range[0] + (abs(vector_sum * 3.45678) % (z_range[1] - z_range[0]))
        return round_coordinates(float(x), float(y), float(z)), None

def simple_interpolation(
    vector: np.ndarray,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    z_range: Tuple[int, int]
) -> Tuple[float, float, float]:
    """Simple interpolation fallback for coordinate calculation."""
    # Ensure vector is 1-dimensional
    if len(vector.shape) > 1:
        vector = vector.flatten()
    
    # Pad vector if needed
    if len(vector) < 3:
        vector = np.pad(vector, (0, 3 - len(vector)), mode='constant', constant_values=0.5)
    
    x = float(np.interp(vector[0], [0, 1], x_range))
    y = float(np.interp(vector[1], [0, 1], y_range))
    z = float(np.interp(vector[2], [0, 1], z_range))
    
    return (x, y, z)


@app.on_event("startup")
async def startup_event():
    """Initialize necessary resources on startup."""
    logger.info("Starting Q-verse Coordinate Service")
    ensure_directories()
    logger.info("Initialization complete")

@app.post("/process-text", response_model=CoordinateResponse)
async def process_text(request: ContentRequest):
    """Process text content and assign coordinates."""
    try:
        # Extract embedding
        vector = embedder.get_text_embedding(request.text)
        
        # Create vector data
        vector_data = VectorData(
            vector=vector,
            category=request.category,
            subcategory=request.subcategory,
            sub_subcategory=request.sub_subcategory,
            detail=request.detail,
            metadata={
                "content_type": "text",
                "original_text": request.text,
                **(request.metadata or {})
            }
        )

        # Get coordinates
        coordinates = await assign_coordinates(vector_data)
        coordinates.source_type = "text"
        return coordinates

    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-image", response_model=CoordinateResponse)
async def process_image(
    file: UploadFile = File(...),
    category: str = Form(...),
    subcategory: Optional[str] = Form(None),
    sub_subcategory: Optional[str] = Form(None),
    detail: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Process image content and assign coordinates."""
    temp_path = None
    try:
        # Save uploaded file
        temp_path = UPLOAD_DIR / file.filename
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract embedding
        vector = embedder.get_image_embedding(str(temp_path))
        
        # Handle metadata
        base_metadata = {
            "content_type": "image",
            "original_filename": file.filename
        }
        
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
                base_metadata.update(metadata_dict)
            except (json.JSONDecodeError, TypeError):
                base_metadata["raw_metadata"] = str(metadata)

        # Create vector data
        vector_data = VectorData(
            vector=vector,
            category=category,
            subcategory=subcategory,
            sub_subcategory=sub_subcategory,
            detail=detail,
            metadata=base_metadata
        )

        # Get coordinates
        coordinates = await assign_coordinates(vector_data)
        coordinates.source_type = "image"
        return coordinates

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {e}")

@app.post("/process-multimodal", response_model=CoordinateResponse)
async def process_multimodal(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    category: str = Form(...),
    subcategory: Optional[str] = Form(None),
    sub_subcategory: Optional[str] = Form(None),
    detail: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Process both text and image content together."""
    temp_path = None
    try:
        vectors = []
        
        # Process text if provided
        if text:
            text_vector = np.array(embedder.get_text_embedding(text))
            vectors.append(text_vector)
        
        # Process image if provided
        if file:
            temp_path = UPLOAD_DIR / file.filename
            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_vector = np.array(embedder.get_image_embedding(str(temp_path)))
            vectors.append(image_vector)
        
        if not vectors:
            raise HTTPException(status_code=400, detail="Either text or image must be provided")
        
        # Ensure all vectors have same shape
        min_dim = min(v.shape[0] for v in vectors)
        vectors = [v[:min_dim] for v in vectors]
        
        # Combine vectors
        combined_vector = np.mean(vectors, axis=0).tolist()

        # Handle metadata
        base_metadata = {
            "content_type": "multimodal",
            "has_text": bool(text),
            "has_image": bool(file),
            "original_filename": file.filename if file else None
        }
        
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
                base_metadata.update(metadata_dict)
            except (json.JSONDecodeError, TypeError):
                base_metadata["raw_metadata"] = str(metadata)

        # Create vector data
        vector_data = VectorData(
            vector=combined_vector,
            category=category,
            subcategory=subcategory,
            sub_subcategory=sub_subcategory,
            detail=detail,
            metadata=base_metadata
        )

        # Get coordinates
        coordinates = await assign_coordinates(vector_data)
        coordinates.source_type = "multimodal"
        return coordinates

    except Exception as e:
        logger.error(f"Error processing multimodal content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {e}")

@app.post("/predict-zone")
async def predict_zone(vector_data: List[float]):
    """Predict the most appropriate zone for a vector."""
    try:
        predicted_zone, confidence = adaptive_zones_manager.predict_zone(vector_data)
        return {
            "predicted_zone": predicted_zone,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error predicting zone: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/zone-learning-status")
async def get_zone_learning_status():
    """Get the current status of zone learning."""
    try:
        status = {}
        for zone in adaptive_zones_manager.vectors.keys():
            status[zone] = {
                "vector_count": len(adaptive_zones_manager.vectors[zone]),
                "has_centroid": zone in adaptive_zones_manager.zone_centroids,
                "clusters": len(adaptive_zones_manager.clusters.get(zone, {})),
                "learning_complete": len(adaptive_zones_manager.vectors[zone]) >= adaptive_zones_manager.min_samples
            }
        return status
    except Exception as e:
        logger.error(f"Error getting zone learning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assign-coordinates", response_model=CoordinateResponse)
async def assign_coordinates(data: VectorData):
    """Assign coordinates and update adaptive zones."""
    try:
        # Update adaptive zones with the new vector
        adaptive_zones_manager.add_vector(
            category=data.category,
            vector=data.vector
        )
        
        # Get coordinate ranges - try adaptive zones first
        stats = adaptive_zones_manager.get_category_stats(data.category)
        
        if stats.get("has_sufficient_samples"):
            adaptive_zones = adaptive_zones_manager.get_adaptive_zones()
            if "adaptive_ranges" in adaptive_zones.get(data.category, {}):
                # TODO: Implement adaptive ranges logic
                pass
        
        # Fall back to base zones
        x_range, y_range, z_range = get_coordinate_range(
            data.category, 
            data.subcategory,
            data.sub_subcategory,
            data.detail
        )
        
        # Calculate coordinates
        (x, y, z), neighbors = calculate_coordinates_with_nn(
            data.vector,
            data.category,
            x_range,
            y_range,
            z_range
        )
        
        timestamp = time.time()
        
        # Create response
        response = CoordinateResponse(
            x=x,
            y=y,
            z=z,
            timestamp=timestamp,
            category=data.category,
            subcategory=data.subcategory,
            sub_subcategory=data.sub_subcategory,
            detail=data.detail,
            nearest_neighbors=neighbors,
            vector=data.vector,
            source_type="direct"  # Will be overridden by process endpoints
        )
        
        # Save to history
        complete_record = {
            "input_data": data.dict(),
            "coordinates": {
                "x": x,
                "y": y,
                "z": z
            },
            "temporal": {
                "unix_timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "date": datetime.fromtimestamp(timestamp).date().isoformat(),
                "time": datetime.fromtimestamp(timestamp).time().isoformat()
            },
            "nearest_neighbors": neighbors
        }
        
        # Add vector to cache
        vector_cache.add_vector(data.category, data.vector)
        
        # Update history
        history = load_coordinate_history()
        history.append(complete_record)
        save_coordinate_history(history)
        
        return response
        
    except Exception as e:
        logger.error(f"Error assigning coordinates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/zones")
async def get_zones():
    """Get all available zones and their configurations."""
    return zones

@app.get("/adaptive-zones")
async def get_adaptive_zones():
    """Get current adaptive zones configuration."""
    return adaptive_zones_manager.get_adaptive_zones()

@app.get("/zone-stats/{category}")
async def get_zone_stats(category: str):
    """Get statistics for a specific zone category."""
    stats = adaptive_zones_manager.get_category_stats(category)
    if not stats:
        raise HTTPException(status_code=404, detail=f"Category {category} not found")
    return stats

@app.get("/coordinate-history")
async def get_coordinate_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None
):
    """
    Retrieve coordinate history with optional filtering by date range and category.
    Dates should be in YYYY-MM-DD format.
    """
    history = load_coordinate_history()
    
    filtered_history = history
    
    if start_date:
        filtered_history = [
            record for record in filtered_history
            if record["temporal"]["date"] >= start_date
        ]
    
    if end_date:
        filtered_history = [
            record for record in filtered_history
            if record["temporal"]["date"] <= end_date
        ]
    
    if category:
        filtered_history = [
            record for record in filtered_history
            if record["input_data"]["category"] == category
        ]
    
    return filtered_history

@app.get("/status")
async def get_status():
    """Get service status including adaptive zones information."""
    try:
        base_status = {
            "status": "healthy",
            "data_directory": str(DATA_DIR),
            "coordinates_file": {
                "exists": COORDINATES_FILE.exists(),
                "size": COORDINATES_FILE.stat().st_size if COORDINATES_FILE.exists() else 0
            },
            "vector_cache_file": {
                "exists": VECTOR_CACHE_FILE.exists(),
                "size": VECTOR_CACHE_FILE.stat().st_size if VECTOR_CACHE_FILE.exists() else 0
            }
        }
        
        # Add adaptive zones status
        adaptive_status = {
            "adaptive_zones": {
                "last_update": adaptive_zones_manager.last_update,
                "categories": {
                    category: adaptive_zones_manager.get_category_stats(category)
                    for category in zones.keys()
                }
            }
        }
        
        return {**base_status, **adaptive_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking service status: {str(e)}")

@app.delete("/coordinate-history")
async def clear_coordinate_history():
    """Clear the coordinate history file and vector cache."""
    try:
        if COORDINATES_FILE.exists():
            COORDINATES_FILE.unlink()
        if VECTOR_CACHE_FILE.exists():
            VECTOR_CACHE_FILE.unlink()
        
        # Reinitialize empty files
        ensure_directories()
        
        return {"message": "Coordinate history and vector cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)

#### DOCUMENTATION - USER GUIDE ##############
# Core Functionality:
# Creates a FastAPI service for managing 3D coordinates in a virtual space (Q-verse)
# Maps high-dimensional vectors to 3D coordinates within predefined zones
# Uses nearest neighbor algorithms to place similar items close together
# Maintains historical records of all assignments

# Data Structures:
# Uses zones configuration imported from zones.py
# Maintains two persistent storage files:
# qverse_coordinates.json: Complete history of coordinate assignments
# vector_cache.json: Cache of vectors for nearest neighbor calculations

# Key Classes:
# VectorData: Input data model including:
# Vector values
# Category hierarchy (category/subcategory/sub-subcategory/detail)
# Optional metadata

# CoordinateResponse: Output data model with:
# x, y, z coordinates
# Timestamp
# Category information
# Nearest neighbor information

# VectorCache: Manages vector storage with:
# Load/save functionality
# Category-based organization
# Vector retrieval methods


# Main Functions:

# calculate_coordinates_with_nn():
# Uses nearest neighbors when enough data exists
# Falls back to simple interpolation for new categories
# Provides weighted coordinate calculations
# Returns related items information

# get_coordinate_range():
# Validates category hierarchy
# Determines coordinate ranges based on zones
# Handles multiple levels of categorization


# API Endpoints:

# POST /assign-coordinates
# Accepts vector data
# Assigns 3D coordinates
# Returns coordinates and neighbor information
# Saves to history and cache

# GET /zones
# Returns complete zone configuration
# Shows available categories and hierarchies

# GET /coordinate-history
# Retrieves historical assignments
# Supports filtering by:
# Date range
# Category
# Other metadata

# DELETE /coordinate-history
# Clears historical data
# Resets vector cache


# Data Storage:
# Saves each coordinate assignment with:
# Input data (vector, categories, metadata)
# Calculated coordinates (x, y, z)
# Temporal information (multiple formats)
# Nearest neighbor relationships

# Error Handling:
# Validates category hierarchies
# Handles missing or invalid data
# Provides meaningful error messages
# Includes fallback mechanisms

# Temporal Features:
# Timestamps each coordinate assignment
# Stores multiple time formats:
# Unix timestamp
# ISO datetime
# Separate date and time
# Supports temporal queries

# Spatial Organization:
# Maintains zone boundaries
# Ensures coordinates stay within assigned ranges
# Uses nearest neighbors for spatial coherence
# Preserves relationships between similar items
