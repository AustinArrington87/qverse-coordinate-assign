# python3.11 -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
from fastapi import FastAPI, HTTPException
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

from zones import zones, validate_category, validate_subcategory, get_zone_ranges

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COORDINATES_FILE = DATA_DIR / "qverse_coordinates.json"
VECTOR_CACHE_FILE = DATA_DIR / "vector_cache.json"

def ensure_data_directory():
    """Ensure the data directory and required files exist."""
    try:
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directory verified: {DATA_DIR}")

        # Initialize coordinates file if it doesn't exist
        if not COORDINATES_FILE.exists():
            COORDINATES_FILE.write_text("[]")
            logger.info(f"Initialized coordinates file: {COORDINATES_FILE}")

        # Initialize vector cache file if it doesn't exist
        if not VECTOR_CACHE_FILE.exists():
            VECTOR_CACHE_FILE.write_text("{}")
            logger.info(f"Initialized vector cache file: {VECTOR_CACHE_FILE}")

    except Exception as e:
        logger.error(f"Error initializing data directory structure: {e}")
        raise RuntimeError(f"Failed to initialize data directory: {e}")

# Ensure data directory exists on startup
ensure_data_directory()

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
        logger.info("Coordinate history saved successfully")
    except Exception as e:
        logger.error(f"Error saving coordinate history: {e}")

def get_coordinate_range(
    category: str, 
    subcategory: Optional[str] = None,
    sub_subcategory: Optional[str] = None,
    detail: Optional[str] = None
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Get the coordinate ranges for the specified category hierarchy."""
    if not validate_category(category):
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
    
    zone = zones[category]
    x_range = zone["x_range"]
    y_range = zone["y_range"]
    
    if not subcategory:
        return x_range, y_range, zone["z_range"]
    
    if not validate_subcategory(category, subcategory):
        raise HTTPException(status_code=400, detail=f"Invalid subcategory: {subcategory}")
    
    subcat = zone["subcategories"][subcategory]
    
    if not sub_subcategory:
        return x_range, y_range, subcat["z_subrange"]
    
    if sub_subcategory not in subcat["sub_subcategories"]:
        raise HTTPException(status_code=400, detail=f"Invalid sub-subcategory: {sub_subcategory}")
    
    sub_subcat = subcat["sub_subcategories"][sub_subcategory]
    
    if isinstance(sub_subcat, tuple):
        return x_range, y_range, sub_subcat
    elif isinstance(sub_subcat, dict):
        if not detail:
            return x_range, y_range, sub_subcat["range"]
        if detail not in sub_subcat["details"]:
            raise HTTPException(status_code=400, detail=f"Invalid detail: {detail}")
        return x_range, y_range, sub_subcat["details"][detail]
    
    raise HTTPException(status_code=500, detail="Invalid zone configuration")

def calculate_coordinates_with_nn(
    vector: List[float],
    category: str,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    z_range: Tuple[int, int]
) -> Tuple[Tuple[float, float, float], Optional[List[Dict]]]:
    """Calculate coordinates using nearest neighbors when possible."""
    vector_np = np.array(vector).reshape(1, -1)
    
    # Get existing vectors for this category
    category_vectors = vector_cache.get_vectors(category)
    
    if len(category_vectors) < 5:  # Not enough vectors for meaningful NN
        # Fall back to simple interpolation
        x = np.interp(vector_np[0, 0] if vector_np.shape[1] > 0 else 0.5, [0, 1], x_range)
        y = np.interp(vector_np[0, 1] if vector_np.shape[1] > 1 else 0.5, [0, 1], y_range)
        z = np.interp(vector_np[0, 2] if vector_np.shape[1] > 2 else 0.5, [0, 1], z_range)
        return (x, y, z), None
    
    # Initialize nearest neighbors
    n_neighbors = min(5, len(category_vectors))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(category_vectors)
    
    # Find nearest neighbors
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
    
    # Calculate position based on weighted average of neighbors
    scaler = MinMaxScaler()
    
    # Prepare coordinates array for scaling
    neighbor_coords = np.array([[n["coordinates"]["x"], n["coordinates"]["y"], n["coordinates"]["z"]] 
                              for n in neighbors_info])
    
    # Calculate weights based on inverse distances
    weights = 1 / (distances[0] + 1e-6)  # Add small epsilon to avoid division by zero
    weights = weights / weights.sum()
    
    # Calculate weighted average
    weighted_coords = np.average(neighbor_coords, weights=weights, axis=0)
    
    # Scale to range
    x = np.clip(weighted_coords[0], x_range[0], x_range[1])
    y = np.clip(weighted_coords[1], y_range[0], y_range[1])
    z = np.clip(weighted_coords[2], z_range[0], z_range[1])
    
    return (x, y, z), neighbors_info

@app.on_event("startup")
async def startup_event():
    """Initialize necessary resources on startup."""
    logger.info("Starting Q-verse Coordinate Service")
    ensure_data_directory()
    logger.info("Initialization complete")

@app.post("/assign-coordinates", response_model=CoordinateResponse)
async def assign_coordinates(data: VectorData):
    """Assign 3D coordinates to the input vector data."""
    # Get coordinate ranges based on all category levels
    x_range, y_range, z_range = get_coordinate_range(
        data.category, 
        data.subcategory,
        data.sub_subcategory,
        data.detail
    )
    
    # Calculate 3D coordinates from the vector using nearest neighbors
    (x, y, z), neighbors = calculate_coordinates_with_nn(
        data.vector,
        data.category,
        x_range,
        y_range,
        z_range
    )
    
    # Get current timestamp
    timestamp = time.time()
    
    # Create response object
    response = CoordinateResponse(
        x=x,
        y=y,
        z=z,
        timestamp=timestamp,
        category=data.category,
        subcategory=data.subcategory,
        sub_subcategory=data.sub_subcategory,
        detail=data.detail,
        nearest_neighbors=neighbors
    )
    
    # Create complete record including input data and results
    complete_record = {
        "input_data": {
            "vector": data.vector,
            "category": data.category,
            "subcategory": data.subcategory,
            "sub_subcategory": data.sub_subcategory,
            "detail": data.detail,
            "metadata": data.metadata
        },
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
    
    # Load existing history
    history = load_coordinate_history()
    
    # Append new record
    history.append(complete_record)
    
    # Save updated history
    save_coordinate_history(history)
    
    return response

@app.get("/zones")
async def get_zones():
    """Get all available zones and their configurations."""
    return zones

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
    """Check the status of the service and its data files."""
    try:
        return {
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
        ensure_data_directory()
        
        return {"message": "Coordinate history and vector cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

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
