import httpx
import asyncio
from typing import Dict, Any, Optional, Union, List  
from pathlib import Path
import json
from datetime import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QVerseVisualizer:
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style properly
        plt.style.use('default')
        sns.set_theme()
        sns.set_palette("husl")

    def plot_3d_coordinates(self, results: List[Dict[str, Any]], title: str = "Q-verse Coordinates"):
        """Plot 3D scatter of coordinates colored by category."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Plot points
        categories = df['category'].unique()
        for cat in categories:
            mask = df['category'] == cat
            ax.scatter(
                df[mask]['x'], 
                df[mask]['y'], 
                df[mask]['z'],
                label=cat,
                alpha=0.6
            )
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(title)
        plt.legend()
        
        # Save plot
        plt.savefig(self.output_dir / f"coordinates_3d_{self.timestamp}.png")
        plt.close()

class QVerseTest:
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url
        self.test_data_dir = Path("test_data")
        self.ensure_test_data()
        self.visualizer = QVerseVisualizer()
        self.performance_metrics = defaultdict(list)
        self.results = []
        self.confidences = []
        self.confidence_threshold = 0.7
        
    def ensure_test_data(self):
        """Ensure test data directory exists."""
        basketball_dir = self.test_data_dir / "basketball"
        basketball_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using test data directory: {basketball_dir}")
        
        # Log available images
        images = list(basketball_dir.glob("*.jpg"))
        if images:
            logger.info("Found images:")
            for img in images:
                logger.info(f"  - {img.name}")
        else:
            logger.warning("No images found in test data directory")

    async def test_text_processing(self, text: str, category: str, **kwargs):
        """Test text processing endpoint."""
        start_time = time.time()
        url = f"{self.base_url}/process-text"
        
        payload = {
            "text": text,
            "category": category
        }
        
        # Add optional fields
        for key in ['subcategory', 'sub_subcategory', 'detail']:
            if key in kwargs:
                payload[key] = kwargs[key]
        
        if 'metadata' in kwargs:
            payload['metadata'] = kwargs['metadata']
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload)
                processing_time = time.time() - start_time
                self.performance_metrics['processing_time'].append(processing_time)
                
                if response.status_code == 200:
                    result = response.json()
                    self.results.append(result)
                    self._log_response(response, "text", text[:50])
                    return result
                else:
                    logger.error(f"Error {response.status_code}: {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Error in text processing test: {e}")
                return None

    async def test_image_processing(self, image_path: str, category: str, **kwargs):
        """Test image processing endpoint."""
        start_time = time.time()
        url = f"{self.base_url}/process-image"
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
            
        async with httpx.AsyncClient() as client:
            try:
                with open(image_path, 'rb') as f:
                    files = {"file": (image_path.name, f, "image/jpeg")}
                    data = {"category": category}
                    
                    for key in ['subcategory', 'sub_subcategory', 'detail']:
                        if key in kwargs:
                            data[key] = kwargs[key]
                    
                    if 'metadata' in kwargs:
                        data['metadata'] = json.dumps(kwargs['metadata'])
                    
                    response = await client.post(url, data=data, files=files)
                    processing_time = time.time() - start_time
                    self.performance_metrics['processing_time'].append(processing_time)
                    
                    if response.status_code == 200:
                        result = response.json()
                        self.results.append(result)
                        self._log_response(response, "image", str(image_path))
                        return result
            except Exception as e:
                logger.error(f"Error in image processing test: {e}")
                return None

    async def test_multimodal_processing(
        self, 
        text: Optional[str], 
        image_path: Optional[str], 
        category: str, 
        **kwargs
    ):
        """Test multimodal processing endpoint."""
        start_time = time.time()
        url = f"{self.base_url}/process-multimodal"
        image_path = Path(image_path) if image_path else None
        
        if image_path and not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
            
        async with httpx.AsyncClient() as client:
            try:
                data = {"category": category}
                
                for key in ['subcategory', 'sub_subcategory', 'detail']:
                    if key in kwargs:
                        data[key] = kwargs[key]
                
                if text:
                    data["text"] = text
                
                if 'metadata' in kwargs:
                    data['metadata'] = json.dumps(kwargs['metadata'])
                
                files = {}
                if image_path and image_path.exists():
                    with open(image_path, 'rb') as f:
                        files["file"] = (image_path.name, f.read(), "image/jpeg")
                
                response = await client.post(url, data=data, files=files)
                processing_time = time.time() - start_time
                self.performance_metrics['processing_time'].append(processing_time)
                
                if response.status_code == 200:
                    result = response.json()
                    self.results.append(result)
                    self._log_response(
                        response, 
                        "multimodal", 
                        f"text: {text[:30] if text else 'None'}, image: {image_path}"
                    )
                    return result
            except Exception as e:
                logger.error(f"Error in multimodal processing test: {e}")
                return None

    def _log_response(self, response, content_type: str, content_preview: str):
        """Log response details."""
        if response.status_code == 200:
            result = response.json()
            logger.info(f"\nSuccess for {content_type} content: {content_preview}")
            logger.info(f"Coordinates: x={result['x']:.2f}, y={result['y']:.2f}, z={result['z']:.2f}")
            if result.get('nearest_neighbors'):
                logger.info("\nNearest neighbors:")
                for neighbor in result['nearest_neighbors']:
                    logger.info(f"Distance: {neighbor['distance']:.2f}")
                    logger.info(f"Category: {neighbor['category']}")
        else:
            logger.error(f"Error {response.status_code}: {response.text}")

    async def test_raw_content(self, content_type: str, content: Union[str, Path], expected_zone: str = None):
        """Test content processing without metadata."""
        logger.info(f"\nTesting raw {content_type} content prediction...")
        
        if content_type == "text":
            url = f"{self.base_url}/process-text"
            payload = {
                "text": content,
                "category": expected_zone or "science_nature"  # Use expected zone or default to valid category
            }
            
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(url, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"Raw text: '{content[:50]}...'")
                        logger.info(f"Coordinates: x={result['x']:.2f}, y={result['y']:.2f}, z={result['z']:.2f}")
                        if expected_zone:
                            logger.info(f"Expected zone: {expected_zone}")
                        return result
                except Exception as e:
                    logger.error(f"Error in raw text test: {e}")
                    
        elif content_type == "image":
            url = f"{self.base_url}/process-image"
            image_path = Path(content)
            
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return None
            
            async with httpx.AsyncClient() as client:
                try:
                    with open(image_path, 'rb') as f:
                        files = {"file": (image_path.name, f, "image/jpeg")}
                        data = {"category": expected_zone or "science_nature"}  # Use expected zone or default
                        
                        response = await client.post(url, data=data, files=files)
                        if response.status_code == 200:
                            result = response.json()
                            logger.info(f"Raw image: {image_path.name}")
                            logger.info(f"Coordinates: x={result['x']:.2f}, y={result['y']:.2f}, z={result['z']:.2f}")
                            if expected_zone:
                                logger.info(f"Expected zone: {expected_zone}")
                            return result
                except Exception as e:
                    logger.error(f"Error in raw image test: {e}")

    async def run_tests(self):
        """Run all tests."""
        logger.info("Starting Q-verse integration tests...")
        
        # Test data
        basketball_texts = [
            "The NBA championship game was intense with multiple lead changes",
            "LeBron James achieved a triple-double in the playoff game",
            "College basketball March Madness tournament highlights",
            "Basic basketball training drills for beginners"
        ]
        
        basketball_images = [
            self.test_data_dir / "basketball" / "player1.jpg",
            self.test_data_dir / "basketball" / "game1.jpg",
            self.test_data_dir / "basketball" / "court1.jpg"
        ]
        
        # Phase 1: Training with metadata
        logger.info("\nPhase 1: Training with labeled data...")
        for text in basketball_texts:
            await self.test_text_processing(
                text=text,
                category="sports_recreation",
                subcategory="team_sports",
                sub_subcategory="basketball",
                detail="nba",
                metadata={
                    "sport": "basketball", 
                    "content_type": "game_description",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Test image processing
        for image_path in basketball_images:
            if image_path.exists():
                logger.info(f"\nProcessing image: {image_path.name}")
                await self.test_image_processing(
                    image_path=str(image_path),
                    category="sports_recreation",
                    subcategory="team_sports",
                    sub_subcategory="basketball",
                    detail="nba",
                    metadata={
                        "sport": "basketball", 
                        "content_type": "game_photo",
                        "image_type": image_path.stem,
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        # Phase 2: Testing without metadata
        logger.info("\nPhase 2: Testing raw content...")
        test_cases = [
            {
                "content_type": "text",
                "content": "The basketball game went into overtime",
                "expected_zone": "sports_recreation"
            },
            {
                "content_type": "image",
                "content": str(basketball_images[0]),
                "expected_zone": "sports_recreation"
            }
        ]
        
        for test_case in test_cases:
            await self.test_raw_content(**test_case)

        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        self.visualizer.plot_3d_coordinates(self.results)

if __name__ == "__main__":
    tester = QVerseTest()
    asyncio.run(tester.run_tests())
