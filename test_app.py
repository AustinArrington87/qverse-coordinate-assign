import httpx
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QVerseTest:
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url
        self.test_data_dir = Path("test_data")
        self.ensure_test_data()
        
    def ensure_test_data(self):
        """Ensure test data directory exists."""
        # Create directories if they don't exist
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
        url = f"{self.base_url}/process-text"
        
        # Prepare the request payload
        payload = {
            "text": text,
            "category": category
        }
        
        # Add optional fields if present
        for key in ['subcategory', 'sub_subcategory', 'detail']:
            if key in kwargs:
                payload[key] = kwargs[key]
        
        # Handle metadata specially
        if 'metadata' in kwargs:
            payload['metadata'] = kwargs['metadata']
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload)
                self._log_response(response, "text", text[:50])
                return response.json() if response.status_code == 200 else None
            except Exception as e:
                logger.error(f"Error in text processing test: {e}")
                return None

    async def test_image_processing(self, image_path: str, category: str, **kwargs):
        """Test image processing endpoint."""
        url = f"{self.base_url}/process-image"
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
            
        async with httpx.AsyncClient() as client:
            try:
                # Prepare files and data
                files = {"file": (image_path.name, image_path.read_bytes(), "image/jpeg")}
                data = {"category": category}
                
                # Add optional fields
                for key in ['subcategory', 'sub_subcategory', 'detail']:
                    if key in kwargs:
                        data[key] = kwargs[key]
                
                # Handle metadata
                if 'metadata' in kwargs:
                    data['metadata'] = json.dumps(kwargs['metadata'])
                
                response = await client.post(url, data=data, files=files)
                self._log_response(response, "image", str(image_path))
                return response.json() if response.status_code == 200 else None
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
        url = f"{self.base_url}/process-multimodal"
        image_path = Path(image_path) if image_path else None
        
        if image_path and not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
            
        async with httpx.AsyncClient() as client:
            try:
                # Prepare data
                data = {"category": category}
                
                # Add optional fields
                for key in ['subcategory', 'sub_subcategory', 'detail']:
                    if key in kwargs:
                        data[key] = kwargs[key]
                
                # Add text if provided
                if text:
                    data["text"] = text
                
                # Handle metadata
                if 'metadata' in kwargs:
                    data['metadata'] = json.dumps(kwargs['metadata'])
                
                # Prepare files if image is provided
                files = {}
                if image_path and image_path.exists():
                    files["file"] = (image_path.name, image_path.read_bytes(), "image/jpeg")
                
                response = await client.post(url, data=data, files=files)
                self._log_response(
                    response, 
                    "multimodal", 
                    f"text: {text[:30] if text else 'None'}, image: {image_path}"
                )
                return response.json() if response.status_code == 200 else None
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

async def run_tests():
    logger.info("Starting Q-verse integration tests...")
    
    tester = QVerseTest()
    
    # Test data
    basketball_texts = [
        "The NBA championship game was intense with multiple lead changes",
        "LeBron James achieved a triple-double in the playoff game",
        "College basketball March Madness tournament highlights",
        "Basic basketball training drills for beginners"
    ]
    
    # Use existing basketball images
    basketball_images = [
        tester.test_data_dir / "basketball" / "player1.jpg",
        tester.test_data_dir / "basketball" / "game1.jpg",
        tester.test_data_dir / "basketball" / "court1.jpg"
    ]
    
    # Test text processing
    logger.info("\nTesting text processing...")
    for text in basketball_texts:
        await tester.test_text_processing(
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
    logger.info("\nTesting image processing...")
    for image_path in basketball_images:
        if image_path.exists():
            logger.info(f"\nProcessing image: {image_path.name}")
            await tester.test_image_processing(
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
        else:
            logger.warning(f"Image not found: {image_path}")
    
    # Test multimodal processing
    logger.info("\nTesting multimodal processing...")
    # Test each text with each image
    for text, image_path in zip(basketball_texts[:len(basketball_images)], basketball_images):
        if image_path.exists():
            logger.info(f"\nProcessing multimodal: {image_path.name}")
            await tester.test_multimodal_processing(
                text=text,
                image_path=str(image_path),
                category="sports_recreation",
                subcategory="team_sports",
                sub_subcategory="basketball",
                detail="nba",
                metadata={
                    "sport": "basketball",
                    "content_type": "mixed_content",
                    "image_type": image_path.stem,
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            logger.warning(f"Image not found: {image_path}")

    logger.info("\nTest suite completed!")

if __name__ == "__main__":
    asyncio.run(run_tests())
