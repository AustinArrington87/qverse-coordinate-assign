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
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style properly
        plt.style.use('default')  # Use default style first
        sns.set_theme()  # Then apply seaborn theme
        sns.set_palette("husl")  # Set color palette
    
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
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.legend()
        
        # Save plot
        plt.savefig(self.output_dir / f"coordinates_3d_{self.timestamp}.png")
        plt.close()
    
    def plot_confidence_distribution(self, confidences: List[float], threshold: float = 0.7):
        """Plot histogram of prediction confidences."""
        plt.figure(figsize=(10, 6))
        sns.histplot(confidences, bins=20)
        plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        plt.title('Distribution of Prediction Confidences')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(self.output_dir / f"confidence_dist_{self.timestamp}.png")
        plt.close()
    
    def plot_performance_metrics(self, metrics: Dict[str, List[float]]):
        """Plot performance metrics over time."""
        plt.figure(figsize=(12, 6))
        
        for metric, values in metrics.items():
            plt.plot(values, label=metric)
        
        plt.title('Performance Metrics Over Time')
        plt.xlabel('Test Case')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(self.output_dir / f"performance_metrics_{self.timestamp}.png")
        plt.close()
    
    def plot_zone_learning_progress(self, zone_stats: Dict[str, Dict[str, Any]]):
        """Plot learning progress for each zone."""
        # Prepare data
        zones = list(zone_stats.keys())
        vector_counts = [stats['vector_count'] for stats in zone_stats.values()]
        completion_status = [stats['learning_complete'] for stats in zone_stats.values()]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(zones, vector_counts)
        
        # Color bars based on completion status
        for bar, complete in zip(bars, completion_status):
            bar.set_color('green' if complete else 'orange')
        
        plt.title('Zone Learning Progress')
        plt.xlabel('Zone')
        plt.ylabel('Number of Vectors')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"zone_progress_{self.timestamp}.png")
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
        self.confidence_threshold = 0.7  # Minimum confidence threshold
        
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
        """Test text processing endpoint with performance tracking."""
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
        """Test image processing endpoint with performance tracking."""
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
                    else:
                        logger.error(f"Error {response.status_code}: {response.text}")
                        return None
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
        """Test multimodal processing endpoint with performance tracking."""
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
                else:
                    logger.error(f"Error {response.status_code}: {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Error in multimodal processing test: {e}")
                return None

    async def test_raw_content(self, content_type: str, content: Union[str, Path], expected_zone: str = None):
        """Test content processing without metadata."""
        start_time = time.time()
        
        if content_type == "text":
            result = await self._test_raw_text(content, expected_zone)
        elif content_type == "image":
            result = await self._test_raw_image(content, expected_zone)
        else:
            logger.error(f"Unsupported content type: {content_type}")
            return None
        
        processing_time = time.time() - start_time
        
        if result:
            self.results.append(result)
            if 'confidence' in result:
                confidence = result['confidence']
                self.confidences.append(confidence)
                
                # Track performance metrics
                self.performance_metrics['processing_time'].append(processing_time)
                self.performance_metrics['confidence'].append(confidence)
                if expected_zone:
                    correct_prediction = result['predicted_zone'] == expected_zone
                    self.performance_metrics['accuracy'].append(float(correct_prediction))
                
                # Log confidence threshold warnings
                if confidence < self.confidence_threshold:
                    logger.warning(f"Low confidence prediction ({confidence:.2f}) for {content_type}")
        
        return result

    async def _test_raw_text(self, text: str, expected_zone: str = None):
        """Helper for testing raw text content."""
        url = f"{self.base_url}/process-text"
        payload = {
            "text": text,
            "category": "unknown"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Raw text: '{text[:50]}...'")
                    logger.info(f"Coordinates: x={result['x']:.2f}, y={result['y']:.2f}, z={result['z']:.2f}")
                    if expected_zone:
                        logger.info(f"Expected zone: {expected_zone}")
                    return result
            except Exception as e:
                logger.error(f"Error in raw text test: {e}")
                return None

    async def _test_raw_image(self, image_path: Union[str, Path], expected_zone: str = None):
        """Helper for testing raw image content."""
        url = f"{self.base_url}/process-image"
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
        
        async with httpx.AsyncClient() as client:
            try:
                with open(image_path, 'rb') as f:
                    files = {"file": (image_path.name, f, "image/jpeg")}
                    data = {"category": "unknown"}
                    
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
                return None

    async def test_zone_prediction(self):
        """Test the system's ability to predict zones for raw content."""
        logger.info("\nTraining system with labeled data...")
        
        # Train with basketball content
        basketball_texts = [
            "The NBA championship game was intense with multiple lead changes",
            "LeBron James achieved a triple-double in the playoff game",
            "College basketball March Madness tournament highlights"
        ]
        
        for text in basketball_texts:
            await self.test_text_processing(
                text=text,
                category="sports_recreation",
                subcategory="team_sports",
                sub_subcategory="basketball",
                detail="nba",
                metadata={"sport": "basketball"}
            )
        
        logger.info("\nTesting raw content prediction...")
        
        test_cases = [
            {
                "content_type": "text",
                "content": "The basketball game went into overtime",
                "expected_zone": "sports_recreation"
            },
            {
                "content_type": "image",
                "content": "test_data/basketball/game1.jpg",
                "expected_zone": "sports_recreation"
            },
            {
                "content_type": "text",
                "content": "A new quantum physics paper was published",
                "expected_zone": "science_nature"
            }
        ]
        
        for test_case in test_cases:
            await self.test_raw_content(
                content_type=test_case["content_type"],
                content=test_case["content"],
                expected_zone=test_case["expected_zone"]
            )
        
        await self._check_zone_learning_status()

    async def _check_zone_learning_status(self):
        """Check and log zone learning status."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/zone-learning-status")
                if response.status_code == 200:
                    zone_stats = response.json()
                    logger.info("\nZone Learning Status:")
                    for zone, stats in zone_stats.items():
                        logger.info(f"{zone}:")
                        logger.info(f"  Vectors: {stats['vector_count']}")
                        logger.info(f"  Has centroid: {stats['has_centroid']}")
                        logger.info(f"  Learning complete: {stats['learning_complete']}")
                    return zone_stats
            except Exception as e:
                logger.error(f"Error getting zone learning status: {e}")
                return None

    async def visualize_results(self):
        """Generate visualizations of test results."""
        logger.info("\nGenerating visualizations...")
        
        # Plot 3D coordinates
        if self.results:
            self.visualizer.plot_3d_coordinates(self.results)
        
        # Plot confidence distribution
        if self.confidences:
            self.visualizer.plot_confidence_distribution(
                self.confidences, 
                self.confidence_threshold
            )
        
        # Plot performance metrics
        if self.performance_metrics:
            self.visualizer.plot_performance_metrics(self.performance_metrics)
        
        # Get and plot zone learning progress
        zone_stats = await self._check_zone_learning_status()
        if zone_stats:
            self.visualizer.plot_zone_learning_progress(zone_stats)

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

    def _log_performance_summary(self):
        """Log performance metrics summary."""
        logger.info("\nPerformance Summary:")
        
        if self.performance_metrics['processing_time']:
            avg_time = np.mean(self.performance_metrics['processing_time'])
            logger.info(f"Average processing time: {avg_time:.3f}s")
        
        if self.confidences:
            avg_confidence = np.mean(self.confidences)
            logger.info(f"Average confidence: {avg_confidence:.3f}")
            below_threshold = sum(1 for c in self.confidences if c < self.confidence_threshold)
            logger.info(f"Predictions below threshold: {below_threshold}/{len(self.confidences)}")
        
        if 'accuracy' in self.performance_metrics:
            accuracy = np.mean(self.performance_metrics['accuracy'])
            logger.info(f"Overall accuracy: {accuracy:.3f}")

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
    
    # Test text processing with metadata (training phase)
    logger.info("\nPhase 1: Training with labeled data...")
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
    
    # Test image processing with metadata (training phase)
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
    
    # Test multimodal processing with metadata (training phase)
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

    # Phase 2: Testing prediction without metadata
    logger.info("\nPhase 2: Testing zone prediction capabilities...")
    await tester.test_zone_prediction()
    
    # Generate visualizations and performance metrics
    await tester.visualize_results()
    tester._log_performance_summary()
    
    logger.info(f"\nResults saved to: {tester.visualizer.output_dir}")
    logger.info("\nTest suite completed!")

if __name__ == "__main__":
    asyncio.run(run_tests())
