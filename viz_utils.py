import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class QVerseVisualizer:
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.style.use('seaborn')
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
