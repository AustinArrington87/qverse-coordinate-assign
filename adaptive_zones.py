from sklearn.cluster import DBSCAN
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AdaptiveZones:
    def __init__(self, base_zones: Dict, min_samples: int = 5):
        self.base_zones = base_zones
        self.vectors: Dict[str, List[List[float]]] = {}
        self.clusters: Dict[str, Dict] = {}
        self.min_samples = min_samples
        self.last_update = None
        
        # Load existing vectors if available
        self.load_state()
    
    def save_state(self):
        """Save the current state of vectors and clusters."""
        state = {
            "vectors": self.vectors,
            "clusters": self.clusters,
            "last_update": self.last_update
        }
        
        try:
            state_file = Path("data/adaptive_zones_state.json")
            state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump(state, f)
            logger.info("Adaptive zones state saved successfully")
        except Exception as e:
            logger.error(f"Error saving adaptive zones state: {e}")
    
    def load_state(self):
        """Load previously saved state."""
        try:
            state_file = Path("data/adaptive_zones_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                self.vectors = state.get("vectors", {})
                self.clusters = state.get("clusters", {})
                self.last_update = state.get("last_update")
                logger.info("Adaptive zones state loaded successfully")
        except Exception as e:
            logger.error(f"Error loading adaptive zones state: {e}")

    def normalize_vector_length(self, vectors: List[List[float]]) -> np.ndarray:
        """Normalize vectors to have the same length."""
        if not vectors:
            return np.array([])
            
        # Find minimum length among all vectors
        min_length = min(len(v) for v in vectors)
        
        # Truncate all vectors to minimum length
        normalized_vectors = [v[:min_length] for v in vectors]
        
        return np.array(normalized_vectors, dtype=np.float32)
    
    def add_vector(self, category: str, vector: List[float], force_update: bool = False):
        """Add a vector and optionally update clusters."""
        try:
            if category not in self.vectors:
                self.vectors[category] = []
            
            self.vectors[category].append(vector)
            
            # Update clusters if we have enough samples or force update is requested
            if len(self.vectors[category]) >= self.min_samples or force_update:
                self.update_clusters(categories=[category])
                self.save_state()
        except Exception as e:
            logger.error(f"Error adding vector: {e}")
            raise
    
    def update_clusters(self, categories: Optional[List[str]] = None):
        """Update clusters for specified categories or all categories."""
        update_cats = categories or list(self.vectors.keys())
        
        for category in update_cats:
            try:
                if category in self.vectors and len(self.vectors[category]) >= self.min_samples:
                    # Normalize vector lengths
                    vectors = self.normalize_vector_length(self.vectors[category])
                    
                    if len(vectors) == 0:
                        continue
                    
                    # Use DBSCAN for clustering
                    clustering = DBSCAN(eps=0.5, min_samples=self.min_samples)
                    clusters = clustering.fit(vectors)
                    
                    # Calculate cluster centers and ranges
                    unique_labels = set(clusters.labels_)
                    cluster_info = {}
                    
                    for label in unique_labels:
                        if label != -1:  # Ignore noise points
                            mask = clusters.labels_ == label
                            cluster_points = vectors[mask]
                            
                            center = np.mean(cluster_points, axis=0)
                            ranges = np.ptp(cluster_points, axis=0)
                            
                            cluster_info[f"cluster_{label}"] = {
                                "center": center.tolist(),
                                "ranges": ranges.tolist(),
                                "size": int(np.sum(mask))
                            }
                    
                    self.clusters[category] = cluster_info
            except Exception as e:
                logger.error(f"Error updating clusters for category {category}: {e}")
                continue
        
        self.last_update = datetime.now().isoformat()
        logger.info(f"Clusters updated for categories: {update_cats}")
    
    def get_adaptive_zones(self) -> Dict:
        """Generate adaptive zones based on clusters."""
        try:
            adaptive_zones = self.base_zones.copy()
            
            for category, cluster_info in self.clusters.items():
                if category in adaptive_zones:
                    # Update zone ranges based on clusters
                    if cluster_info:  # Only update if we have clusters
                        cluster_ranges = [info["ranges"] for info in cluster_info.values()]
                        if cluster_ranges:
                            total_range = np.sum(cluster_ranges, axis=0).tolist()
                            
                            adaptive_zones[category].update({
                                "adaptive_ranges": total_range,
                                "clusters": cluster_info,
                                "last_update": self.last_update,
                                "sample_size": len(self.vectors.get(category, []))
                            })
            
            return adaptive_zones
        except Exception as e:
            logger.error(f"Error generating adaptive zones: {e}")
            return self.base_zones.copy()
    
    def get_category_stats(self, category: str) -> Dict:
        """Get statistics for a specific category."""
        try:
            if category not in self.vectors:
                return {}
            
            vectors = self.vectors[category]
            return {
                "sample_size": len(vectors),
                "clusters": len(self.clusters.get(category, {})),
                "last_update": self.last_update,
                "has_sufficient_samples": len(vectors) >= self.min_samples,
                "vector_dimensions": len(vectors[0]) if vectors else 0
            }
        except Exception as e:
            logger.error(f"Error getting category stats: {e}")
            return {}

# Import zones and initialize manager
from zones import zones
adaptive_zones_manager = AdaptiveZones(base_zones=zones)
