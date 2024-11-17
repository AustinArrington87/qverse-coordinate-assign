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
        # Add centroids for each zone
        self.zone_centroids: Dict[str, np.ndarray] = {}
        
        # Load existing state
        self.load_state()
    
    def save_state(self):
        """Save the current state of vectors, clusters, and centroids."""
        state = {
            "vectors": self.vectors,
            "clusters": self.clusters,
            "last_update": self.last_update,
            "zone_centroids": {
                zone: centroid.tolist() if isinstance(centroid, np.ndarray) else centroid
                for zone, centroid in self.zone_centroids.items()
            }
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
                
                # Load centroids and convert to numpy arrays
                centroids = state.get("zone_centroids", {})
                self.zone_centroids = {
                    zone: np.array(centroid) if centroid is not None else None
                    for zone, centroid in centroids.items()
                }
                logger.info("Adaptive zones state loaded successfully")
        except Exception as e:
            logger.error(f"Error loading adaptive zones state: {e}")

    def predict_zone(self, vector: List[float]) -> Tuple[str, float]:
        """Predict the most likely zone for a vector based on learned patterns."""
        vector_np = np.array(vector, dtype=np.float32)
        best_zone = None
        best_similarity = -1
        
        # Compare with each zone's centroid
        for zone, centroid in self.zone_centroids.items():
            if centroid is not None:
                similarity = self.calculate_similarity(vector_np, centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_zone = zone
        
        # If no prediction possible, use first zone as default
        if best_zone is None and self.base_zones:
            best_zone = next(iter(self.base_zones.keys()))
            best_similarity = 0.0
            
        return best_zone, best_similarity
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        # Ensure same dimension
        min_dim = min(len(vec1), len(vec2))
        vec1, vec2 = vec1[:min_dim], vec2[:min_dim]
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)

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
        """Add a vector and update zone learning."""
        try:
            if category not in self.vectors:
                self.vectors[category] = []
            
            self.vectors[category].append(vector)
            
            # Update clusters and centroids if enough samples
            if len(self.vectors[category]) >= self.min_samples or force_update:
                self.update_clusters(categories=[category])
                self.update_zone_centroids()
                self.save_state()
            
            # Predict zone for verification
            predicted_zone, confidence = self.predict_zone(vector)
            logger.info(f"Vector added to {category}, predicted zone: {predicted_zone} "
                       f"(confidence: {confidence:.2f})")
        
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

    def update_zone_centroids(self):
        """Update centroids for each zone based on existing vectors."""
        for zone in self.vectors.keys():
            if len(self.vectors[zone]) > 0:
                try:
                    # Calculate zone centroid from all vectors in zone
                    vectors = self.normalize_vector_length(self.vectors[zone])
                    self.zone_centroids[zone] = np.mean(vectors, axis=0)
                    logger.info(f"Updated centroid for zone {zone} using {len(vectors)} vectors")
                except Exception as e:
                    logger.error(f"Error updating centroid for zone {zone}: {e}")
    
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
                                "sample_size": len(self.vectors.get(category, [])),
                                "has_centroid": category in self.zone_centroids
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
                "has_centroid": category in self.zone_centroids,
                "vector_dimensions": len(vectors[0]) if vectors else 0
            }
        except Exception as e:
            logger.error(f"Error getting category stats: {e}")
            return {}

    def get_learning_metrics(self) -> Dict:
        """Get metrics about the learning progress."""
        try:
            return {
                "total_vectors": sum(len(v) for v in self.vectors.values()),
                "zones_with_centroids": sum(1 for c in self.zone_centroids.values() if c is not None),
                "total_clusters": sum(len(c) for c in self.clusters.values()),
                "learning_status": {
                    zone: {
                        "vectors": len(self.vectors.get(zone, [])),
                        "has_centroid": zone in self.zone_centroids,
                        "clusters": len(self.clusters.get(zone, {})),
                        "ready": len(self.vectors.get(zone, [])) >= self.min_samples
                    }
                    for zone in self.base_zones.keys()
                }
            }
        except Exception as e:
            logger.error(f"Error getting learning metrics: {e}")
            return {}

# Import zones and initialize manager
from zones import zones
adaptive_zones_manager = AdaptiveZones(base_zones=zones)
