import numpy as np
import faiss
from typing import List, Tuple, Optional
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

import config
from .utils import load_pickle, save_pickle

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """
    Search for similar images using FAISS or cosine similarity.
    """

    def __init__(self, use_faiss: bool = True):
        """
        Initialize similarity search.

        Args:
            use_faiss: Whether to use FAISS for search (faster for large datasets)
        """
        self.use_faiss = use_faiss
        self.index = None
        self.features = None
        self.image_paths = None
        self.is_built = False

        logger.info(f"SimilaritySearch initialized (FAISS: {use_faiss})")

    def build_index(self, features: np.ndarray, image_paths: List[str]):
        """
        Build search index from features.

        Args:
            features: Feature matrix (num_images x feature_dim)
            image_paths: List of corresponding image paths
        """
        try:
            if len(features) != len(image_paths):
                raise ValueError("Number of features must match number of image paths")

            self.features = features.astype('float32')
            self.image_paths = image_paths

            if self.use_faiss:
                # Build FAISS index
                dimension = features.shape[1]

                # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
                self.index = faiss.IndexFlatIP(dimension)

                # Add features to index
                self.index.add(self.features)

                logger.info(f"FAISS index built with {self.index.ntotal} vectors")

            self.is_built = True
            logger.info(f"Search index built successfully for {len(image_paths)} images")

        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise

    def search(self, query_features: np.ndarray, top_k: int = config.TOP_K,
               threshold: float = config.SIMILARITY_THRESHOLD) -> List[Tuple[str, float]]:
        """
        Search for similar images.

        Args:
            query_features: Query feature vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (image_path, similarity_score) tuples
        """
        if not self.is_built:
            raise ValueError("Index not built. Call build_index() first.")

        try:
            query_features = query_features.astype('float32').reshape(1, -1)

            if self.use_faiss:
                # Search using FAISS
                similarities, indices = self.index.search(query_features, top_k)
                similarities = similarities[0]
                indices = indices[0]
            else:
                # Use cosine similarity
                similarities = cosine_similarity(query_features, self.features)[0]
                # Get top k indices
                indices = np.argsort(similarities)[::-1][:top_k]
                similarities = similarities[indices]

            # Filter by threshold and prepare results
            results = []
            for idx, score in zip(indices, similarities):
                if score >= threshold:
                    results.append((self.image_paths[idx], float(score)))

            logger.info(f"Found {len(results)} similar images above threshold {threshold}")
            return results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def save_index(self, index_path: Path = config.FAISS_INDEX_FILE,
                   paths_path: Path = config.IMAGE_PATHS_FILE):
        """
        Save FAISS index and image paths to disk.

        Args:
            index_path: Path to save FAISS index
            paths_path: Path to save image paths
        """
        try:
            if not self.is_built:
                raise ValueError("Index not built. Cannot save.")

            if self.use_faiss and self.index is not None:
                faiss.write_index(self.index, str(index_path))
                logger.info(f"FAISS index saved to {index_path}")

            # Save features and image paths
            if self.features is not None:
                np.save(config.EMBEDDINGS_FILE, self.features)
                logger.info(f"Features saved to {config.EMBEDDINGS_FILE}")

            if self.image_paths is not None:
                save_pickle(self.image_paths, paths_path)
                logger.info(f"Image paths saved to {paths_path}")

        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

    def load_index(self, index_path: Path = config.FAISS_INDEX_FILE,
                   paths_path: Path = config.IMAGE_PATHS_FILE,
                   features_path: Path = config.EMBEDDINGS_FILE):
        """
        Load FAISS index and image paths from disk.

        Args:
            index_path: Path to FAISS index
            paths_path: Path to image paths
            features_path: Path to features
        """
        try:
            # Load features
            if features_path.exists():
                self.features = np.load(features_path).astype('float32')
                logger.info(f"Features loaded from {features_path}")

            # Load FAISS index
            if self.use_faiss and index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"FAISS index loaded from {index_path}")

            # Load image paths
            if paths_path.exists():
                self.image_paths = load_pickle(paths_path)
                logger.info(f"Image paths loaded from {paths_path}")

            if self.features is not None and self.image_paths is not None:
                self.is_built = True
                logger.info("Search index loaded successfully")
            else:
                logger.warning("Index not fully loaded")

        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise

    def get_statistics(self) -> dict:
        """
        Get statistics about the search index.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'is_built': self.is_built,
            'num_images': len(self.image_paths) if self.image_paths else 0,
            'feature_dim': self.features.shape[1] if self.features is not None else 0,
            'use_faiss': self.use_faiss
        }

        if self.use_faiss and self.index is not None:
            stats['faiss_total'] = self.index.ntotal

        return stats


def compute_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """
    Compute pairwise similarity matrix for all features.

    Args:
        features: Feature matrix (num_images x feature_dim)

    Returns:
        Similarity matrix (num_images x num_images)
    """
    return cosine_similarity(features)


def find_duplicates(features: np.ndarray, image_paths: List[str],
                    threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """
    Find near-duplicate images based on high similarity.

    Args:
        features: Feature matrix
        image_paths: List of image paths
        threshold: Similarity threshold for duplicates

    Returns:
        List of (image1, image2, similarity) tuples
    """
    similarity_matrix = compute_similarity_matrix(features)

    duplicates = []
    n = len(image_paths)

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                duplicates.append((
                    image_paths[i],
                    image_paths[j],
                    float(similarity_matrix[i, j])
                ))

    logger.info(f"Found {len(duplicates)} potential duplicates")
    return duplicates


def batch_search(searcher: SimilaritySearch, query_features_list: List[np.ndarray],
                 top_k: int = config.TOP_K) -> List[List[Tuple[str, float]]]:
    """
    Perform batch search for multiple queries.

    Args:
        searcher: SimilaritySearch instance
        query_features_list: List of query feature vectors
        top_k: Number of results per query

    Returns:
        List of results for each query
    """
    results = []
    for query_features in query_features_list:
        result = searcher.search(query_features, top_k)
        results.append(result)

    return results