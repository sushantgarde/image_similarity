import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from typing import List, Union
import logging
from pathlib import Path
from tqdm import tqdm

import config
from .preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract deep features from images using pretrained CNN models.
    """

    def __init__(self, model_name: str = config.MODEL_NAME, device: str = config.DEVICE):
        """
        Initialize feature extractor.

        Args:
            model_name: Name of pretrained model (resnet50, vgg16, efficientnet_b0)
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = self._load_model()
        self.preprocessor = ImagePreprocessor()

        logger.info(f"FeatureExtractor initialized with {model_name} on {device}")

    def _load_model(self) -> nn.Module:
        """
        Load pretrained model and remove classification head.

        Returns:
            Modified model for feature extraction
        """
        try:
            if self.model_name == 'resnet50':
                model = models.resnet50(pretrained=True)
                # Remove final classification layer
                model = nn.Sequential(*list(model.children())[:-1])

            elif self.model_name == 'vgg16':
                model = models.vgg16(pretrained=True)
                # Use features only, remove classifier
                model = model.features

            elif self.model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=True)
                # Remove classifier
                model = nn.Sequential(*list(model.children())[:-1])

            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            # Set to evaluation mode
            model.eval()
            model.to(self.device)

            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False

            logger.info(f"Model {self.model_name} loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def extract_features(self, image: Union[str, torch.Tensor]) -> np.ndarray:
        """
        Extract features from a single image.

        Args:
            image: Image path or preprocessed tensor

        Returns:
            Feature vector as numpy array
        """
        try:
            # Preprocess if image path is provided
            if isinstance(image, str):
                tensor = self.preprocessor.preprocess(image)
            else:
                tensor = image

            # Move to device
            tensor = tensor.to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model(tensor)

            # Flatten and convert to numpy
            features = features.squeeze().cpu().numpy()

            # Normalize features
            features = features / (np.linalg.norm(features) + 1e-8)

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def extract_features_batch(self, images: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract features from multiple images in batches.

        Args:
            images: List of image paths
            batch_size: Batch size for processing

        Returns:
            Feature matrix (num_images x feature_dim)
        """
        all_features = []

        logger.info(f"Extracting features from {len(images)} images...")

        # Process in batches
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch_paths = images[i:i + batch_size]

            try:
                # Preprocess batch
                batch_tensors = []
                for path in batch_paths:
                    tensor = self.preprocessor.preprocess(path)
                    batch_tensors.append(tensor)

                batch = torch.cat(batch_tensors, dim=0).to(self.device)

                # Extract features
                with torch.no_grad():
                    features = self.model(batch)

                # Process features
                features = features.squeeze().cpu().numpy()

                # Handle single image case
                if len(batch_paths) == 1:
                    features = features.reshape(1, -1)

                # Normalize each feature vector
                norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
                features = features / norms

                all_features.append(features)

            except Exception as e:
                logger.error(f"Error processing batch {i}: {str(e)}")
                # Skip failed images and continue
                continue

        # Concatenate all features
        if all_features:
            all_features = np.vstack(all_features)
            logger.info(f"Extracted features shape: {all_features.shape}")
            return all_features
        else:
            raise ValueError("No features could be extracted")

    def save_features(self, features: np.ndarray, save_path: Path):
        """
        Save extracted features to disk.

        Args:
            features: Feature matrix
            save_path: Path to save features
        """
        try:
            np.save(save_path, features)
            logger.info(f"Features saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

    def load_features(self, load_path: Path) -> np.ndarray:
        """
        Load features from disk.

        Args:
            load_path: Path to load features from

        Returns:
            Feature matrix
        """
        try:
            features = np.load(load_path)
            logger.info(f"Features loaded from {load_path}")
            return features
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise