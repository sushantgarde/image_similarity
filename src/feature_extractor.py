import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from typing import List, Union
import logging
from pathlib import Path
from tqdm import tqdm
from torchvision.models import ResNet50_Weights

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
            model_name: Name of pretrained model (resnet50, vgg16, efficientnet_b0, efficientnet_v2_l, convnext_large)
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
                model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                # Remove final classification layer
                model = nn.Sequential(*list(model.children())[:-1])

            elif self.model_name == 'vgg16':
                from torchvision.models import VGG16_Weights
                model = models.vgg16(weights=VGG16_Weights.DEFAULT)
                # Use features only, remove classifier
                model = model.features

            elif self.model_name == 'efficientnet_b0':
                from torchvision.models import EfficientNet_B0_Weights
                model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
                # Remove classifier
                model = nn.Sequential(*list(model.children())[:-1])

            elif self.model_name == 'efficientnet_v2_l':
                from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
                model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
                # Remove classifier
                model = nn.Sequential(*list(model.children())[:-1])

            elif self.model_name == 'convnext_large':
                from torchvision.models import convnext_large, ConvNeXt_Large_Weights
                model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
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


class EnsembleFeatureExtractor:
    """
    Combine features from multiple models for better accuracy.
    Uses multiple CNN models and concatenates their features.
    """

    def __init__(self, model_names: List[str] = None, device: str = config.DEVICE):
        """
        Initialize ensemble with multiple models.

        Args:
            model_names: List of model names to use
            device: Device to run models on
        """
        if model_names is None:
            model_names = config.ENSEMBLE_MODELS

        self.model_names = model_names
        self.device = device
        self.extractors = []

        logger.info(f"Initializing ensemble with models: {model_names}")

        for model_name in model_names:
            extractor = FeatureExtractor(model_name=model_name, device=device)
            self.extractors.append(extractor)

        logger.info(f"Ensemble initialized with {len(self.extractors)} models")

    def extract_features(self, image: Union[str, torch.Tensor]) -> np.ndarray:
        """
        Extract and combine features from all models.

        Args:
            image: Image path or preprocessed tensor

        Returns:
            Combined feature vector
        """
        try:
            features_list = []

            for extractor in self.extractors:
                feats = extractor.extract_features(image)
                features_list.append(feats)

            # Concatenate all features
            combined = np.concatenate(features_list)

            # Normalize combined features
            combined = combined / (np.linalg.norm(combined) + 1e-8)

            return combined

        except Exception as e:
            logger.error(f"Error extracting ensemble features: {str(e)}")
            raise

    def extract_features_batch(self, images: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Extract features from multiple images in batches.

        Args:
            images: List of image paths
            batch_size: Batch size for processing (reduced for ensemble)

        Returns:
            Feature matrix (num_images x combined_feature_dim)
        """
        all_features = []

        logger.info(f"Extracting ensemble features from {len(images)} images...")

        for i in tqdm(range(0, len(images), batch_size), desc="Extracting ensemble features"):
            batch_paths = images[i:i + batch_size]

            try:
                batch_features = []

                for path in batch_paths:
                    feats = self.extract_features(path)
                    batch_features.append(feats)

                batch_features = np.vstack(batch_features)
                all_features.append(batch_features)

            except Exception as e:
                logger.error(f"Error processing batch {i}: {str(e)}")
                continue

        if all_features:
            all_features = np.vstack(all_features)
            logger.info(f"Extracted ensemble features shape: {all_features.shape}")
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
            logger.info(f"Ensemble features saved to {save_path}")
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
            logger.info(f"Ensemble features loaded from {load_path}")
            return features
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise