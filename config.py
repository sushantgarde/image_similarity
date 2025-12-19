import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Paths
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
DATA_FOLDER = BASE_DIR / 'data'
IMAGES_FOLDER = DATA_FOLDER / 'images'
EMBEDDINGS_FOLDER = DATA_FOLDER / 'embeddings'
MODEL_FOLDER = DATA_FOLDER / 'model'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, DATA_FOLDER, IMAGES_FOLDER, EMBEDDINGS_FOLDER, MODEL_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Flask Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Model Configuration
MODEL_NAME = 'efficientnet_v2_l'  # Options: resnet50, vgg16, efficientnet_b0, efficientnet_v2_l, convnext_large
FEATURE_DIM = 1280  # EfficientNetV2-L output dimension (2048 for ResNet50, 1280 for EfficientNetV2-L)
IMAGE_SIZE = (384, 384)  # Increased from (224, 224) for better accuracy

# Search Configuration
TOP_K = 10  # Number of similar images to return
SIMILARITY_THRESHOLD = 0.0  # Show all results, ranked by similarity

# PCA Configuration
USE_PCA = False  # Set to True to use PCA dimensionality reduction
PCA_COMPONENTS = 512  # Number of PCA components (only used if USE_PCA=True)

# Ensemble Configuration
USE_ENSEMBLE = False  # Set to True to use ensemble of models for maximum accuracy
ENSEMBLE_MODELS = ['efficientnet_v2_l', 'resnet50']  # Models to combine in ensemble

# Reranking Configuration
USE_RERANKING = True  # Use two-stage search with reranking for better accuracy
RERANKING_CANDIDATES = 50  # Number of candidates to retrieve before reranking

# Feature Files
EMBEDDINGS_FILE = EMBEDDINGS_FOLDER / 'features.npy'
IMAGE_PATHS_FILE = EMBEDDINGS_FOLDER / 'image_paths.pkl'
FAISS_INDEX_FILE = EMBEDDINGS_FOLDER / 'faiss_index.bin'
PCA_MODEL_FILE = EMBEDDINGS_FOLDER / 'pca_model.pkl'

# Device Configuration
USE_GPU = False  # Set to True if GPU available
DEVICE = 'cuda' if USE_GPU else 'cpu'

# Performance Tips:
# - USE_ENSEMBLE=True + USE_PCA=True + USE_RERANKING=True = Maximum Accuracy (slower)
# - USE_ENSEMBLE=False + USE_PCA=False + USE_RERANKING=True = Good balance
# - USE_ENSEMBLE=False + USE_PCA=False + USE_RERANKING=False = Fastest (lower accuracy)