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
MODEL_NAME = 'resnet50'  # Options: resnet50, vgg16, efficientnet_b0
FEATURE_DIM = 2048  # ResNet50 output dimension
IMAGE_SIZE = (224, 224)

# Search Configuration
TOP_K = 10  # Number of similar images to return
SIMILARITY_THRESHOLD = 0.0  # Changed from 0.5 to 0.0 - show all results, ranked by similarity

# Feature Files
EMBEDDINGS_FILE = EMBEDDINGS_FOLDER / 'features.npy'
IMAGE_PATHS_FILE = EMBEDDINGS_FOLDER / 'image_paths.pkl'
FAISS_INDEX_FILE = EMBEDDINGS_FOLDER / 'faiss_index.bin'

# Device Configuration
USE_GPU = False  # Set to True if GPU available
DEVICE = 'cuda' if USE_GPU else 'cpu'