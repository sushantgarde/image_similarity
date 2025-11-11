import os
import pickle
import logging
from pathlib import Path
from typing import List, Optional
from werkzeug.utils import secure_filename
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed.

    Args:
        filename: Name of the file

    Returns:
        True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def save_uploaded_file(file, upload_folder: Path = config.UPLOAD_FOLDER) -> Optional[str]:
    """
    Save uploaded file to disk.

    Args:
        file: File object from request
        upload_folder: Folder to save the file

    Returns:
        Path to saved file or None if failed
    """
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid conflicts
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"

            filepath = upload_folder / filename
            file.save(str(filepath))
            logger.info(f"File saved: {filepath}")
            return str(filepath)
        return None
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return None


def get_image_paths(image_folder: Path = config.IMAGES_FOLDER) -> List[str]:
    """
    Get all image paths from a folder.

    Args:
        image_folder: Folder containing images

    Returns:
        List of image paths
    """
    image_paths = []
    valid_extensions = config.ALLOWED_EXTENSIONS

    for ext in valid_extensions:
        image_paths.extend(image_folder.glob(f"*.{ext}"))
        image_paths.extend(image_folder.glob(f"*.{ext.upper()}"))

    image_paths = [str(p) for p in image_paths]
    logger.info(f"Found {len(image_paths)} images in {image_folder}")
    return sorted(image_paths)


def save_pickle(data, filepath: Path):
    """
    Save data to pickle file.

    Args:
        data: Data to save
        filepath: Path to save file
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving pickle file: {str(e)}")
        raise


def load_pickle(filepath: Path):
    """
    Load data from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded data
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle file: {str(e)}")
        raise


def ensure_dir(directory: Path):
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path
    """
    directory.mkdir(parents=True, exist_ok=True)


def clean_upload_folder(folder: Path = config.UPLOAD_FOLDER, max_age_hours: int = 24):
    """
    Clean old files from upload folder.

    Args:
        folder: Folder to clean
        max_age_hours: Maximum age of files in hours
    """
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    try:
        for filepath in folder.glob('*'):
            if filepath.is_file():
                file_age = current_time - filepath.stat().st_mtime
                if file_age > max_age_seconds:
                    filepath.unlink()
                    logger.info(f"Deleted old file: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning upload folder: {str(e)}")