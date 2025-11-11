from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import logging

import config
from src.feature_extractor import FeatureExtractor
from src.similarity_search import SimilaritySearch
from src.utils import allowed_file, save_uploaded_file, get_image_paths, clean_upload_folder
from src.preprocessing import load_and_validate_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# Initialize feature extractor and similarity search
feature_extractor = FeatureExtractor()
similarity_search = SimilaritySearch(use_faiss=True)

# Global flag to check if index is loaded
index_loaded = False


def initialize_search_index():
    """
    Initialize or load the search index.
    """
    global index_loaded

    try:
        # Check if pre-computed index exists
        if (config.FAISS_INDEX_FILE.exists() and
                config.IMAGE_PATHS_FILE.exists() and
                config.EMBEDDINGS_FILE.exists()):

            logger.info("Loading existing search index...")
            similarity_search.load_index()
            index_loaded = True
            logger.info("Search index loaded successfully")

        else:
            logger.info("No existing index found. Building new index...")

            # Get all images from dataset
            image_paths = get_image_paths(config.IMAGES_FOLDER)

            if len(image_paths) == 0:
                logger.warning("No images found in dataset folder")
                return False

            # Extract features
            features = feature_extractor.extract_features_batch(image_paths)

            # Build search index
            similarity_search.build_index(features, image_paths)

            # Save index
            similarity_search.save_index()

            index_loaded = True
            logger.info(f"Search index built for {len(image_paths)} images")

        return True

    except Exception as e:
        logger.error(f"Error initializing search index: {str(e)}")
        return False


@app.route('/')
def index():
    """
    Home page with upload form.
    """
    stats = similarity_search.get_statistics() if index_loaded else None
    return render_template('index.html', stats=stats, index_loaded=index_loaded)


@app.route('/search', methods=['POST'])
def search():
    """
    Handle image upload and search.
    """
    logger.info("=" * 50)
    logger.info("NEW SEARCH REQUEST RECEIVED")
    logger.info("=" * 50)

    if not index_loaded:
        logger.error("Search index not initialized")
        flash('Search index not initialized. Please wait or check logs.', 'error')
        return redirect(url_for('index'))

    # Check if file was uploaded
    if 'file' not in request.files:
        logger.error("No file in request")
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    logger.info(f"File received: {file.filename}")

    # Check if file is selected
    if file.filename == '':
        logger.error("Empty filename")
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    # Validate and save file
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            logger.info("Saving uploaded file...")
            filepath = save_uploaded_file(file)

            if filepath is None:
                logger.error("Failed to save file")
                flash('Error saving file', 'error')
                return redirect(url_for('index'))

            logger.info(f"File saved to: {filepath}")

            # Validate image
            logger.info("Validating image...")
            load_and_validate_image(filepath)
            logger.info("Image validated successfully")

            # Extract features from uploaded image
            logger.info(f"Extracting features from {filepath}")
            query_features = feature_extractor.extract_features(filepath)
            logger.info(f"Features extracted. Shape: {query_features.shape}")

            # Get top_k from form or use default
            top_k = int(request.form.get('top_k', config.TOP_K))
            top_k = min(max(top_k, 1), 50)  # Limit between 1 and 50
            logger.info(f"Searching for top {top_k} similar images")

            # Search for similar images
            results = similarity_search.search(query_features, top_k=top_k)
            logger.info(f"Search completed. Found {len(results)} results")

            # DEBUG: Log top 5 similarity scores
            if len(results) > 0:
                logger.info("--- Top Similarity Scores ---")
                for idx, (path, score) in enumerate(results[:5]):
                    logger.info(f"  Top {idx + 1}: score={score:.4f} ({score * 100:.2f}%)")
            else:
                logger.warning("No results returned from search!")

            # Check if no results found
            if len(results) == 0:
                logger.warning("No similar images found")
                return render_template('result.html',
                                       query_image=os.path.basename(filepath),
                                       similar_images=[],
                                       num_results=0)

            # Get correct relative path for query image (just filename, not uploads/)
            query_image = os.path.basename(filepath)
            logger.info(f"Query image filename: {query_image}")

            similar_images = []
            for idx, (img_path, score) in enumerate(results):
                logger.info(f"  Result {idx + 1}: {img_path} (score: {score:.4f})")

                # Convert absolute path to relative web path
                if os.path.isabs(img_path):
                    rel_path = os.path.relpath(img_path, config.BASE_DIR)
                else:
                    rel_path = img_path

                # Convert backslashes to forward slashes for web
                rel_path = rel_path.replace('\\', '/')
                logger.info(f"    Relative path: {rel_path}")

                similar_images.append({
                    'path': rel_path,
                    'score': round(score * 100, 2),
                    'filename': os.path.basename(img_path)
                })

            logger.info(f"Rendering result page with {len(similar_images)} images")
            logger.info("=" * 50)

            return render_template('result.html',
                                   query_image=query_image,
                                   similar_images=similar_images,
                                   num_results=len(similar_images))

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('error.html',
                                   error_code='Processing Error',
                                   error_message=str(e))

    else:
        logger.error(f"Invalid file type: {file.filename}")
        flash('Invalid file type. Allowed types: ' + ', '.join(config.ALLOWED_EXTENSIONS), 'error')
        return redirect(url_for('index'))


@app.route('/data/<path:filename>')
def serve_data_file(filename):
    """
    Serve files from data directory.
    """
    try:
        logger.info(f"Serving data file: {filename}")
        return send_from_directory(config.DATA_FOLDER, filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return "File not found", 404


@app.route('/rebuild-index', methods=['POST'])
def rebuild_index():
    """
    Rebuild the search index.
    """
    try:
        logger.info("Rebuilding search index...")
        success = initialize_search_index()

        if success:
            flash('Search index rebuilt successfully', 'success')
        else:
            flash('Error rebuilding search index', 'error')

    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        flash(f'Error: {str(e)}', 'error')

    return redirect(url_for('index'))


@app.route('/stats')
def stats():
    """
    Get search index statistics as JSON.
    """
    if index_loaded:
        stats = similarity_search.get_statistics()
        return jsonify(stats)
    else:
        return jsonify({'error': 'Index not loaded'}), 503


@app.route('/health')
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'index_loaded': index_loaded
    })


@app.errorhandler(413)
def too_large(e):
    """
    Handle file too large error.
    """
    flash('File is too large. Maximum size is 16MB', 'error')
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(e):
    """
    Handle 404 errors.
    """
    return render_template('error.html',
                           error_code=404,
                           error_message='Page not found'), 404


@app.errorhandler(500)
def internal_error(e):
    """
    Handle 500 errors.
    """
    logger.error(f"Internal server error: {str(e)}")
    return render_template('error.html',
                           error_code=500,
                           error_message='Internal server error'), 500


if __name__ == '__main__':
    # Clean old uploads on startup
    logger.info("Cleaning old upload files...")
    clean_upload_folder()

    # Initialize search index
    logger.info("Initializing search index...")
    initialize_search_index()

    # Run app
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)