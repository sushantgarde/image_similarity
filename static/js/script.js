// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadPage();
});

// Initialize upload page functionality
function initializeUploadPage() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadContent = document.getElementById('uploadContent');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const filename = document.getElementById('filename');
    const removeBtn = document.getElementById('removeBtn');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');

    if (!uploadArea || !fileInput) return; // Not on upload page

    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    });

    // Remove button
    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            removeFile();
        });
    }

    // Form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }

    // Handle file selection
    function handleFileSelect() {
        const file = fileInput.files[0];

        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            showError('Please select a valid image file');
            fileInput.value = '';
            return;
        }

        // Validate file size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            showError('File size must be less than 10MB');
            fileInput.value = '';
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            filename.textContent = file.name;

            // Hide upload content, show preview
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');

            // Enable submit button
            submitBtn.disabled = false;

            // Add animation
            previewContainer.style.animation = 'zoomIn 0.4s ease-out';
        };
        reader.readAsDataURL(file);
    }

    // Remove file
    function removeFile() {
        fileInput.value = '';
        imagePreview.src = '';
        filename.textContent = '';

        // Show upload content, hide preview
        uploadContent.classList.remove('hidden');
        previewContainer.classList.add('hidden');

        // Disable submit button
        submitBtn.disabled = true;
    }

    // Handle form submission
    function handleFormSubmit(e) {
        if (!fileInput.files[0]) {
            e.preventDefault();
            showError('Please select an image first');
            return;
        }

        // Show loading state
        const btnText = submitBtn.querySelector('.btn-text');
        const btnLoader = submitBtn.querySelector('.btn-loader');

        if (btnText && btnLoader) {
            btnText.classList.add('hidden');
            btnLoader.classList.remove('hidden');
            submitBtn.disabled = true;
        }

        // Add a slight delay for better UX
        setTimeout(() => {
            // Form will submit naturally
        }, 300);
    }

    // Show error message
    function showError(message) {
        // Create error toast
        const toast = document.createElement('div');
        toast.className = 'error-toast';
        toast.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z" fill="currentColor"/>
            </svg>
            <span>${message}</span>
        `;

        document.body.appendChild(toast);

        // Animate in
        setTimeout(() => toast.classList.add('show'), 10);

        // Remove after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Add CSS for error toast dynamically
if (!document.getElementById('toast-styles')) {
    const style = document.createElement('style');
    style.id = 'toast-styles';
    style.textContent = `
        .error-toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #f56565;
            color: white;
            padding: 16px 24px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 9999;
            max-width: 400px;
        }

        .error-toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        .error-toast svg {
            width: 24px;
            height: 24px;
            flex-shrink: 0;
        }

        .error-toast span {
            font-weight: 500;
            line-height: 1.4;
        }

        @media (max-width: 768px) {
            .error-toast {
                bottom: 20px;
                right: 20px;
                left: 20px;
                max-width: calc(100% - 40px);
            }
        }
    `;
    document.head.appendChild(style);
}

// Lazy loading for images on results page
function initializeLazyLoading() {
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src || img.src;
                    img.classList.add('loaded');
                    observer.unobserve(img);
                }
            });
        });

        const images = document.querySelectorAll('img[loading="lazy"]');
        images.forEach(img => imageObserver.observe(img));
    }
}

// Initialize lazy loading if on results page
if (document.querySelector('.results-grid')) {
    initializeLazyLoading();
}

// Add smooth scroll behavior
document.documentElement.style.scrollBehavior = 'smooth';

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // ESC to close modal
    if (e.key === 'Escape') {
        const modal = document.querySelector('.image-modal.active');
        if (modal) {
            modal.classList.remove('active');
            setTimeout(() => modal.remove(), 300);
        }
    }

    // Ctrl/Cmd + U to trigger file upload
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.click();
        }
    }
});

// Add progress indicator for form submission
window.addEventListener('beforeunload', (e) => {
    const form = document.getElementById('uploadForm');
    if (form && form.classList.contains('submitting')) {
        e.preventDefault();
        e.returnValue = '';
    }
});

// Performance monitoring (optional)
if (window.performance && window.performance.timing) {
    window.addEventListener('load', () => {
        const perfData = window.performance.timing;
        const loadTime = perfData.loadEventEnd - perfData.navigationStart;
        console.log(`Page loaded in ${loadTime}ms`);
    });
}

// Service Worker registration for offline support (optional)
if ('serviceWorker' in navigator && location.protocol === 'https:') {
    window.addEventListener('load', () => {
        // Uncomment to enable service worker
        // navigator.serviceWorker.register('/sw.js')
        //     .then(reg => console.log('Service Worker registered'))
        //     .catch(err => console.log('Service Worker registration failed'));
    });
}

// Add touch feedback for mobile
if ('ontouchstart' in window) {
    document.addEventListener('touchstart', function() {}, { passive: true });
}

// Optimize animations on low-end devices
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
if (reducedMotion.matches) {
    document.documentElement.style.setProperty('--transition', 'none');
}

// Network status indicator
function updateOnlineStatus() {
    if (!navigator.onLine) {
        showNetworkToast('You are offline. Some features may not work.');
    }
}

window.addEventListener('online', () => {
    showNetworkToast('Back online!', 'success');
});

window.addEventListener('offline', updateOnlineStatus);

function showNetworkToast(message, type = 'error') {
    const toast = document.createElement('div');
    toast.className = `network-toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Add CSS for network toast
if (!document.getElementById('network-toast-styles')) {
    const style = document.createElement('style');
    style.id = 'network-toast-styles';
    style.textContent = `
        .network-toast {
            position: fixed;
            top: 30px;
            right: 30px;
            padding: 12px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateY(-100px);
            opacity: 0;
            transition: all 0.3s ease;
            z-index: 9999;
        }

        .network-toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        .network-toast.error {
            background: #f56565;
        }

        .network-toast.success {
            background: #48bb78;
        }

        @media (max-width: 768px) {
            .network-toast {
                top: 20px;
                right: 20px;
                left: 20px;
            }
        }
    `;
    document.head.appendChild(style);
}