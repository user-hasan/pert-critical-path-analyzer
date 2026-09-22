"""
Image preprocessing pipeline for the PERT & Critical Path Analyzer.

Provides robust, configurable preprocessing for diagram images from
various sources (screenshots, photos, scans, textbooks).

The pipeline produces multiple image representations to support
different downstream CV tasks while preserving coordinate mapping
between original and processed images.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pert_analyzer.cv.exceptions import (
    CVError,
    ImageFormatError,
    ImageLoadError,
    ImageSizeError,
    ImageValidationError,
    InvalidConfigError,
    PreprocessingError,
)
from pert_analyzer.cv.models import (
    BlurMethod,
    ContrastMethod,
    CoordinateMapping,
    EdgeMethod,
    PreprocessingConfig,
    PreprocessingResult,
    ResizeStrategy,
    ThresholdMethod,
)

logger = logging.getLogger(__name__)

# Supported image file extensions
SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".gif",
}

# Minimum valid image dimensions
MIN_IMAGE_DIMENSION = 10
MAX_IMAGE_DIMENSION = 20000


class ImagePreprocessor:
    """
    Image preprocessing pipeline for project network diagrams.

    This preprocessor handles images from various sources:
    - Clean screenshots
    - PowerPoint/Word screenshots
    - Textbook diagrams
    - Phone camera photos
    - Scanned images
    - Low-resolution or slightly rotated images

    The pipeline produces multiple representations optimized for
    different downstream tasks:
    - Shape detection
    - Arrow detection
    - OCR text extraction

    Usage:
        config = PreprocessingConfig()
        preprocessor = ImagePreprocessor(config)
        result = preprocessor.process(image_path)
        # or
        result = preprocessor.process_image(image_array)
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor with configuration.

        Args:
            config: Preprocessing configuration. Uses defaults if None.
        """
        self._config = config or PreprocessingConfig()
        self._validate_config()

    @property
    def config(self) -> PreprocessingConfig:
        """Get the current configuration."""
        return self._config

    def set_config(self, config: PreprocessingConfig) -> None:
        """Set a new configuration."""
        self._config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the current configuration."""
        issues = self._config.validate()
        if issues:
            raise InvalidConfigError(
                "preprocessing_config",
                str(issues),
                "; ".join(issues),
            )

    def process(self, image_path: str) -> PreprocessingResult:
        """
        Load and preprocess an image from a file path.

        Args:
            image_path: Path to the image file.

        Returns:
            PreprocessingResult with all representations.

        Raises:
            ImageLoadError: If the image cannot be loaded.
            ImageValidationError: If the image fails validation.
        """
        start_time = time.time()

        # Load image
        image = self._load_image(image_path)
        if image is None:
            raise ImageLoadError(image_path, "cv2.imread returned None")

        result = self.process_image(image)
        result.processing_time = time.time() - start_time
        return result

    def process_image(self, image: np.ndarray) -> PreprocessingResult:
        """
        Preprocess an already-loaded image.

        Args:
            image: Input image as numpy array (BGR or grayscale).

        Returns:
            PreprocessingResult with all representations.

        Raises:
            ImageValidationError: If the image fails validation.
        """
        start_time = time.time()

        # Validate input
        self._validate_image(image)

        # Create result container
        result = PreprocessingResult()

        # Store original
        if self._config.preserve_original:
            result.original = image.copy()

        # Get dimensions
        h, w = image.shape[:2]
        result.original_dimensions = (w, h)
        result.was_grayscale_input = len(image.shape) == 2

        # Step 1: Convert to grayscale if needed
        if len(image.shape) == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = image.copy()
        result.grayscale = grayscale

        # Step 2: Resize
        resized, was_resized = self._resize(grayscale)
        new_h, new_w = resized.shape[:2]

        # Build coordinate mapping
        if was_resized:
            result.coordinate_mapping = CoordinateMapping(
                original_width=w,
                original_height=h,
                processed_width=new_w,
                processed_height=new_h,
            )
            result.processed_dimensions = (new_w, new_h)
            result.scale_factor = new_w / w if w > 0 else 1.0
        else:
            result.coordinate_mapping = CoordinateMapping(
                original_width=w,
                original_height=h,
                processed_width=w,
                processed_height=h,
            )
            result.processed_dimensions = (w, h)
            result.scale_factor = 1.0

        # Step 3: Noise reduction
        denoised = self._reduce_noise(resized)
        result.denoised = denoised

        # Step 4: Contrast enhancement
        contrast_enhanced = self._enhance_contrast(denoised)
        result.contrast_enhanced = contrast_enhanced

        # Step 5: Binary threshold
        binary = self._apply_threshold(contrast_enhanced)
        result.binary = binary

        # Step 6: Adaptive threshold
        adaptive_binary = self._apply_adaptive_threshold(contrast_enhanced)
        result.adaptive_binary = adaptive_binary

        # Step 7: Edge detection
        edges = self._detect_edges(contrast_enhanced)
        result.edges = edges

        # Step 8: Deskew
        if self._config.enable_deskew:
            deskew_angle = self._estimate_skew_angle(binary)
            if abs(deskew_angle) > 0.1:
                if abs(deskew_angle) <= self._config.deskew_max_angle:
                    result.rotation_angle = deskew_angle
                    result.was_deskewed = True
                else:
                    result.warnings.append(
                        f"Detected rotation ({deskew_angle:.1f}deg) exceeds "
                        f"max angle ({self._config.deskew_max_angle}deg). "
                        f"Deskew skipped."
                    )

        result.processing_config = self._config
        result.processing_time = time.time() - start_time

        logger.info(
            "Preprocessing complete: %dx%d -> %dx%d, representations: %s",
            w, h,
            result.processed_dimensions[0], result.processed_dimensions[1],
            ", ".join(result.get_available_representations()),
        )

        return result

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """
        Load an image from disk.

        Args:
            path: File path to the image.

        Returns:
            Loaded image as numpy array, or None on failure.
        """
        try:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ImageLoadError(path, "cv2.imread returned None")
            return image
        except Exception as e:
            if isinstance(e, ImageLoadError):
                raise
            raise ImageLoadError(path, str(e))

    def _validate_image(self, image: np.ndarray) -> None:
        """
        Validate an image for preprocessing.

        Checks:
        - Not None
        - Has valid dimensions
        - Dimensions within bounds
        - Not empty

        Args:
            image: Image to validate.

        Raises:
            ImageValidationError: If validation fails.
        """
        if image is None:
            raise ImageValidationError("Image is None")

        if len(image.shape) not in (2, 3):
            raise ImageValidationError(
                f"Image has unexpected shape: {image.shape}",
                checks_failed=["valid_shape"],
            )

        h, w = image.shape[:2]

        if w == 0 or h == 0:
            raise ImageSizeError(w, h, "zero dimension")

        if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
            raise ImageSizeError(
                w, h,
                f"too small (minimum: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION})",
            )

        if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
            raise ImageSizeError(
                w, h,
                f"too large (maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})",
            )

    def _resize(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Resize image according to configuration.

        Preserves aspect ratio to avoid distorting the diagram.

        Args:
            image: Input grayscale image.

        Returns:
            Tuple of (resized_image, was_resized).
        """
        h, w = image.shape[:2]
        strategy = self._config.resize_strategy

        if strategy == ResizeStrategy.NONE:
            return image, False

        new_w, new_h = w, h
        should_resize = False

        if strategy == ResizeStrategy.MAX_DIMENSION:
            if w > self._config.max_width or h > self._config.max_height:
                scale = min(
                    self._config.max_width / w,
                    self._config.max_height / h,
                )
                new_w = int(w * scale)
                new_h = int(h * scale)
                should_resize = True

        elif strategy == ResizeStrategy.SCALE_FACTOR:
            if self._config.scale_factor != 1.0:
                new_w = int(w * self._config.scale_factor)
                new_h = int(h * self._config.scale_factor)
                should_resize = True

        elif strategy == ResizeStrategy.FIXED_SIZE:
            new_w = self._config.target_width
            new_h = self._config.target_height
            should_resize = True

        if not should_resize:
            return image, False

        # Ensure minimum dimensions
        new_w = max(new_w, self._config.min_width)
        new_h = max(new_h, self._config.min_height)

        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        logger.debug("Resized: %dx%d -> %dx%d", w, h, new_w, new_h)
        return resized, True

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce image noise using configured method.

        Methods:
        - Gaussian blur: Good general-purpose smoothing
        - Median blur: Better for salt-and-pepper noise
        - Bilateral filter: Preserves edges while smoothing

        Args:
            image: Input grayscale image.

        Returns:
            Denoised image.
        """
        method = self._config.blur_method

        if method == BlurMethod.NONE:
            return image

        if method == BlurMethod.GAUSSIAN:
            k = self._config.blur_kernel_size
            return cv2.GaussianBlur(image, (k, k), 0)

        if method == BlurMethod.MEDIAN:
            k = self._config.median_blur_kernel_size
            return cv2.medianBlur(image, k)

        if method == BlurMethod.BILATERAL:
            return cv2.bilateralFilter(
                image,
                self._config.bilateral_d,
                self._config.bilateral_sigma_color,
                self._config.bilateral_sigma_space,
            )

        logger.warning("Unknown blur method '%s', returning unchanged", method)
        return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using configured method.

        Methods:
        - Histogram equalization: Global contrast improvement
        - CLAHE: Contrast Limited Adaptive Histogram Equalization

        Args:
            image: Input grayscale image.

        Returns:
            Contrast-enhanced image.
        """
        method = self._config.contrast_method

        if method == ContrastMethod.NONE:
            return image

        if method == ContrastMethod.HISTOGRAM_EQUALIZATION:
            return cv2.equalizeHist(image)

        if method == ContrastMethod.CLAHE:
            clahe = cv2.createCLAHE(
                clipLimit=self._config.clahe_clip_limit,
                tileGridSize=self._config.clahe_tile_size,
            )
            return clahe.apply(image)

        logger.warning(
            "Unknown contrast method '%s', returning unchanged", method
        )
        return image

    def _apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply binary thresholding using configured method.

        Methods:
        - Global: Fixed threshold value
        - Otsu: Automatic threshold selection

        Args:
            image: Input grayscale image.

        Returns:
            Binary image (0 or 255).
        """
        method = self._config.threshold_method

        if method == ThresholdMethod.NONE:
            return image

        if method == ThresholdMethod.GLOBAL:
            _, binary = cv2.threshold(
                image,
                self._config.global_threshold_value,
                255,
                cv2.THRESH_BINARY,
            )
            return binary

        if method == ThresholdMethod.OTSU:
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary

        if method in (ThresholdMethod.ADAPTIVE_MEAN, ThresholdMethod.ADAPTIVE_GAUSSIAN):
            # Expected: adaptive methods are handled by the dedicated adaptive
            # threshold step; the global binary stays unchanged by design.
            logger.debug(
                "Adaptive threshold method '%s' is applied by the adaptive "
                "threshold step; global binary returned unchanged (expected)",
                method,
            )
        else:
            logger.warning(
                "Unknown threshold method '%s', returning unchanged", method
            )
        return image

    def _apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding using configured method.

        Methods:
        - Adaptive Mean: Local mean-based threshold
        - Adaptive Gaussian: Local Gaussian-weighted threshold

        Args:
            image: Input grayscale image.

        Returns:
            Adaptively thresholded binary image.
        """
        method = self._config.threshold_method
        block_size = self._config.adaptive_block_size
        c = self._config.adaptive_c

        if method == ThresholdMethod.ADAPTIVE_MEAN:
            return cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c,
            )

        if method == ThresholdMethod.ADAPTIVE_GAUSSIAN:
            return cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c,
            )

        # For other methods, use adaptive Gaussian as fallback
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using configured method.

        Methods:
        - Canny: Multi-stage edge detection
        - Sobel: Gradient-based edge detection

        Args:
            image: Input grayscale image.

        Returns:
            Edge-detected image.
        """
        method = self._config.edge_method

        if method == EdgeMethod.NONE:
            return image

        if method == EdgeMethod.CANNY:
            return cv2.Canny(
                image,
                self._config.canny_low_threshold,
                self._config.canny_high_threshold,
                apertureSize=self._config.canny_aperture_size,
            )

        if method == EdgeMethod.SOBEL:
            k = self._config.sobel_kernel_size
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=k)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=k)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            magnitude = np.uint8(np.clip(magnitude, 0, 255))
            return magnitude

        logger.warning(
            "Unknown edge method '%s', returning unchanged", method
        )
        return image

    def _estimate_skew_angle(self, binary_image: np.ndarray) -> float:
        """
        Estimate the skew angle of the image.

        Uses the Hough line transform to detect dominant line angles.

        Args:
            binary_image: Binary image for skew estimation.

        Returns:
            Estimated skew angle in degrees.
        """
        try:
            # Use Hough lines to detect dominant angle
            lines = cv2.HoughLinesP(
                binary_image,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=50,
                maxLineGap=10,
            )

            if lines is None or len(lines) < 2:
                return 0.0

            angles = []
            for line in lines:
                # HoughLinesP output shape varies by OpenCV version:
                # (N, 1, 4) on older builds, (N, 4) on newer ones.
                coords = np.asarray(line).reshape(-1)
                if coords.size != 4:
                    continue
                x1, y1, x2, y2 = (int(v) for v in coords)
                if x2 - x1 == 0:
                    continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines (within 30 degrees)
                if abs(angle) < 30:
                    angles.append(angle)

            if not angles:
                return 0.0

            # Use median to be robust against outliers
            return float(np.median(angles))

        except Exception as e:
            logger.debug("Skew estimation failed: %s", e)
            return 0.0

    def get_available_representations(self) -> List[str]:
        """Get list of representation names that can be produced."""
        return [
            "original",
            "normalized",
            "grayscale",
            "denoised",
            "contrast_enhanced",
            "binary",
            "adaptive_binary",
            "edges",
        ]
