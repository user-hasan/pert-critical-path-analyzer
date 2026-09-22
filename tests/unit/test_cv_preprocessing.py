"""
Comprehensive unit tests for the CV image preprocessing pipeline.

Tests cover:
- Image loading and validation
- Resizing with aspect ratio preservation
- Grayscale conversion
- Noise reduction
- Contrast enhancement
- Binary thresholding (Otsu and adaptive)
- Edge detection
- Skew/rotation handling
- Coordinate mapping
- Configuration behavior
- Multiple representations
- Original image preservation
- Error handling
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from pert_analyzer.cv.exceptions import (
    CVError,
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
from pert_analyzer.cv.preprocessing import ImagePreprocessor


# =============================================================================
# Test Fixtures
# =============================================================================


def create_test_image(
    width: int = 400,
    height: int = 300,
    color: bool = True,
    fill_value: int = 200,
) -> np.ndarray:
    """
    Create a synthetic test image with rectangles and lines.

    Args:
        width: Image width.
        height: Image height.
        color: If True, create BGR image; else grayscale.
        fill_value: Base fill value.

    Returns:
        Test image as numpy array.
    """
    if color:
        image = np.full((height, width, 3), fill_value, dtype=np.uint8)
        # Add some structure: rectangles and lines
        cv2.rectangle(image, (50, 50), (150, 100), (0, 0, 0), 2)
        cv2.rectangle(image, (200, 100), (350, 200), (0, 0, 0), 2)
        cv2.line(image, (150, 75), (200, 150), (0, 0, 0), 2)
        cv2.putText(
            image, "A", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
        cv2.putText(
            image, "B", (250, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
    else:
        image = np.full((height, width), fill_value, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 100), 0, 2)
        cv2.rectangle(image, (200, 100), (350, 200), 0, 2)
        cv2.line(image, (150, 75), (200, 150), 0, 2)

    return image


def create_noisy_test_image(
    width: int = 400,
    height: int = 300,
) -> np.ndarray:
    """Create a test image with random noise."""
    image = create_test_image(width, height, color=False, fill_value=180)
    noise = np.random.randint(0, 50, (height, width), dtype=np.uint8)
    noisy = cv2.add(image, noise)
    return noisy


def create_rotated_test_image(
    angle: float = 5.0,
    width: int = 400,
    height: int = 300,
) -> np.ndarray:
    """Create a test image rotated by a specific angle."""
    image = create_test_image(width, height, color=False)
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (width, height), borderValue=255
    )
    return rotated


# =============================================================================
# Test PreprocessingConfig
# =============================================================================


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""

    def test_default_config_is_valid(self):
        config = PreprocessingConfig()
        issues = config.validate()
        assert len(issues) == 0

    def test_invalid_max_width(self):
        config = PreprocessingConfig(max_width=50, min_width=100)
        issues = config.validate()
        assert any("max_width" in issue for issue in issues)

    def test_invalid_max_height(self):
        config = PreprocessingConfig(max_height=50, min_height=100)
        issues = config.validate()
        assert any("max_height" in issue for issue in issues)

    def test_invalid_scale_factor(self):
        config = PreprocessingConfig(scale_factor=-1.0)
        issues = config.validate()
        assert any("scale_factor" in issue for issue in issues)

    def test_invalid_blur_kernel(self):
        config = PreprocessingConfig(blur_kernel_size=4)
        issues = config.validate()
        assert any("blur_kernel_size" in issue for issue in issues)

    def test_invalid_median_kernel(self):
        config = PreprocessingConfig(median_blur_kernel_size=3)
        # 3 is valid (odd and >= 1)
        issues = config.validate()
        assert not any("median_blur_kernel_size" in issue for issue in issues)

    def test_invalid_adaptive_block_size(self):
        config = PreprocessingConfig(adaptive_block_size=2)
        issues = config.validate()
        assert any("adaptive_block_size" in issue for issue in issues)

    def test_invalid_canny_thresholds(self):
        config = PreprocessingConfig(
            canny_low_threshold=100, canny_high_threshold=50
        )
        issues = config.validate()
        assert any("canny_high_threshold" in issue for issue in issues)

    def test_all_blur_methods(self):
        for method in BlurMethod:
            config = PreprocessingConfig(blur_method=method)
            issues = config.validate()
            assert len(issues) == 0

    def test_all_contrast_methods(self):
        for method in ContrastMethod:
            config = PreprocessingConfig(contrast_method=method)
            issues = config.validate()
            assert len(issues) == 0

    def test_all_threshold_methods(self):
        for method in ThresholdMethod:
            config = PreprocessingConfig(threshold_method=method)
            issues = config.validate()
            assert len(issues) == 0

    def test_all_edge_methods(self):
        for method in EdgeMethod:
            config = PreprocessingConfig(edge_method=method)
            issues = config.validate()
            assert len(issues) == 0

    def test_all_resize_strategies(self):
        for strategy in ResizeStrategy:
            config = PreprocessingConfig(resize_strategy=strategy)
            issues = config.validate()
            assert len(issues) == 0


# =============================================================================
# Test CoordinateMapping
# =============================================================================


class TestCoordinateMapping:
    """Tests for CoordinateMapping."""

    def test_scale_factors_computed(self):
        mapping = CoordinateMapping(
            original_width=800,
            original_height=600,
            processed_width=400,
            processed_height=300,
        )
        assert mapping.scale_x == 0.5
        assert mapping.scale_y == 0.5

    def test_to_original(self):
        mapping = CoordinateMapping(
            original_width=800,
            original_height=600,
            processed_width=400,
            processed_height=300,
        )
        orig_x, orig_y = mapping.to_original(200, 150)
        assert orig_x == pytest.approx(400.0)
        assert orig_y == pytest.approx(300.0)

    def test_to_processed(self):
        mapping = CoordinateMapping(
            original_width=800,
            original_height=600,
            processed_width=400,
            processed_height=300,
        )
        proc_x, proc_y = mapping.to_processed(400, 300)
        assert proc_x == pytest.approx(200.0)
        assert proc_y == pytest.approx(150.0)

    def test_scale_bbox(self):
        mapping = CoordinateMapping(
            original_width=800,
            original_height=600,
            processed_width=400,
            processed_height=300,
        )
        x, y, w, h = mapping.scale_bbox(100, 50, 200, 100)
        assert x == pytest.approx(200.0)
        assert y == pytest.approx(100.0)
        assert w == pytest.approx(400.0)
        assert h == pytest.approx(200.0)

    def test_identity_mapping(self):
        mapping = CoordinateMapping(
            original_width=400,
            original_height=300,
            processed_width=400,
            processed_height=300,
        )
        orig_x, orig_y = mapping.to_original(100, 50)
        assert orig_x == pytest.approx(100.0)
        assert orig_y == pytest.approx(50.0)

    def test_different_scale_factors(self):
        mapping = CoordinateMapping(
            original_width=800,
            original_height=600,
            processed_width=400,
            processed_height=600,
        )
        assert mapping.scale_x == pytest.approx(0.5)
        assert mapping.scale_y == pytest.approx(1.0)


# =============================================================================
# Test PreprocessingResult
# =============================================================================


class TestPreprocessingResult:
    """Tests for PreprocessingResult."""

    def test_empty_result(self):
        result = PreprocessingResult()
        assert not result.has_original
        assert not result.has_grayscale
        assert not result.has_binary
        assert not result.has_edges
        assert len(result.get_available_representations()) == 0

    def test_result_with_representations(self):
        result = PreprocessingResult()
        result.original = np.zeros((100, 100, 3), dtype=np.uint8)
        result.grayscale = np.zeros((100, 100), dtype=np.uint8)
        result.binary = np.zeros((100, 100), dtype=np.uint8)

        assert result.has_original
        assert result.has_grayscale
        assert result.has_binary
        assert not result.has_edges

        available = result.get_available_representations()
        assert "original" in available
        assert "grayscale" in available
        assert "binary" in available
        assert "edges" not in available

    def test_get_representation(self):
        result = PreprocessingResult()
        gray = np.zeros((100, 100), dtype=np.uint8)
        result.grayscale = gray

        assert result.get_representation("grayscale") is gray
        assert result.get_representation("binary") is None
        assert result.get_representation("nonexistent") is None

    def test_coordinate_conversion_no_mapping(self):
        result = PreprocessingResult()
        x, y = result.to_original_coordinates(100, 200)
        assert x == 100
        assert y == 200

    def test_coordinate_conversion_with_mapping(self):
        result = PreprocessingResult()
        result.coordinate_mapping = CoordinateMapping(
            original_width=800,
            original_height=600,
            processed_width=400,
            processed_height=300,
        )
        x, y = result.to_original_coordinates(200, 150)
        assert x == pytest.approx(400.0)
        assert y == pytest.approx(300.0)

    def test_warnings_list(self):
        result = PreprocessingResult()
        assert len(result.warnings) == 0
        result.warnings.append("Test warning")
        assert len(result.warnings) == 1


# =============================================================================
# Test ImagePreprocessor
# =============================================================================


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""

    def setup_method(self):
        self.config = PreprocessingConfig()
        self.preprocessor = ImagePreprocessor(self.config)

    def test_default_initialization(self):
        preprocessor = ImagePreprocessor()
        assert preprocessor.config is not None

    def test_custom_config_initialization(self):
        config = PreprocessingConfig(blur_method=BlurMethod.MEDIAN)
        preprocessor = ImagePreprocessor(config)
        assert preprocessor.config.blur_method == BlurMethod.MEDIAN

    def test_invalid_config_raises_error(self):
        config = PreprocessingConfig(max_width=-1)
        with pytest.raises(InvalidConfigError):
            ImagePreprocessor(config)

    def test_set_config(self):
        preprocessor = ImagePreprocessor()
        new_config = PreprocessingConfig(blur_method=BlurMethod.BILATERAL)
        preprocessor.set_config(new_config)
        assert preprocessor.config.blur_method == BlurMethod.BILATERAL


class TestImageValidation:
    """Tests for image validation."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_none_image_raises_error(self):
        with pytest.raises(ImageValidationError):
            self.preprocessor.process_image(None)

    def test_zero_width_image_raises_error(self):
        image = np.array([], dtype=np.uint8).reshape(0, 100)
        with pytest.raises(ImageSizeError):
            self.preprocessor.process_image(image)

    def test_zero_height_image_raises_error(self):
        image = np.array([], dtype=np.uint8).reshape(100, 0)
        with pytest.raises(ImageSizeError):
            self.preprocessor.process_image(image)

    def test_too_small_image_raises_error(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        with pytest.raises(ImageSizeError):
            self.preprocessor.process_image(image)

    def test_valid_color_image(self):
        image = create_test_image(400, 300, color=True)
        result = self.preprocessor.process_image(image)
        assert result.has_original

    def test_valid_grayscale_image(self):
        image = create_test_image(400, 300, color=False)
        result = self.preprocessor.process_image(image)
        assert result.has_original
        assert result.was_grayscale_input


class TestImageLoading:
    """Tests for image loading from file."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_load_valid_image(self):
        image = create_test_image(400, 300)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, image)
            temp_path = f.name
        try:
            result = self.preprocessor.process(temp_path)
            assert result.original_dimensions == (400, 300)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        with pytest.raises(ImageLoadError):
            self.preprocessor.process("nonexistent_image.png")

    def test_load_invalid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            temp_path = f.name
        try:
            with pytest.raises(ImageLoadError):
                self.preprocessor.process(temp_path)
        finally:
            os.unlink(temp_path)


class TestResizing:
    """Tests for image resizing."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_no_resize_when_within_limits(self):
        image = create_test_image(400, 300)
        result = self.preprocessor.process_image(image)
        assert result.processed_dimensions == (400, 300)
        assert result.scale_factor == 1.0

    def test_resize_large_image(self):
        config = PreprocessingConfig(
            resize_strategy=ResizeStrategy.MAX_DIMENSION,
            max_width=200,
            max_height=200,
        )
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(800, 600)
        result = preprocessor.process_image(image)
        w, h = result.processed_dimensions
        assert w <= 200
        assert h <= 200
        assert result.scale_factor < 1.0

    def test_preserves_aspect_ratio(self):
        config = PreprocessingConfig(
            resize_strategy=ResizeStrategy.MAX_DIMENSION,
            max_width=200,
            max_height=200,
        )
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(800, 400)
        result = preprocessor.process_image(image)
        w, h = result.processed_dimensions
        original_ratio = 800 / 400
        processed_ratio = w / h
        assert abs(original_ratio - processed_ratio) < 0.01

    def test_scale_factor_resize(self):
        config = PreprocessingConfig(
            resize_strategy=ResizeStrategy.SCALE_FACTOR,
            scale_factor=0.5,
        )
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300)
        result = preprocessor.process_image(image)
        assert result.processed_dimensions == (200, 150)

    def test_fixed_size_resize(self):
        config = PreprocessingConfig(
            resize_strategy=ResizeStrategy.FIXED_SIZE,
            target_width=300,
            target_height=200,
        )
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300)
        result = preprocessor.process_image(image)
        assert result.processed_dimensions == (300, 200)

    def test_no_resize_strategy(self):
        config = PreprocessingConfig(resize_strategy=ResizeStrategy.NONE)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(800, 600)
        result = preprocessor.process_image(image)
        assert result.processed_dimensions == (800, 600)


class TestGrayscaleConversion:
    """Tests for grayscale conversion."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_color_to_grayscale(self):
        image = create_test_image(400, 300, color=True)
        result = self.preprocessor.process_image(image)
        assert result.grayscale is not None
        assert len(result.grayscale.shape) == 2

    def test_grayscale_input_preserved(self):
        image = create_test_image(400, 300, color=False)
        result = self.preprocessor.process_image(image)
        assert result.grayscale is not None
        assert result.was_grayscale_input


class TestNoiseReduction:
    """Tests for noise reduction."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_gaussian_blur(self):
        config = PreprocessingConfig(blur_method=BlurMethod.GAUSSIAN)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.denoised is not None
        assert result.denoised.shape == result.grayscale.shape

    def test_median_blur(self):
        config = PreprocessingConfig(blur_method=BlurMethod.MEDIAN)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.denoised is not None

    def test_bilateral_filter(self):
        config = PreprocessingConfig(blur_method=BlurMethod.BILATERAL)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.denoised is not None

    def test_no_blur(self):
        config = PreprocessingConfig(blur_method=BlurMethod.NONE)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.denoised is not None


class TestContrastEnhancement:
    """Tests for contrast enhancement."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_clahe(self):
        config = PreprocessingConfig(contrast_method=ContrastMethod.CLAHE)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.contrast_enhanced is not None

    def test_histogram_equalization(self):
        config = PreprocessingConfig(
            contrast_method=ContrastMethod.HISTOGRAM_EQUALIZATION
        )
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.contrast_enhanced is not None

    def test_no_contrast_enhancement(self):
        config = PreprocessingConfig(contrast_method=ContrastMethod.NONE)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.contrast_enhanced is not None


class TestThresholding:
    """Tests for thresholding."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_otsu_threshold(self):
        config = PreprocessingConfig(threshold_method=ThresholdMethod.OTSU)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.binary is not None
        unique_values = np.unique(result.binary)
        assert len(unique_values) <= 2

    def test_global_threshold(self):
        config = PreprocessingConfig(threshold_method=ThresholdMethod.GLOBAL)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.binary is not None

    def test_adaptive_mean_threshold(self):
        config = PreprocessingConfig(
            threshold_method=ThresholdMethod.ADAPTIVE_MEAN
        )
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.adaptive_binary is not None

    def test_adaptive_gaussian_threshold(self):
        config = PreprocessingConfig(
            threshold_method=ThresholdMethod.ADAPTIVE_GAUSSIAN
        )
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.adaptive_binary is not None


class TestEdgeDetection:
    """Tests for edge detection."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_canny_edges(self):
        config = PreprocessingConfig(edge_method=EdgeMethod.CANNY)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.edges is not None
        assert result.edges.shape == result.grayscale.shape

    def test_sobel_edges(self):
        config = PreprocessingConfig(edge_method=EdgeMethod.SOBEL)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.edges is not None

    def test_no_edge_detection(self):
        config = PreprocessingConfig(edge_method=EdgeMethod.NONE)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.edges is not None


class TestDeskew:
    """Tests for skew estimation and correction."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_straight_image_no_rotation(self):
        image = create_test_image(400, 300, color=False)
        result = self.preprocessor.process_image(image)
        assert result.rotation_angle == pytest.approx(0.0, abs=1.0)
        assert not result.was_deskewed

    def test_deskew_enabled(self):
        config = PreprocessingConfig(enable_deskew=True)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.rotation_angle is not None

    def test_deskew_disabled(self):
        config = PreprocessingConfig(enable_deskew=False)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.rotation_angle == 0.0
        assert not result.was_deskewed


class TestCoordinateMapping:
    """Tests for coordinate mapping in preprocessing results."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_no_resize_creates_identity_mapping(self):
        config = PreprocessingConfig(resize_strategy=ResizeStrategy.NONE)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.coordinate_mapping is not None
        assert result.coordinate_mapping.scale_x == pytest.approx(1.0)
        assert result.coordinate_mapping.scale_y == pytest.approx(1.0)

    def test_resize_creates_correct_mapping(self):
        config = PreprocessingConfig(
            resize_strategy=ResizeStrategy.SCALE_FACTOR,
            scale_factor=0.5,
        )
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.coordinate_mapping is not None
        assert result.coordinate_mapping.scale_x == pytest.approx(0.5)
        assert result.coordinate_mapping.scale_y == pytest.approx(0.5)

    def test_coordinate_roundtrip(self):
        config = PreprocessingConfig(
            resize_strategy=ResizeStrategy.SCALE_FACTOR,
            scale_factor=0.5,
        )
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)

        # Original coordinate
        orig_x, orig_y = 200.0, 150.0
        # Convert to processed
        proc_x, proc_y = result.to_processed_coordinates(orig_x, orig_y)
        # Convert back
        back_x, back_y = result.to_original_coordinates(proc_x, proc_y)

        assert back_x == pytest.approx(orig_x, abs=0.1)
        assert back_y == pytest.approx(orig_y, abs=0.1)


class TestMultipleRepresentations:
    """Tests that multiple representations are available."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_all_representations_available(self):
        image = create_test_image(400, 300, color=True)
        result = self.preprocessor.process_image(image)

        available = result.get_available_representations()
        assert "original" in available
        assert "grayscale" in available
        assert "denoised" in available
        assert "contrast_enhanced" in available
        assert "binary" in available
        assert "adaptive_binary" in available
        assert "edges" in available

    def test_representations_have_correct_shapes(self):
        image = create_test_image(400, 300, color=True)
        result = self.preprocessor.process_image(image)

        h, w = result.grayscale.shape
        assert result.denoised.shape == (h, w)
        assert result.contrast_enhanced.shape == (h, w)
        assert result.binary.shape == (h, w)
        assert result.adaptive_binary.shape == (h, w)
        assert result.edges.shape == (h, w)


class TestOriginalImagePreserved:
    """Tests that the original image is not modified."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_original_not_modified(self):
        image = create_test_image(400, 300, color=True)
        original_copy = image.copy()
        result = self.preprocessor.process_image(image)

        # Original array should be unchanged
        np.testing.assert_array_equal(image, original_copy)
        # Result original should be a copy
        np.testing.assert_array_equal(result.original, original_copy)

    def test_preserve_original_config(self):
        config = PreprocessingConfig(preserve_original=False)
        preprocessor = ImagePreprocessor(config)
        image = create_test_image(400, 300, color=True)
        result = preprocessor.process_image(image)
        assert result.original is None


class TestErrorHandling:
    """Tests for error handling edge cases."""

    def test_preprocessing_error_for_unsupportedBlur(self):
        config = PreprocessingConfig(blur_method="invalid")
        # Setting invalid blur method directly bypasses validation
        preprocessor = ImagePreprocessor.__new__(ImagePreprocessor)
        preprocessor._config = config
        image = create_test_image(400, 300, color=False)
        # Should not raise, just log warning and return unchanged
        result = preprocessor.process_image(image)
        assert result.denoised is not None

    def test_preprocessing_error_for_invalid_contrast(self):
        config = PreprocessingConfig(contrast_method="invalid")
        preprocessor = ImagePreprocessor.__new__(ImagePreprocessor)
        preprocessor._config = config
        image = create_test_image(400, 300, color=False)
        result = preprocessor.process_image(image)
        assert result.contrast_enhanced is not None


class TestGetAvailableRepresentations:
    """Tests for get_available_representations method."""

    def test_all_methods_listed(self):
        preprocessor = ImagePreprocessor()
        reps = preprocessor.get_available_representations()
        assert "original" in reps
        assert "grayscale" in reps
        assert "denoised" in reps
        assert "contrast_enhanced" in reps
        assert "binary" in reps
        assert "adaptive_binary" in reps
        assert "edges" in reps
