"""
Unit tests for LEDDiffusionCSCMatrix class.

Tests the CSC sparse matrix wrapper used for LED diffusion pattern storage.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_csc_matrix():
    """Create a sample CSC matrix for testing."""
    # Create a matrix with shape (pixels, led_count * channels)
    # For a 10x10 image with 5 LEDs and 3 channels
    height, width = 10, 10
    pixels = height * width
    led_count = 5
    channels = 3
    n_cols = led_count * channels

    # Create random sparse data
    np.random.seed(42)
    density = 0.1
    matrix = sp.random(pixels, n_cols, density=density, format="csc", dtype=np.float32)

    return matrix, height, width, channels


@pytest.fixture
def led_diffusion_matrix(sample_csc_matrix):
    """Create a LEDDiffusionCSCMatrix instance."""
    from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

    matrix, height, width, channels = sample_csc_matrix
    return LEDDiffusionCSCMatrix(
        csc_matrix=matrix,
        height=height,
        width=width,
        channels=channels,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestLEDDiffusionCSCMatrixInit:
    """Test LEDDiffusionCSCMatrix initialization."""

    def test_basic_initialization(self, sample_csc_matrix):
        """Test basic matrix initialization."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        matrix, height, width, channels = sample_csc_matrix
        wrapper = LEDDiffusionCSCMatrix(
            csc_matrix=matrix,
            height=height,
            width=width,
            channels=channels,
        )

        assert wrapper.height == height
        assert wrapper.width == width
        assert wrapper.channels == channels
        assert wrapper.pixels == height * width
        assert wrapper.led_count == 5

    def test_properties(self, led_diffusion_matrix):
        """Test matrix properties."""
        assert led_diffusion_matrix.shape == (100, 15)  # 10*10 pixels, 5*3 cols
        assert led_diffusion_matrix.data is not None
        assert led_diffusion_matrix.indices is not None
        assert led_diffusion_matrix.indptr is not None

    def test_invalid_rows_raises(self):
        """Test that invalid row count raises error."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        matrix = sp.random(100, 15, density=0.1, format="csc", dtype=np.float32)

        with pytest.raises(ValueError, match="Matrix rows"):
            LEDDiffusionCSCMatrix(
                csc_matrix=matrix,
                height=5,  # Wrong height
                width=10,
                channels=3,
            )

    def test_invalid_columns_raises(self):
        """Test that non-divisible columns raises error."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        matrix = sp.random(100, 16, density=0.1, format="csc", dtype=np.float32)

        with pytest.raises(ValueError, match="not divisible by channels"):
            LEDDiffusionCSCMatrix(
                csc_matrix=matrix,
                height=10,
                width=10,
                channels=3,  # 16 not divisible by 3
            )

    def test_non_csc_raises(self):
        """Test that non-CSC matrix raises error."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        matrix = sp.random(100, 15, density=0.1, format="csr", dtype=np.float32)

        with pytest.raises(ValueError, match="CSC format"):
            LEDDiffusionCSCMatrix(
                csc_matrix=matrix,
                height=10,
                width=10,
                channels=3,
            )


# =============================================================================
# Factory Method Tests
# =============================================================================


class TestFactoryMethods:
    """Test factory methods for creating matrices."""

    def test_from_csc_matrix(self, sample_csc_matrix):
        """Test creating from existing CSC matrix."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        matrix, height, width, channels = sample_csc_matrix
        wrapper = LEDDiffusionCSCMatrix.from_csc_matrix(
            csc_matrix=matrix,
            height=height,
            width=width,
            channels=channels,
        )

        assert wrapper.height == height
        assert wrapper.led_count == 5

    def test_from_arrays(self, sample_csc_matrix):
        """Test creating from CSC component arrays."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        matrix, height, width, channels = sample_csc_matrix
        wrapper = LEDDiffusionCSCMatrix.from_arrays(
            data=matrix.data,
            indices=matrix.indices,
            indptr=matrix.indptr,
            shape=matrix.shape,
            height=height,
            width=width,
            channels=channels,
        )

        assert wrapper.shape == matrix.shape


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Test serialization methods."""

    def test_to_dict(self, led_diffusion_matrix):
        """Test converting to dictionary."""
        result = led_diffusion_matrix.to_dict()

        assert "csc_data" in result
        assert "csc_indices" in result
        assert "csc_indptr" in result
        assert "csc_shape" in result
        assert "csc_height" in result
        assert "csc_width" in result
        assert "csc_channels" in result
        assert "csc_led_count" in result
        assert "csc_nnz" in result

    def test_from_dict(self, led_diffusion_matrix):
        """Test loading from dictionary."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        data_dict = led_diffusion_matrix.to_dict()
        restored = LEDDiffusionCSCMatrix.from_dict(data_dict)

        assert restored.height == led_diffusion_matrix.height
        assert restored.width == led_diffusion_matrix.width
        assert restored.channels == led_diffusion_matrix.channels
        assert restored.led_count == led_diffusion_matrix.led_count

    def test_roundtrip(self, led_diffusion_matrix):
        """Test round-trip serialization preserves data."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        data_dict = led_diffusion_matrix.to_dict()
        restored = LEDDiffusionCSCMatrix.from_dict(data_dict)

        np.testing.assert_array_almost_equal(
            restored.matrix.toarray(),
            led_diffusion_matrix.matrix.toarray(),
        )

    def test_to_csc_matrix(self, led_diffusion_matrix, sample_csc_matrix):
        """Test converting back to scipy CSC matrix."""
        result = led_diffusion_matrix.to_csc_matrix()

        assert sp.isspmatrix_csc(result)
        assert result.shape == led_diffusion_matrix.shape


# =============================================================================
# Dense Materialization Tests
# =============================================================================


class TestMaterializeDense:
    """Test dense pattern materialization."""

    def test_materialize_dense_shape(self, led_diffusion_matrix):
        """Test materialize_dense returns correct shape."""
        pattern = led_diffusion_matrix.materialize_dense(led_idx=0, channel=0)

        assert pattern.shape == (10, 10)

    def test_materialize_dense_all_leds(self, led_diffusion_matrix):
        """Test materializing all LEDs."""
        for led_idx in range(led_diffusion_matrix.led_count):
            for channel in range(led_diffusion_matrix.channels):
                pattern = led_diffusion_matrix.materialize_dense(led_idx, channel)
                assert pattern.shape == (10, 10)

    def test_materialize_dense_invalid_led(self, led_diffusion_matrix):
        """Test invalid LED index raises error."""
        with pytest.raises(ValueError, match="LED index"):
            led_diffusion_matrix.materialize_dense(led_idx=100, channel=0)

    def test_materialize_dense_invalid_channel(self, led_diffusion_matrix):
        """Test invalid channel raises error."""
        with pytest.raises(ValueError, match="Channel"):
            led_diffusion_matrix.materialize_dense(led_idx=0, channel=10)


# =============================================================================
# Bounding Box Tests
# =============================================================================


class TestBoundingBox:
    """Test bounding box functionality."""

    def test_get_bounding_box(self, led_diffusion_matrix):
        """Test getting bounding box."""
        bbox = led_diffusion_matrix.get_bounding_box(led_idx=0, channel=0)

        assert len(bbox) == 4
        min_row, min_col, max_row, max_col = bbox
        assert min_row <= max_row
        assert min_col <= max_col

    def test_get_bounding_box_empty_pattern(self):
        """Test bounding box for empty pattern."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        # Create matrix with one empty column
        matrix = sp.csc_matrix((100, 3), dtype=np.float32)
        wrapper = LEDDiffusionCSCMatrix(
            csc_matrix=matrix,
            height=10,
            width=10,
            channels=3,
        )

        bbox = wrapper.get_bounding_box(led_idx=0, channel=0)

        assert bbox == (0, 0, 0, 0)

    def test_get_bounding_box_invalid_led(self, led_diffusion_matrix):
        """Test invalid LED raises error."""
        with pytest.raises(ValueError, match="LED index"):
            led_diffusion_matrix.get_bounding_box(led_idx=100, channel=0)


# =============================================================================
# Region Extraction Tests
# =============================================================================


class TestExtractRegion:
    """Test region extraction functionality."""

    def test_extract_region(self, led_diffusion_matrix):
        """Test extracting a region."""
        region = led_diffusion_matrix.extract_region(
            led_idx=0,
            channel=0,
            min_row=2,
            min_col=2,
            max_row=5,
            max_col=5,
        )

        assert region.shape == (4, 4)  # 5-2+1, 5-2+1

    def test_extract_region_invalid_bounds(self, led_diffusion_matrix):
        """Test invalid bounds raise error."""
        with pytest.raises(ValueError, match="Invalid row range"):
            led_diffusion_matrix.extract_region(
                led_idx=0,
                channel=0,
                min_row=5,
                min_col=0,
                max_row=2,  # min > max
                max_col=5,
            )


# =============================================================================
# Set Image Tests
# =============================================================================


class TestSetImage:
    """Test setting image data."""

    def test_set_image(self, led_diffusion_matrix):
        """Test setting image for LED/channel."""
        new_image = np.random.rand(10, 10).astype(np.float32)

        led_diffusion_matrix.set_image(led_idx=0, channel=0, dense_image=new_image)

        result = led_diffusion_matrix.materialize_dense(led_idx=0, channel=0)
        np.testing.assert_array_almost_equal(result, new_image, decimal=5)

    def test_set_image_with_threshold(self, led_diffusion_matrix):
        """Test setting image with sparsity threshold."""
        new_image = np.zeros((10, 10), dtype=np.float32)
        new_image[5, 5] = 0.5

        led_diffusion_matrix.set_image(
            led_idx=0,
            channel=0,
            dense_image=new_image,
            sparsity_threshold=0.1,
        )

        result = led_diffusion_matrix.materialize_dense(led_idx=0, channel=0)
        assert result[5, 5] == pytest.approx(0.5, abs=0.01)

    def test_set_image_invalid_shape(self, led_diffusion_matrix):
        """Test setting image with wrong shape raises error."""
        wrong_shape = np.zeros((5, 5), dtype=np.float32)

        with pytest.raises(ValueError, match="Dense image shape"):
            led_diffusion_matrix.set_image(led_idx=0, channel=0, dense_image=wrong_shape)


# =============================================================================
# Horizontal Stack Tests
# =============================================================================


class TestHStack:
    """Test horizontal stacking of matrices."""

    def test_hstack(self, sample_csc_matrix):
        """Test stacking multiple matrices."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        matrix, height, width, channels = sample_csc_matrix
        wrapper1 = LEDDiffusionCSCMatrix(
            csc_matrix=matrix,
            height=height,
            width=width,
            channels=channels,
        )
        wrapper2 = LEDDiffusionCSCMatrix(
            csc_matrix=matrix,
            height=height,
            width=width,
            channels=channels,
        )

        stacked = LEDDiffusionCSCMatrix.hstack([wrapper1, wrapper2])

        assert stacked.led_count == 10  # 5 + 5
        assert stacked.shape[0] == 100  # Same pixels

    def test_hstack_empty_raises(self):
        """Test stacking empty list raises error."""
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        with pytest.raises(ValueError, match="empty list"):
            LEDDiffusionCSCMatrix.hstack([])


# =============================================================================
# Memory Info Tests
# =============================================================================


class TestMemoryInfo:
    """Test memory information functionality."""

    def test_memory_info(self, led_diffusion_matrix):
        """Test getting memory info."""
        info = led_diffusion_matrix.memory_info()

        assert "data_mb" in info
        assert "indices_mb" in info
        assert "indptr_mb" in info
        assert "total_mb" in info
        assert "equivalent_dense_mb" in info
        assert "compression_ratio" in info
        assert "sparsity_ratio" in info
        assert "nnz" in info
        assert "shape" in info

    def test_memory_info_values(self, led_diffusion_matrix):
        """Test memory info values are reasonable."""
        info = led_diffusion_matrix.memory_info()

        assert info["total_mb"] > 0
        assert info["equivalent_dense_mb"] > 0
        assert 0 <= info["sparsity_ratio"] <= 1
        assert info["compression_ratio"] > 0


# =============================================================================
# Dense Patterns Tests
# =============================================================================


class TestDensePatterns:
    """Test dense pattern conversion."""

    def test_to_dense_patterns_shape(self, led_diffusion_matrix):
        """Test dense patterns output shape."""
        patterns = led_diffusion_matrix.to_dense_patterns()

        assert patterns.shape == (5, 10, 10, 3)  # (led_count, H, W, channels)

    def test_to_dense_patterns_dtype(self, led_diffusion_matrix):
        """Test dense patterns dtype."""
        patterns = led_diffusion_matrix.to_dense_patterns()

        assert patterns.dtype == np.float32


# =============================================================================
# Pattern Summary Tests
# =============================================================================


class TestPatternSummary:
    """Test pattern summary functionality."""

    def test_get_pattern_summary(self, led_diffusion_matrix):
        """Test getting pattern summary."""
        summary = led_diffusion_matrix.get_pattern_summary()

        assert summary["led_count"] == 5
        assert summary["channels"] == 3
        assert "max_intensities" in summary
        assert "mean_intensities" in summary
        assert "pattern_extents" in summary
        assert "channel_nnz" in summary
        assert "memory_mb" in summary


# =============================================================================
# LED Bounding Boxes Tests
# =============================================================================


class TestLEDBoundingBoxes:
    """Test LED bounding boxes functionality."""

    def test_get_led_bounding_boxes(self, led_diffusion_matrix):
        """Test getting all LED bounding boxes."""
        bboxes = led_diffusion_matrix.get_led_bounding_boxes()

        assert bboxes.shape == (5, 4)  # (led_count, 4)
        assert bboxes.dtype == np.int_


# =============================================================================
# RGB Channel Tests
# =============================================================================


class TestRGBChannels:
    """Test RGB channel extraction."""

    def test_extract_rgb_channels(self, led_diffusion_matrix):
        """Test extracting RGB channels."""
        R, G, B = led_diffusion_matrix.extract_rgb_channels()

        assert R.shape == (100, 5)  # (pixels, led_count)
        assert G.shape == (100, 5)
        assert B.shape == (100, 5)

    def test_create_block_diagonal_matrix(self, led_diffusion_matrix):
        """Test creating block diagonal matrix."""
        block_diag = led_diffusion_matrix.create_block_diagonal_matrix()

        # Shape: (pixels * 3, led_count * 3)
        assert block_diag.shape == (300, 15)


# =============================================================================
# Repr Tests
# =============================================================================


class TestRepr:
    """Test string representation."""

    def test_repr(self, led_diffusion_matrix):
        """Test __repr__ output."""
        result = repr(led_diffusion_matrix)

        assert "LEDDiffusionCSCMatrix" in result
        assert "led_count=5" in result
        assert "channels=3" in result
