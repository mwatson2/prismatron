"""
Unit tests for DenseATAMatrix class.

Tests the methods used by tools/compute_matrices.py for intermediate
ATA computation during pattern matrix generation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_dense_matrices():
    """Create sample dense ATA matrices (3 channels, 100 LEDs)."""
    # Create symmetric positive semi-definite matrices
    led_count = 100
    channels = 3
    matrices = np.zeros((channels, led_count, led_count), dtype=np.float32)

    for c in range(channels):
        # Create a random matrix and make it symmetric positive semi-definite
        A = np.random.rand(led_count, led_count).astype(np.float32)
        matrices[c] = A @ A.T  # A @ A.T is always symmetric positive semi-definite

    return matrices


# =============================================================================
# DenseATAMatrix Initialization Tests
# =============================================================================


class TestDenseATAMatrixInit:
    """Test DenseATAMatrix initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with defaults."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)

        assert matrix.led_count == 100
        assert matrix.channels == 3
        assert matrix.is_built is False
        assert matrix.memory_mb == 0.0
        assert matrix.dense_matrices_gpu is None
        assert matrix.dense_matrices_cpu is None

    def test_initialization_with_custom_channels(self):
        """Test initialization with custom channel count."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=50, channels=4)

        assert matrix.led_count == 50
        assert matrix.channels == 4

    def test_initialization_with_float32(self):
        """Test initialization with float32 dtype."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        try:
            import cupy

            dtype = cupy.float32
        except ImportError:
            dtype = np.float32

        matrix = DenseATAMatrix(led_count=100, storage_dtype=dtype, output_dtype=dtype)

        assert matrix.storage_dtype == dtype
        assert matrix.output_dtype == dtype

    def test_initialization_with_float16(self):
        """Test initialization with float16 dtype."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        try:
            import cupy

            dtype = cupy.float16
        except ImportError:
            dtype = np.float16

        matrix = DenseATAMatrix(led_count=100, storage_dtype=dtype, output_dtype=dtype)

        assert matrix.storage_dtype == dtype
        assert matrix.output_dtype == dtype


# =============================================================================
# DenseATAMatrix Attribute Setting Tests
# =============================================================================


class TestDenseATAMatrixAttributes:
    """Test setting DenseATAMatrix attributes directly (as done in compute_matrices.py)."""

    def test_set_dense_matrices_cpu(self, sample_dense_matrices):
        """Test setting dense_matrices_cpu directly."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)
        matrix.dense_matrices_cpu = sample_dense_matrices

        assert matrix.dense_matrices_cpu is not None
        assert matrix.dense_matrices_cpu.shape == (3, 100, 100)

    def test_set_memory_mb(self):
        """Test setting memory_mb directly."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)
        matrix.memory_mb = 123.45

        assert matrix.memory_mb == 123.45

    def test_set_is_built(self):
        """Test setting is_built directly."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)
        assert matrix.is_built is False

        matrix.is_built = True
        assert matrix.is_built is True


# =============================================================================
# DenseATAMatrix to_dict Tests
# =============================================================================


class TestDenseATAMatrixToDict:
    """Test DenseATAMatrix.to_dict() method."""

    def test_to_dict_returns_required_fields(self, sample_dense_matrices):
        """Test to_dict returns all required fields."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)
        matrix.dense_matrices_cpu = sample_dense_matrices
        matrix.memory_mb = sample_dense_matrices.nbytes / (1024 * 1024)
        matrix.is_built = True

        result = matrix.to_dict()

        assert "dense_matrices" in result
        assert "led_count" in result
        assert "channels" in result
        assert "storage_dtype" in result
        assert "output_dtype" in result
        assert "memory_mb" in result
        assert "version" in result
        assert "format" in result
        assert "matrix_shape" in result

    def test_to_dict_values(self, sample_dense_matrices):
        """Test to_dict returns correct values."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)
        matrix.dense_matrices_cpu = sample_dense_matrices
        matrix.memory_mb = 11.44  # Approximate for 3*100*100*4 bytes
        matrix.is_built = True

        result = matrix.to_dict()

        assert result["led_count"] == 100
        assert result["channels"] == 3
        assert result["format"] == "dense_ata"
        assert result["matrix_shape"] == (3, 100, 100)
        np.testing.assert_array_equal(result["dense_matrices"], sample_dense_matrices)

    def test_to_dict_raises_when_not_built(self):
        """Test to_dict raises error when not built."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)
        # is_built is False by default

        with pytest.raises(RuntimeError, match="not built"):
            matrix.to_dict()


# =============================================================================
# DenseATAMatrix from_dict Tests
# =============================================================================


class TestDenseATAMatrixFromDict:
    """Test DenseATAMatrix.from_dict() class method."""

    def test_from_dict_basic(self, sample_dense_matrices):
        """Test loading from dict."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        data = {
            "dense_matrices": sample_dense_matrices,
            "led_count": 100,
            "channels": 3,
            "storage_dtype": "float32",
            "output_dtype": "float32",
            "memory_mb": 11.44,
            "version": "1.0",
        }

        matrix = DenseATAMatrix.from_dict(data)

        assert matrix.led_count == 100
        assert matrix.channels == 3
        assert matrix.is_built is True
        assert matrix.memory_mb == 11.44
        np.testing.assert_array_equal(matrix.dense_matrices_cpu, sample_dense_matrices)

    def test_from_dict_float16(self, sample_dense_matrices):
        """Test loading from dict with float16."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        data = {
            "dense_matrices": sample_dense_matrices.astype(np.float16),
            "led_count": 100,
            "channels": 3,
            "storage_dtype": "float16",
            "output_dtype": "float16",
            "memory_mb": 5.72,
            "version": "1.0",
        }

        matrix = DenseATAMatrix.from_dict(data)

        try:
            import cupy

            assert matrix.storage_dtype == cupy.float16
            assert matrix.output_dtype == cupy.float16
        except ImportError:
            # Without cupy, dtype comparison may differ
            pass

    def test_from_dict_gpu_not_loaded(self, sample_dense_matrices):
        """Test that GPU matrices are not loaded until needed."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        data = {
            "dense_matrices": sample_dense_matrices,
            "led_count": 100,
            "channels": 3,
            "storage_dtype": "float32",
            "output_dtype": "float32",
        }

        matrix = DenseATAMatrix.from_dict(data)

        # GPU matrices should not be loaded yet
        assert matrix.dense_matrices_gpu is None
        # CPU matrices should be loaded
        assert matrix.dense_matrices_cpu is not None


# =============================================================================
# DenseATAMatrix Round-trip Tests
# =============================================================================


class TestDenseATAMatrixRoundTrip:
    """Test serialization round-trip (to_dict -> from_dict)."""

    def test_round_trip(self, sample_dense_matrices):
        """Test that to_dict -> from_dict preserves data."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        # Create and populate original
        original = DenseATAMatrix(led_count=100)
        original.dense_matrices_cpu = sample_dense_matrices
        original.memory_mb = 11.44
        original.is_built = True

        # Serialize and deserialize
        data = original.to_dict()
        restored = DenseATAMatrix.from_dict(data)

        # Verify
        assert restored.led_count == original.led_count
        assert restored.channels == original.channels
        assert restored.is_built == original.is_built
        assert restored.memory_mb == original.memory_mb
        np.testing.assert_array_equal(restored.dense_matrices_cpu, original.dense_matrices_cpu)


# =============================================================================
# DenseATAMatrix Info Methods Tests
# =============================================================================


class TestDenseATAMatrixInfo:
    """Test info and utility methods."""

    def test_memory_info(self, sample_dense_matrices):
        """Test memory_info method."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)
        matrix.dense_matrices_cpu = sample_dense_matrices
        matrix.memory_mb = 11.44
        matrix.is_built = True

        info = matrix.memory_info()

        assert "total_mb" in info
        assert "gpu_mb" in info
        assert "cpu_mb" in info
        assert info["total_mb"] == 11.44

    def test_get_info(self, sample_dense_matrices):
        """Test get_info method."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)
        matrix.dense_matrices_cpu = sample_dense_matrices
        matrix.memory_mb = 11.44
        matrix.is_built = True

        info = matrix.get_info()

        assert info["format"] == "dense"
        assert info["led_count"] == 100
        assert info["channels"] == 3
        assert info["is_built"] is True
        assert info["matrix_shape"] == (3, 100, 100)

    def test_str_representation_built(self, sample_dense_matrices):
        """Test string representation when built."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)
        matrix.dense_matrices_cpu = sample_dense_matrices
        matrix.memory_mb = 11.44
        matrix.is_built = True

        s = str(matrix)

        assert "100 LEDs" in s
        assert "3 channels" in s
        assert "11.4" in s  # memory

    def test_str_representation_not_built(self):
        """Test string representation when not built."""
        from src.utils.dense_ata_matrix import DenseATAMatrix

        matrix = DenseATAMatrix(led_count=100)

        s = str(matrix)

        assert "100 LEDs" in s
        assert "not built" in s
