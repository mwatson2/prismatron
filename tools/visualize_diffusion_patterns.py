#!/usr/bin/env python3
"""
Diffusion Pattern Visualization Tool.

This tool creates a web interface to visualize LED diffusion patterns using
the new nested storage format with SingleBlockMixedSparseTensor.

Features:
1. Grid view of all LED patterns
2. Individual LED/channel navigation
3. Support for mixed tensor format
4. Interactive controls and pattern comparison
5. Comprehensive pattern statistics
6. ATA matrix visualization
7. Source/optimized image comparison

Supported format:
- New nested format: NPZ with mixed_tensor (SingleBlockMixedSparseTensor) and metadata

Usage:
    python visualize_diffusion_patterns.py --patterns patterns.npz \
        --host 0.0.0.0 --port 8080
"""

import argparse
import base64
import io
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from flask import Flask, jsonify, render_template_string, request, send_file, send_from_directory

try:
    import cupy as cp
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cp

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Add src to path for imports - this allows importing as packages
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Import constants directly to avoid relative import issues
FRAME_HEIGHT = 480
FRAME_WIDTH = 800

from utils.dense_ata_matrix import DenseATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.frame_optimizer import optimize_frame_led_values

# Now import from utils as a package
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix

DIAGONAL_ATA_AVAILABLE = True
FRAME_OPTIMIZER_AVAILABLE = True

logger = logging.getLogger(__name__)


class DiffusionPatternVisualizer:
    """Web-based diffusion pattern visualization tool using new wrapper classes."""

    def __init__(self, patterns_file: Optional[str] = None):
        """
        Initialize visualizer.

        Args:
            patterns_file: Path to patterns file (.npz) in new nested format
        """
        self.patterns_file = patterns_file

        # Mixed tensor format
        self.mixed_tensor: Optional[SingleBlockMixedSparseTensor] = None

        # Pattern cache for visualization (on-demand loading)
        self.pattern_cache: Dict[int, np.ndarray] = {}

        # Metadata and LED info
        self.metadata: Dict = {}
        self.led_positions: Optional[np.ndarray] = None
        self.led_spatial_mapping: Optional[Dict] = None
        self.dense_ata_data: Optional[Dict] = None
        self.dia_matrix_data: Optional[Dict] = None
        self.symmetric_dia_data: Optional[Dict] = None
        self.ata_matrix_format: Optional[str] = None  # "dia", "dense", or "symmetric_dia"
        self.ata_inverse_data: Optional[np.ndarray] = None

        # Image directories - relative to project root
        project_root = Path(__file__).parent.parent
        self.source_images_dir = project_root / "images/source"
        self.optimized_images_dir = project_root / "images/optimized"

        # Flask app
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)  # Reduce Flask logging

        # Setup routes
        self._setup_routes()

    def _add_block_borders(self, image_array: np.ndarray, block_size: int = 64) -> np.ndarray:
        """Add a green border around the entire 64x64 LED diffusion pattern.

        This is for individual LED images only (thumbnails and detail pages).
        Simply adds a 2-pixel green border around the entire 64x64 block.

        Args:
            image_array: RGB image array of shape (height, width, 3) - typically (64, 64, 3)
            block_size: Not used, kept for compatibility

        Returns:
            RGB image array with a green border around the entire block
        """
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            # Not an RGB image, return as-is
            return image_array

        height, width, channels = image_array.shape

        # For individual LED blocks, add a border around the entire 64x64 region
        # This shows the boundary of the captured diffusion pattern
        border_width = 2
        result = np.zeros((height + 2 * border_width, width + 2 * border_width, 3), dtype=np.uint8)

        # Green color for border (RGB: 0, 255, 0)
        green_color = np.array([0, 255, 0], dtype=np.uint8)

        # Copy the original image to the center
        result[border_width:-border_width, border_width:-border_width] = image_array

        # Draw the border around the entire 64x64 block
        result[0:border_width, :] = green_color  # Top
        result[-border_width:, :] = green_color  # Bottom
        result[:, 0:border_width] = green_color  # Left
        result[:, -border_width:] = green_color  # Right

        return result

    def load_patterns(self) -> bool:
        """Load diffusion patterns from new nested format file."""
        try:
            if not self.patterns_file or not Path(self.patterns_file).exists():
                logger.error("No patterns file provided or file does not exist")
                logger.error(
                    "Generate patterns first with: python tools/generate_synthetic_patterns.py --output patterns.npz"
                )
                return False

            logger.info(f"Loading patterns from {self.patterns_file}")

            # Load the nested format data
            data = np.load(self.patterns_file, allow_pickle=True)

            # Check for new nested format
            if not all(key in data for key in ["mixed_tensor", "metadata"]):
                logger.error("File does not contain the required format")
                logger.error("Expected keys: mixed_tensor, metadata")
                logger.error(f"Found keys: {list(data.keys())}")
                logger.error("Please regenerate patterns with the updated generate_synthetic_patterns.py")
                return False

            logger.info("Detected pattern format, loading wrapper objects...")
            return self._load_nested_format(data)

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def _load_nested_format(self, data: np.lib.npyio.NpzFile) -> bool:
        """Load patterns from nested format."""
        try:
            # Load SingleBlockMixedSparseTensor
            logger.info("Loading SingleBlockMixedSparseTensor...")
            mixed_tensor_dict = data["mixed_tensor"].item()
            self.mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
            logger.info(f"Loaded mixed tensor: {self.mixed_tensor}")

            # No need for CSC matrix - we use mixed tensor directly

            # Load metadata
            self.metadata = data["metadata"].item() if "metadata" in data else {}

            # Load LED positions and spatial mapping
            self.led_positions = data.get("led_positions", None)
            self.led_spatial_mapping = data.get("led_spatial_mapping", None)
            if self.led_spatial_mapping is not None and hasattr(self.led_spatial_mapping, "item"):
                self.led_spatial_mapping = self.led_spatial_mapping.item()

            # Load ATA matrix data - support DIA, dense, and symmetric DIA formats
            self.dia_matrix_data = data.get("dia_matrix", None)
            if self.dia_matrix_data is not None and hasattr(self.dia_matrix_data, "item"):
                self.dia_matrix_data = self.dia_matrix_data.item()
                self.ata_matrix_format = "dia"
                logger.info("Loaded DIA format ATA matrix")

            self.dense_ata_data = data.get("dense_ata_matrix", None)
            if self.dense_ata_data is not None and hasattr(self.dense_ata_data, "item"):
                self.dense_ata_data = self.dense_ata_data.item()
                self.ata_matrix_format = "dense"
                logger.info("Loaded dense format ATA matrix")

            self.symmetric_dia_data = data.get("symmetric_dia_matrix", None)
            if self.symmetric_dia_data is not None and hasattr(self.symmetric_dia_data, "item"):
                self.symmetric_dia_data = self.symmetric_dia_data.item()
                self.ata_matrix_format = "symmetric_dia"
                logger.info("Loaded symmetric DIA format ATA matrix")

            # Check priority order: prefer symmetric_dia > dense > dia
            if self.symmetric_dia_data is not None:
                self.ata_matrix_format = "symmetric_dia"
                logger.info("Using symmetric DIA format (preferred)")
            elif self.dense_ata_data is not None:
                self.ata_matrix_format = "dense"
                logger.info("Using dense format")
            elif self.dia_matrix_data is not None:
                self.ata_matrix_format = "dia"
                logger.info("Using DIA format")

            # Load ATA inverse data if available
            self.ata_inverse_data = data.get("ata_inverse", None)
            logger.debug(f"Loaded ata_inverse_data: {type(self.ata_inverse_data)}")
            if self.ata_inverse_data is not None:
                logger.debug(f"ATA inverse shape: {getattr(self.ata_inverse_data, 'shape', 'no shape')}")
                logger.debug(f"ATA inverse dtype: {getattr(self.ata_inverse_data, 'dtype', 'no dtype')}")

            # Don't preload all patterns - generate on demand to save memory
            logger.info("Mixed tensor loaded successfully - patterns will be generated on demand")

            return True

        except Exception as e:
            logger.error(f"Failed to load nested format: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def _setup_routes(self):
        """Setup Flask routes for the new wrapper-based visualization."""

        @self.app.route("/")
        def index():
            """Main visualization page."""
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/api/metadata")
        def get_metadata():
            """Get pattern metadata using mixed tensor."""
            if not self.mixed_tensor:
                return jsonify({"error": "No patterns loaded"}), 404

            # Get actual LED count from wrapper objects
            actual_led_count = self.mixed_tensor.batch_size

            metadata_response = {
                "metadata": self.metadata,
                "led_count": actual_led_count,
                "frame_width": self.mixed_tensor.width,
                "frame_height": self.mixed_tensor.height,
                "channels": self.mixed_tensor.channels,
                "patterns_loaded": True,
                "has_both_formats": False,  # No longer using CSC format
                "supports_storage_order": self.led_spatial_mapping is not None,
            }

            # Add mixed tensor info using wrapper methods
            if self.mixed_tensor:
                block_summary = self.mixed_tensor.get_block_summary()
                memory_info = self.mixed_tensor.memory_info()

                # Handle mixed tensor dtype (could be class or dtype object)
                mixed_dtype = self.mixed_tensor.dtype
                if hasattr(mixed_dtype, "name"):
                    mixed_dtype_str = mixed_dtype.name
                elif hasattr(mixed_dtype, "__name__"):
                    mixed_dtype_str = mixed_dtype.__name__
                else:
                    mixed_dtype_str = str(mixed_dtype).replace("<class 'numpy.", "").replace("'>", "")

                metadata_response["mixed_tensor_info"] = {
                    "batch_size": int(block_summary["batch_size"]),
                    "channels": int(block_summary["channels"]),
                    "height": int(self.mixed_tensor.height),
                    "width": int(self.mixed_tensor.width),
                    "block_size": int(block_summary["block_size"]),
                    "total_blocks": int(block_summary["total_blocks"]),
                    "memory_mb": float(memory_info["total_mb"]),
                    "dtype": mixed_dtype_str,
                    "sparse_values_dtype": (
                        self.mixed_tensor.sparse_values.dtype.name
                        if hasattr(self.mixed_tensor.sparse_values.dtype, "name")
                        else str(self.mixed_tensor.sparse_values.dtype)
                    ),
                }

            # Add ATA matrix information based on format
            metadata_response["ata_matrix_format"] = self.ata_matrix_format

            if self.ata_matrix_format == "dia" and self.dia_matrix_data:
                metadata_response["dia_matrix_info"] = {
                    "led_count": int(self.dia_matrix_data["led_count"]),
                    "bandwidth": int(self.dia_matrix_data["bandwidth"]),
                    "k_diagonals": int(self.dia_matrix_data["k"]),
                    "sparsity": float(self.dia_matrix_data["sparsity"]),
                    "version": str(self.dia_matrix_data["version"]),
                }
                if "dia_data_3d" in self.dia_matrix_data:
                    dia_shape = self.dia_matrix_data["dia_data_3d"].shape
                    metadata_response["dia_matrix_info"]["dia_data_shape"] = list(dia_shape)
                    metadata_response["dia_matrix_info"]["memory_mb"] = float(
                        self.dia_matrix_data["dia_data_3d"].nbytes / (1024 * 1024)
                    )
                    # Get proper dtype name
                    dia_dtype = self.dia_matrix_data["dia_data_3d"].dtype
                    metadata_response["dia_matrix_info"]["dtype"] = (
                        dia_dtype.name if hasattr(dia_dtype, "name") else str(dia_dtype)
                    )

                # Add storage/output dtype info if available (for mixed precision support)
                if "storage_dtype" in self.dia_matrix_data:
                    metadata_response["dia_matrix_info"]["storage_dtype"] = str(self.dia_matrix_data["storage_dtype"])
                if "output_dtype" in self.dia_matrix_data:
                    metadata_response["dia_matrix_info"]["output_dtype"] = str(self.dia_matrix_data["output_dtype"])

            elif self.ata_matrix_format == "dense" and self.dense_ata_data:
                # Get matrix shape from data or compute it
                led_count = int(self.dense_ata_data["led_count"])
                channels = int(self.dense_ata_data["channels"])
                matrix_shape = self.dense_ata_data.get("matrix_shape", [channels, led_count, led_count])

                metadata_response["dense_ata_info"] = {
                    "led_count": led_count,
                    "channels": channels,
                    "memory_mb": float(self.dense_ata_data["memory_mb"]),
                    "version": str(self.dense_ata_data["version"]),
                    "matrix_shape": [int(x) for x in matrix_shape],
                }
                # Add storage/output dtype info
                if "storage_dtype" in self.dense_ata_data:
                    metadata_response["dense_ata_info"]["storage_dtype"] = str(self.dense_ata_data["storage_dtype"])
                if "output_dtype" in self.dense_ata_data:
                    metadata_response["dense_ata_info"]["output_dtype"] = str(self.dense_ata_data["output_dtype"])

            elif self.ata_matrix_format == "symmetric_dia" and self.symmetric_dia_data:
                # Symmetric DIA matrix info
                metadata_response["symmetric_dia_info"] = {
                    "led_count": int(self.symmetric_dia_data["led_count"]),
                    "channels": int(self.symmetric_dia_data["channels"]),
                    "k_upper": int(self.symmetric_dia_data["k_upper"]),
                    "bandwidth": int(self.symmetric_dia_data["bandwidth"]),
                    "sparsity": float(self.symmetric_dia_data["sparsity"]),
                    "nnz": int(self.symmetric_dia_data["nnz"]),
                }
                if "dia_data_gpu" in self.symmetric_dia_data:
                    dia_shape = self.symmetric_dia_data["dia_data_gpu"].shape
                    metadata_response["symmetric_dia_info"]["dia_data_shape"] = list(dia_shape)
                    metadata_response["symmetric_dia_info"]["memory_mb"] = float(
                        self.symmetric_dia_data["dia_data_gpu"].nbytes / (1024 * 1024)
                    )
                    # Get proper dtype name
                    dia_dtype = self.symmetric_dia_data["dia_data_gpu"].dtype
                    metadata_response["symmetric_dia_info"]["dtype"] = (
                        dia_dtype.name if hasattr(dia_dtype, "name") else str(dia_dtype)
                    )
                # Add memory reduction info
                if "original_k" in self.symmetric_dia_data:
                    original_k = int(self.symmetric_dia_data["original_k"])
                    k_upper = int(self.symmetric_dia_data["k_upper"])
                    if original_k and k_upper:
                        reduction = (1.0 - k_upper / original_k) * 100
                        metadata_response["symmetric_dia_info"]["memory_reduction_percent"] = float(reduction)
                        metadata_response["symmetric_dia_info"]["original_k"] = original_k
                # Add output dtype info
                if "output_dtype" in self.symmetric_dia_data:
                    metadata_response["symmetric_dia_info"]["output_dtype"] = str(
                        self.symmetric_dia_data["output_dtype"]
                    )

            # Add ATA inverse info if available
            if self.ata_inverse_data is not None:
                metadata_response["ata_inverse_info"] = {
                    "shape": [int(x) for x in self.ata_inverse_data.shape],
                    "memory_mb": float(self.ata_inverse_data.nbytes / (1024 * 1024)),
                    "dtype": str(self.ata_inverse_data.dtype),
                }

            return jsonify(metadata_response)

        @self.app.route("/api/patterns")
        def get_patterns():
            """Get pattern list with thumbnails using on-demand pattern loading."""
            if not self.mixed_tensor:
                return jsonify({"error": "No patterns loaded"}), 404

            try:
                page = int(request.args.get("page", 0))
                per_page = int(request.args.get("per_page", 50))
                channel = request.args.get("channel", "all")
                order = request.args.get("order", "numerical")  # "numerical" or "storage"
                format_type = request.args.get("format", "mixed")  # "mixed" format

                # Use mixed tensor as authoritative for LED count
                actual_led_count = self.mixed_tensor.batch_size

                patterns = []
                start_idx = page * per_page
                end_idx = min(start_idx + per_page, actual_led_count)

                # Create LED display information based on requested ordering
                if order == "storage" and self.led_spatial_mapping:
                    # Storage order: show LEDs in matrix/file column order (spatial indices)
                    # Dense patterns are indexed by spatial indices, so use directly
                    display_info = [{"spatial_idx": i, "display_id": i} for i in range(actual_led_count)]
                else:
                    # Numerical order: show LEDs in physical ID order (0, 1, 2, ...)
                    # Need to map from physical IDs to spatial indices for array access
                    if self.led_spatial_mapping:
                        reverse_mapping = {
                            matrix_idx: physical_id for physical_id, matrix_idx in self.led_spatial_mapping.items()
                        }
                        # Sort by physical ID for display, but keep track of spatial index for array access
                        display_info = [
                            {"spatial_idx": i, "display_id": reverse_mapping.get(i, i)} for i in range(actual_led_count)
                        ]
                        # Sort by display_id (physical LED ID) for numerical order
                        display_info.sort(key=lambda x: x["display_id"])
                    else:
                        display_info = [{"spatial_idx": i, "display_id": i} for i in range(actual_led_count)]

                # Get the LEDs for this page
                page_info = display_info[start_idx:end_idx]

                format_label = "Mixed Tensor"

                for info in page_info:
                    spatial_idx = info["spatial_idx"]
                    display_id = info["display_id"]

                    # Get pattern on demand
                    pattern = self._get_pattern(spatial_idx)  # Shape: (height, width, 3)

                    if channel == "all":
                        # Use full RGB pattern
                        rgb_pattern = pattern
                    else:
                        # Single channel - extract the specific channel
                        channel_idx = {"red": 0, "green": 1, "blue": 2}.get(channel, 0)
                        single_channel = pattern[:, :, channel_idx]  # (height, width)
                        rgb_pattern = np.stack([single_channel] * 3, axis=-1)

                    # Create thumbnail
                    thumbnail = self._create_thumbnail(rgb_pattern, size=(150, 90))

                    # Get physical LED ID for this spatial index
                    if self.led_spatial_mapping:
                        reverse_mapping = {
                            matrix_idx: physical_id for physical_id, matrix_idx in self.led_spatial_mapping.items()
                        }
                        physical_led_id = reverse_mapping.get(spatial_idx, spatial_idx)
                    else:
                        physical_led_id = spatial_idx

                    patterns.append(
                        {
                            "led_id": int(display_id),  # For compatibility with existing click handlers
                            "spatial_idx": int(spatial_idx),  # Column index in matrix
                            "physical_led_id": int(physical_led_id),  # Original LED numbering
                            "thumbnail": thumbnail,
                            "max_intensity": float(np.max(rgb_pattern)),
                            "center_of_mass": self._calculate_center_of_mass(rgb_pattern).tolist(),
                        }
                    )

                return jsonify(
                    {
                        "patterns": patterns,
                        "page": int(page),
                        "per_page": int(per_page),
                        "total_leds": int(actual_led_count),
                        "total_pages": int((actual_led_count + per_page - 1) // per_page),
                        "order": order,
                        "format": format_type,
                        "format_label": format_label,
                        "supports_storage_order": self.led_spatial_mapping is not None,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to generate patterns: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/pattern/<int:led_id>")
        def get_pattern_detail(led_id):
            """Get detailed view of specific LED pattern."""
            if not self.mixed_tensor:
                return jsonify({"error": "No patterns loaded"}), 404

            # Get ordering mode to interpret led_id correctly
            order = request.args.get("order", "numerical")

            actual_led_count = self.mixed_tensor.batch_size

            # Map display ID to spatial index for array access
            if order == "storage" or not self.led_spatial_mapping:
                # Storage order: led_id is already a spatial index
                spatial_idx = led_id
                display_id = led_id
            else:
                # Numerical order: led_id is a physical ID, need to find spatial index
                spatial_idx = self.led_spatial_mapping.get(led_id, led_id)
                display_id = led_id

            if spatial_idx >= actual_led_count:
                return jsonify({"error": "Invalid LED ID"}), 400

            try:
                # Get physical LED ID for this spatial index
                if self.led_spatial_mapping:
                    reverse_mapping = {
                        matrix_idx: physical_id for physical_id, matrix_idx in self.led_spatial_mapping.items()
                    }
                    physical_led_id = reverse_mapping.get(spatial_idx, spatial_idx)
                else:
                    physical_led_id = spatial_idx

                pattern_data = {
                    "led_id": int(display_id),
                    "spatial_idx": int(spatial_idx),
                    "physical_led_id": int(physical_led_id),
                    "channels": {},
                    "formats": {},
                }

                channel_names = ["red", "green", "blue"]

                # Get pattern on demand
                full_pattern = self._get_pattern(spatial_idx)  # Shape: (height, width, 3)
                format_data = {"channels": {}}

                for ch_idx, ch_name in enumerate(channel_names):
                    pattern = full_pattern[:, :, ch_idx]

                    # Full resolution image
                    full_image = self._pattern_to_base64(pattern)

                    # Statistics
                    stats = {
                        "max_intensity": float(np.max(pattern)),
                        "min_intensity": float(np.min(pattern)),
                        "mean_intensity": float(np.mean(pattern)),
                        "std_intensity": float(np.std(pattern)),
                        "center_of_mass": self._calculate_center_of_mass(pattern).tolist(),
                    }

                    format_data["channels"][ch_name] = {
                        "image": full_image,
                        "statistics": stats,
                    }

                # Create composite RGB view for this format
                rgb_pattern = full_pattern  # Already in (height, width, 3) format

                format_data["composite"] = {
                    "image": self._create_full_image(rgb_pattern),
                    "statistics": {
                        "max_intensity": float(np.max(rgb_pattern)),
                        "center_of_mass": self._calculate_center_of_mass(rgb_pattern).tolist(),
                    },
                }

                pattern_data["formats"]["mixed"] = format_data

                return jsonify(pattern_data)

            except Exception as e:
                logger.error(f"Failed to get pattern detail: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/ata_matrix/<int:channel>")
        def get_ata_matrix(channel):
            """Get ATA matrix visualization for a specific channel."""
            if self.ata_matrix_format is None:
                return jsonify({"error": "No ATA matrix data available"}), 404

            if channel < 0 or channel >= 3:
                return jsonify({"error": "Invalid channel, must be 0, 1, or 2"}), 400

            try:
                if self.ata_matrix_format == "dia":
                    # Get the DIA matrix data for this channel
                    dia_data = self.dia_matrix_data["dia_data_3d"][channel]  # Shape: (k, led_count)
                    dia_offsets = self.dia_matrix_data["dia_offsets_3d"]  # Shape: (k,)
                    led_count = self.dia_matrix_data["led_count"]

                    # Convert DIA format to dense for visualization
                    dense_matrix = self._dia_to_dense(dia_data, dia_offsets, led_count)

                    # Use the extracted function to create image response
                    response_data = self._ata_matrix_to_image(dense_matrix, channel)

                    # Add DIA-specific metadata
                    response_data.update(
                        {
                            "format": "dia",
                            "bandwidth": self.dia_matrix_data["bandwidth"],
                            "k_diagonals": self.dia_matrix_data["k"],
                            "sparsity": self.dia_matrix_data["sparsity"],
                        }
                    )

                elif self.ata_matrix_format == "dense":
                    # Get the dense matrix data for this channel
                    dense_matrices = self.dense_ata_data["dense_matrices"]  # Shape: (3, led_count, led_count)
                    dense_matrix = dense_matrices[channel]  # Shape: (led_count, led_count)

                    # Use the extracted function to create image response
                    response_data = self._ata_matrix_to_image(dense_matrix, channel)

                    # Add dense-specific metadata
                    led_count = int(self.dense_ata_data["led_count"])
                    channels = int(self.dense_ata_data["channels"])
                    matrix_shape = [int(channels), int(led_count), int(led_count)]

                    response_data.update(
                        {
                            "format": "dense",
                            "memory_mb": float(self.dense_ata_data["memory_mb"]),
                            "matrix_shape": matrix_shape,
                        }
                    )

                elif self.ata_matrix_format == "symmetric_dia":
                    # Get the symmetric DIA matrix data for this channel
                    dia_data = self.symmetric_dia_data["dia_data_gpu"][channel]  # Shape: (k_upper, led_count)
                    dia_offsets_upper = self.symmetric_dia_data["dia_offsets_upper"]  # Shape: (k_upper,)
                    led_count = self.symmetric_dia_data["led_count"]

                    # Convert symmetric DIA format to dense for visualization
                    dense_matrix = self._symmetric_dia_to_dense(dia_data, dia_offsets_upper, led_count)

                    # Use the extracted function to create image response
                    response_data = self._ata_matrix_to_image(dense_matrix, channel)

                    # Add symmetric DIA-specific metadata
                    response_data.update(
                        {
                            "format": "symmetric_dia",
                            "bandwidth": int(self.symmetric_dia_data["bandwidth"]),
                            "k_upper": int(self.symmetric_dia_data["k_upper"]),
                            "sparsity": float(self.symmetric_dia_data["sparsity"]),
                            "nnz": int(self.symmetric_dia_data["nnz"]),
                            "memory_reduction_percent": (
                                (
                                    1.0
                                    - int(self.symmetric_dia_data["k_upper"])
                                    / int(self.symmetric_dia_data["original_k"])
                                )
                                * 100
                                if self.symmetric_dia_data.get("original_k")
                                else None
                            ),
                        }
                    )

                return jsonify(response_data)

            except Exception as e:
                logger.error(f"Failed to get ATA matrix: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/ata_inverse_raw/<int:channel>")
        def get_ata_inverse_raw(channel):
            """Return raw matrix sample without any processing."""
            if self.ata_inverse_data is None:
                return jsonify({"error": "No ATA inverse data available"}), 404

            if channel < 0 or channel >= 3:
                return jsonify({"error": "Invalid channel"}), 400

            try:
                # Get raw matrix like the main endpoint
                if len(self.ata_inverse_data.shape) == 2:
                    inverse_matrix = self.ata_inverse_data
                elif len(self.ata_inverse_data.shape) == 3:
                    inverse_matrix = self.ata_inverse_data[channel]
                else:
                    return jsonify({"error": f"Invalid shape: {self.ata_inverse_data.shape}"}), 500

                # Return raw sample of actual matrix values
                sample_size = 10
                sample = inverse_matrix[:sample_size, :sample_size]

                return jsonify(
                    {
                        "raw_sample_10x10": sample.tolist(),
                        "sample_min": float(np.min(sample)),
                        "sample_max": float(np.max(sample)),
                        "sample_dtype": str(sample.dtype),
                        "matrix_shape": list(inverse_matrix.shape),
                        "matrix_min": float(np.min(inverse_matrix)),
                        "matrix_max": float(np.max(inverse_matrix)),
                        "matrix_abs_max": float(np.max(np.abs(inverse_matrix))),
                    }
                )

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/ata_inverse_debug/<int:channel>")
        def get_ata_inverse_debug(channel):
            """Debug endpoint - return raw matrix stats instead of image."""
            if self.ata_inverse_data is None:
                return jsonify({"error": "No ATA inverse data available"}), 404

            if channel < 0 or channel >= 3:
                return jsonify({"error": "Invalid channel"}), 400

            try:
                # Get matrix data like main endpoint
                if len(self.ata_inverse_data.shape) == 2:
                    inverse_matrix = self.ata_inverse_data
                elif len(self.ata_inverse_data.shape) == 3:
                    inverse_matrix = self.ata_inverse_data[channel]
                else:
                    return jsonify({"error": f"Invalid shape: {self.ata_inverse_data.shape}"}), 500

                # Return raw statistics instead of image
                stats = {
                    "matrix_shape": list(inverse_matrix.shape),
                    "matrix_dtype": str(inverse_matrix.dtype),
                    "min_value": float(np.min(inverse_matrix)),
                    "max_value": float(np.max(inverse_matrix)),
                    "nonzero_count": int(np.count_nonzero(inverse_matrix)),
                    "total_elements": int(inverse_matrix.size),
                    "sample_values": inverse_matrix.flat[:10].tolist(),
                    "abs_max": float(np.max(np.abs(inverse_matrix))),
                }

                # Test normalization
                matrix_abs = np.abs(inverse_matrix)
                max_abs = np.max(matrix_abs)
                if max_abs > 0:
                    normalized = (matrix_abs / max_abs * 255).astype(np.uint8)
                    stats["normalized_min"] = int(np.min(normalized))
                    stats["normalized_max"] = int(np.max(normalized))
                    stats["normalized_nonzero"] = int(np.count_nonzero(normalized))
                    stats["normalized_sample"] = normalized.flat[:10].tolist()

                return jsonify(stats)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/ata_inverse/<int:channel>")
        def get_ata_inverse(channel):
            """Get ATA inverse matrix visualization for a specific channel."""
            if self.ata_inverse_data is None:
                return jsonify({"error": "No ATA inverse data available"}), 404

            if channel < 0 or channel >= 3:
                return jsonify({"error": "Invalid channel, must be 0, 1, or 2"}), 400

            try:
                # Get the inverse matrix for this channel
                if len(self.ata_inverse_data.shape) == 2:
                    # 2D matrix - same for all channels
                    inverse_matrix = self.ata_inverse_data
                elif len(self.ata_inverse_data.shape) == 3:
                    # 3D matrix - per-channel
                    inverse_matrix = self.ata_inverse_data[channel]
                else:
                    return jsonify({"error": f"Invalid ATA inverse shape: {self.ata_inverse_data.shape}"}), 500

                # Force the same processing path as ATA matrices by artificially making it sparse
                # Create a copy and manually increase sparsity by zeroing out small values
                inverse_matrix_copy = inverse_matrix.copy()
                abs_matrix = np.abs(inverse_matrix_copy)

                # Set threshold to make matrix appear sparse (>90% zeros)
                threshold_percentile = 95  # Zero out bottom 95% of values by magnitude
                threshold = np.percentile(abs_matrix[abs_matrix > 0], threshold_percentile)
                inverse_matrix_copy[abs_matrix < threshold] = 0

                # Now it should trigger sparse matrix visualization path
                response_data = self._ata_matrix_to_image(inverse_matrix_copy, channel)

                # Add ATA inverse specific metadata (using float32 for linalg)
                inverse_matrix_f32 = inverse_matrix.astype(np.float32)
                response_data.update(
                    {
                        "condition_number": float(np.linalg.cond(inverse_matrix_f32)),
                        "determinant": float(np.linalg.det(inverse_matrix_f32)),
                    }
                )

                return jsonify(response_data)

            except Exception as e:
                logger.error(f"Failed to get ATA inverse matrix: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/image_pairs")
        def get_image_pairs():
            """Get list of source/optimized image pairs."""
            try:
                pairs = self._find_image_pairs()
                return jsonify(
                    {
                        "pairs": pairs,
                        "total_pairs": len(pairs),
                        "source_dir": str(self.source_images_dir),
                        "optimized_dir": str(self.optimized_images_dir),
                    }
                )
            except Exception as e:
                logger.error(f"Failed to get image pairs: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/image/<path:image_name>")
        def serve_image(image_name):
            """Serve an image file from source or optimized directories."""
            try:
                # Check if it's a source or optimized image
                if (
                    image_name.endswith("_optimized.jpg")
                    or image_name.endswith("_optimized.png")
                    or image_name.endswith("_optimized.jpeg")
                ):
                    # Optimized image - automatically generate if not exists
                    image_path = self.optimized_images_dir / image_name
                    if not image_path.exists():
                        # Automatically generate optimized version
                        source_name = image_name.replace("_optimized", "")
                        logger.info(f"Optimized image not found, automatically generating for: {source_name}")
                        generated_path = self._generate_optimized_image(source_name)
                        if generated_path and generated_path.exists():
                            image_path = generated_path
                        else:
                            return jsonify({"error": "Could not generate optimized image"}), 404
                else:
                    # Source image
                    image_path = self.source_images_dir / image_name

                if not image_path.exists():
                    return jsonify({"error": "Image not found"}), 404

                return send_file(image_path)

            except Exception as e:
                logger.error(f"Failed to serve image {image_name}: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/generate_optimized/<path:source_name>", methods=["POST"])
        def generate_optimized_image(source_name):
            """Generate optimized version of a source image."""
            try:
                generated_path = self._generate_optimized_image(source_name)
                if generated_path and generated_path.exists():
                    return jsonify(
                        {
                            "success": True,
                            "optimized_path": f"/api/image/{generated_path.name}",
                            "optimized_name": generated_path.name,
                        }
                    )
                else:
                    return jsonify({"error": "Failed to generate optimized image"}), 500

            except Exception as e:
                logger.error(f"Failed to generate optimized image for {source_name}: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/regenerate_all_patterns", methods=["POST"])
        def regenerate_all_patterns():
            """Regenerate all optimized images for all source images."""
            try:
                # Get all source images
                source_files = []
                if self.source_images_dir.exists():
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
                        source_files.extend(self.source_images_dir.glob(ext))
                        source_files.extend(self.source_images_dir.glob(ext.upper()))

                if not source_files:
                    return jsonify({"error": "No source images found"}), 404

                # Regenerate all patterns
                results = []
                total_files = len(source_files)
                successful = 0
                failed = 0

                logger.info(f"Regenerating optimized images for {total_files} source images...")

                for i, source_file in enumerate(source_files):
                    source_name = source_file.name
                    logger.info(f"Processing {i+1}/{total_files}: {source_name}")

                    try:
                        # Generate optimized image (this will overwrite existing ones)
                        generated_path = self._generate_optimized_image(source_name)
                        if generated_path and generated_path.exists():
                            results.append(
                                {"source_name": source_name, "status": "success", "optimized_name": generated_path.name}
                            )
                            successful += 1
                        else:
                            results.append(
                                {"source_name": source_name, "status": "failed", "error": "Generation failed"}
                            )
                            failed += 1
                    except Exception as e:
                        results.append({"source_name": source_name, "status": "failed", "error": str(e)})
                        failed += 1
                        logger.error(f"Failed to generate optimized image for {source_name}: {e}")

                return jsonify(
                    {
                        "success": True,
                        "total": total_files,
                        "successful": successful,
                        "failed": failed,
                        "results": results,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to regenerate all patterns: {e}")
                return jsonify({"error": str(e)}), 500

    def _create_thumbnail(self, pattern: np.ndarray, size: Tuple[int, int]) -> str:
        """Create base64 thumbnail from pattern."""
        try:
            if not PIL_AVAILABLE:
                return ""

            # Normalize pattern to 0-255 based on data type
            if len(pattern.shape) == 3:
                # RGB pattern
                if self.mixed_tensor.dtype == cp.uint8:
                    # Data is already in [0,255] range
                    normalized = np.clip(pattern, 0, 255).astype(np.uint8)
                else:
                    # Data is in [0,1] range, scale to [0,255]
                    normalized = np.clip(pattern * 255, 0, 255).astype(np.uint8)
            else:
                # Single channel - convert to RGB
                if self.mixed_tensor.dtype == cp.uint8:
                    # Data is already in [0,255] range
                    normalized = np.clip(pattern, 0, 255).astype(np.uint8)
                else:
                    # Data is in [0,1] range, scale to [0,255]
                    normalized = np.clip(pattern * 255, 0, 255).astype(np.uint8)
                normalized = np.stack([normalized] * 3, axis=-1)

            # Add green borders around 64x64 blocks
            normalized_with_borders = self._add_block_borders(normalized, block_size=64)

            # Create PIL image
            img = Image.fromarray(normalized_with_borders, mode="RGB")

            # Resize to thumbnail size
            img = img.resize(size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            logger.warning(f"Failed to create thumbnail: {e}")
            return ""

    def _create_full_image(self, pattern: np.ndarray) -> str:
        """Create full resolution base64 image from RGB pattern."""
        try:
            if not PIL_AVAILABLE:
                return ""

            # Normalize to 0-255 based on data type
            if self.mixed_tensor.dtype == cp.uint8:
                # Data is already in [0,255] range
                normalized = np.clip(pattern, 0, 255).astype(np.uint8)
            else:
                # Data is in [0,1] range, scale to [0,255]
                normalized = np.clip(pattern * 255, 0, 255).astype(np.uint8)

            # Add green borders around 64x64 blocks
            normalized_with_borders = self._add_block_borders(normalized, block_size=64)

            # Create PIL image
            img = Image.fromarray(normalized_with_borders, mode="RGB")

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            logger.warning(f"Failed to create full image: {e}")
            return ""

    def _pattern_to_base64(self, pattern: np.ndarray) -> str:
        """Convert single channel pattern to base64 grayscale image."""
        try:
            if not PIL_AVAILABLE:
                return ""

            # Normalize to 0-255 based on data type
            if self.mixed_tensor.dtype == cp.uint8:
                # Data is already in [0,255] range
                normalized = np.clip(pattern, 0, 255).astype(np.uint8)
            else:
                # Data is in [0,1] range, scale to [0,255]
                normalized = np.clip(pattern * 255, 0, 255).astype(np.uint8)

            # Convert grayscale to RGB for border drawing
            normalized_rgb = np.stack([normalized] * 3, axis=-1)

            # Add green borders around 64x64 blocks
            normalized_with_borders = self._add_block_borders(normalized_rgb, block_size=64)

            # Create PIL image from RGB array
            img = Image.fromarray(normalized_with_borders, mode="RGB")

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            logger.warning(f"Failed to convert pattern to base64: {e}")
            return ""

    def _calculate_center_of_mass(self, pattern: np.ndarray) -> np.ndarray:
        """Calculate center of mass for pattern."""
        try:
            if len(pattern.shape) == 3:
                # Multi-channel - use combined intensity
                intensity = np.mean(pattern, axis=-1)
            else:
                intensity = pattern

            # Calculate center of mass
            total_mass = np.sum(intensity)
            if total_mass == 0:
                return np.array(
                    [
                        self.mixed_tensor.width // 2,
                        self.mixed_tensor.height // 2,
                    ]
                )

            y_indices, x_indices = np.indices(intensity.shape)
            x_cm = np.sum(x_indices * intensity) / total_mass
            y_cm = np.sum(y_indices * intensity) / total_mass

            return np.array([x_cm, y_cm])

        except Exception:
            return np.array([self.mixed_tensor.width // 2, self.mixed_tensor.height // 2])

    def _to_numpy(self, array) -> np.ndarray:
        """Convert CuPy or NumPy array to NumPy array."""
        if hasattr(array, "__cuda_array_interface__"):
            # It's a CuPy array
            return cp.asnumpy(array)
        else:
            # It's already a NumPy array or compatible
            return np.asarray(array)

    def _get_pattern(self, led_id: int) -> np.ndarray:
        """Get pattern for a specific LED (with caching)."""
        if led_id in self.pattern_cache:
            return self.pattern_cache[led_id]

        # Extract pattern from mixed tensor for all RGB channels
        r_pattern = self.mixed_tensor.to_array(led_id, 0)  # Red channel
        g_pattern = self.mixed_tensor.to_array(led_id, 1)  # Green channel
        b_pattern = self.mixed_tensor.to_array(led_id, 2)  # Blue channel

        # Convert to numpy arrays for visualization
        r_pattern = self._to_numpy(r_pattern)
        g_pattern = self._to_numpy(g_pattern)
        b_pattern = self._to_numpy(b_pattern)

        # Combine into RGB pattern
        pattern = np.stack([r_pattern, g_pattern, b_pattern], axis=2)  # Shape: (height, width, 3)

        # Cache the pattern
        self.pattern_cache[led_id] = pattern

        # Limit cache size to avoid memory issues
        if len(self.pattern_cache) > 100:  # Keep only 100 patterns cached
            # Remove oldest entry
            oldest_key = next(iter(self.pattern_cache))
            del self.pattern_cache[oldest_key]

        return pattern

    def _generate_optimized_image(self, source_name: str) -> Optional[Path]:
        """Generate optimized image using frame optimizer."""
        try:
            # Check if required modules are available
            if not FRAME_OPTIMIZER_AVAILABLE or not DIAGONAL_ATA_AVAILABLE:
                logger.error("Frame optimizer or DiagonalATAMatrix not available")
                return None

            # Check if we have the required data loaded
            if self.mixed_tensor is None or self.ata_inverse_data is None or self.ata_matrix_format is None:
                logger.error("Missing required optimization data (mixed_tensor, ata_inverse, or ata_matrix)")
                return None

            # Find source image
            source_path = self.source_images_dir / source_name
            if not source_path.exists():
                logger.error(f"Source image not found: {source_path}")
                return None

            # Load and preprocess source image
            logger.info(f"Loading source image: {source_path}")
            from PIL import Image

            with Image.open(source_path) as img:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize to target frame size (480x800)
                img_resized = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.LANCZOS)

                # Convert to numpy array in HWC format
                target_frame_np = np.array(img_resized, dtype=np.uint8)  # Shape: (480, 800, 3)

                # Convert to GPU array for the optimizer
                target_frame = cp.asarray(target_frame_np)

            logger.info(f"Target frame shape: {target_frame.shape}")

            # Create ATA matrix from loaded data based on format
            if self.ata_matrix_format == "dia":
                ata_matrix = DiagonalATAMatrix.from_dict(self.dia_matrix_data)
                logger.info(f"Using DIA format ATA matrix: {ata_matrix.led_count} LEDs, {ata_matrix.k} diagonals")
            elif self.ata_matrix_format == "dense":
                ata_matrix = DenseATAMatrix.from_dict(self.dense_ata_data)
                logger.info(f"Using dense format ATA matrix: {ata_matrix.led_count} LEDs, {ata_matrix.memory_mb:.1f}MB")
            elif self.ata_matrix_format == "symmetric_dia":
                # Create SymmetricDiagonalATAMatrix from stored data
                ata_matrix = SymmetricDiagonalATAMatrix(
                    led_count=self.symmetric_dia_data["led_count"],
                    crop_size=self.symmetric_dia_data.get("crop_size", 64),
                    output_dtype=cp.float32,
                )
                # Restore the internal state from saved data
                ata_matrix.k_upper = self.symmetric_dia_data["k_upper"]
                ata_matrix.dia_offsets_upper = self.symmetric_dia_data["dia_offsets_upper"]
                ata_matrix.bandwidth = self.symmetric_dia_data["bandwidth"]
                ata_matrix.sparsity = self.symmetric_dia_data["sparsity"]
                ata_matrix.nnz = self.symmetric_dia_data["nnz"]
                ata_matrix.original_k = self.symmetric_dia_data.get("original_k")
                # Convert GPU data back to CuPy arrays with proper data types
                ata_matrix.dia_data_gpu = cp.asarray(self.symmetric_dia_data["dia_data_gpu"], dtype=cp.float32)
                ata_matrix.dia_offsets_upper_gpu = cp.asarray(ata_matrix.dia_offsets_upper, dtype=cp.int32)

                logger.info(
                    f"Using symmetric DIA format ATA matrix: {ata_matrix.led_count} LEDs, {ata_matrix.k_upper} upper diagonals, {ata_matrix.bandwidth} bandwidth"
                )
            else:
                logger.error(f"Unknown ATA matrix format: {self.ata_matrix_format}")
                return None

            # Run frame optimization using the production frame_optimizer
            logger.info("Running frame optimization...")

            # For dense format, we still use the numpy array ATA inverse
            # The DenseATAMatrix is only for the forward ATA matrix, not the inverse
            ata_inverse_to_use = self.ata_inverse_data

            result = optimize_frame_led_values(
                target_frame=target_frame,
                at_matrix=self.mixed_tensor,
                ata_matrix=ata_matrix,
                ata_inverse=ata_inverse_to_use,
                max_iterations=10,
                convergence_threshold=0.3,
                step_size_scaling=0.9,
                compute_error_metrics=True,
                debug=True,
            )

            logger.info(f"Optimization completed: converged={result.converged}, iterations={result.iterations}")

            # result.led_values is in range [0, 255] uint8 from frame optimizer
            # Convert to [0, 1] range for forward_pass_3d
            led_values_float32 = result.led_values.astype(np.float32) / 255.0  # Shape: (3, led_count)

            # Render LED values back to frame using diffusion patterns
            optimized_frame = self._render_led_values_to_frame(led_values_float32)

            # Save optimized image
            stem = source_path.stem
            suffix = source_path.suffix
            optimized_name = f"{stem}_optimized{suffix}"
            optimized_path = self.optimized_images_dir / optimized_name

            # Ensure optimized directory exists
            self.optimized_images_dir.mkdir(exist_ok=True)

            # Save as PIL image
            optimized_img = Image.fromarray(optimized_frame, mode="RGB")
            optimized_img.save(optimized_path)

            logger.info(f"Saved optimized image: {optimized_path}")
            return optimized_path

        except Exception as e:
            logger.error(f"Failed to generate optimized image: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _render_led_values_to_frame(self, led_values: np.ndarray) -> np.ndarray:
        """Render LED values back to frame using forward_pass_3d."""
        try:
            # led_values shape: (3, led_count) in range [0, 1] - planar format (converted from optimizer [0, 255])
            # Mixed tensor forward_pass_3d expects LED values in [0, 1] range
            led_values_float32 = led_values.astype(np.float32)

            logger.info(
                f"Rendering LED values (planar): shape={led_values_float32.shape}, range=[{led_values_float32.min():.3f}, {led_values_float32.max():.3f}]"
            )

            # Convert from planar (3, led_count) to interleaved (led_count, 3) for forward_pass_3d
            # Hardware expects interleaved format: (led_count, 3)
            led_values_interleaved = led_values_float32.T  # (3, 2624) -> (2624, 3)

            logger.info(
                f"Converted to interleaved: shape={led_values_interleaved.shape}, range=[{led_values_interleaved.min():.3f}, {led_values_interleaved.max():.3f}]"
            )

            # Debug: Check mixed tensor dtype
            logger.info(f"Mixed tensor dtype: {self.mixed_tensor.dtype}")
            logger.info(f"Mixed tensor sparse_values dtype: {self.mixed_tensor.sparse_values.dtype}")

            # Use forward_pass_3d to render all LEDs and channels at once
            # This is the same method used by the test renderer
            output_frame = self.mixed_tensor.forward_pass_3d(led_values_interleaved)

            # output_frame should be shape (3, height, width) in range [0, 1]
            logger.info(
                f"Forward pass output: shape={output_frame.shape}, range=[{output_frame.min():.3f}, {output_frame.max():.3f}]"
            )

            # Convert from planar (3, H, W) to interleaved (H, W, 3) format
            if output_frame.shape[0] == 3:
                output_frame = output_frame.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)

            # forward_pass_3d now outputs [0, 1] range - scale to [0, 255] for image
            output_frame = np.clip(output_frame, 0.0, 1.0)  # Ensure valid range
            output_frame = (output_frame * 255.0).astype(np.uint8)

            logger.info(f"Scaled to uint8 [0, 255]: [{output_frame.min()}, {output_frame.max()}]")

            # Ensure it's a NumPy array (CPU) for PIL compatibility
            if hasattr(output_frame, "get"):
                # It's a CuPy array - convert to NumPy
                output_frame = output_frame.get()
                logger.info("Converted CuPy array to NumPy for PIL compatibility")

            logger.info(
                f"Final rendered frame: shape={output_frame.shape}, range=[{output_frame.min()}, {output_frame.max()}]"
            )
            return output_frame

        except Exception as e:
            logger.error(f"Failed to render LED values to frame: {e}")
            raise

    def _ata_matrix_to_image(self, dense_matrix: np.ndarray, channel: int) -> dict:
        """Convert dense ATA matrix to image and return response data (extracted from working ATA code)."""
        try:
            # Create visualization image using the same method as working ATA matrices
            matrix_image = self._matrix_to_image(dense_matrix)

            return {
                "channel": channel,
                "matrix_shape": list(dense_matrix.shape),
                "matrix_image": matrix_image,
                "max_value": float(np.max(dense_matrix)),
                "min_value": float(np.min(dense_matrix)),
                "nnz": int(np.count_nonzero(dense_matrix)),
            }
        except Exception as e:
            logger.error(f"Failed to convert ATA matrix to image: {e}")
            raise

    def _dia_to_dense(self, dia_data: np.ndarray, dia_offsets: np.ndarray, size: int) -> np.ndarray:
        """Convert DIA format matrix to dense format for visualization."""
        dense = np.zeros((size, size), dtype=dia_data.dtype)

        for i, offset in enumerate(dia_offsets):
            diagonal = dia_data[i, :]

            if offset >= 0:
                # Upper diagonal
                diag_len = min(size - offset, size)
                for j in range(diag_len):
                    if j < len(diagonal) and j + offset < size:
                        dense[j, j + offset] = diagonal[j]
            else:
                # Lower diagonal
                diag_len = min(size + offset, size)
                for j in range(diag_len):
                    if j < len(diagonal) and j - offset < size:
                        dense[j - offset, j] = diagonal[j]

        return dense

    def _symmetric_dia_to_dense(self, dia_data: np.ndarray, dia_offsets_upper: np.ndarray, size: int) -> np.ndarray:
        """Convert symmetric DIA format matrix to dense format for visualization."""
        dense = np.zeros((size, size), dtype=dia_data.dtype)

        for i, offset in enumerate(dia_offsets_upper):
            diagonal = dia_data[i, :]

            # For symmetric DIA, we only store upper diagonals (offset >= 0)
            # But we need to fill both upper and lower parts (except main diagonal)

            # Fill upper diagonal: A[j, j+offset]
            diag_len = size - offset
            for j in range(diag_len):
                if j + offset < size and offset < len(diagonal):
                    # DIA format: diagonal data starts at column index = offset
                    dense[j, j + offset] = diagonal[j + offset]

            # Fill symmetric lower diagonal: A[j+offset, j] = A[j, j+offset]
            # Skip main diagonal (offset=0) to avoid double-counting
            if offset > 0:
                for j in range(diag_len):
                    if j + offset < size and offset < len(diagonal):
                        dense[j + offset, j] = diagonal[j + offset]

        return dense

    def _matrix_to_image(self, matrix: np.ndarray) -> str:
        """Convert matrix to base64 image for visualization using log magnitude color scale."""
        try:
            if not PIL_AVAILABLE:
                return ""

            # Debug logging
            logger.debug(f"Matrix shape: {matrix.shape}, dtype: {matrix.dtype}")
            logger.debug(f"Matrix min: {np.min(matrix)}, max: {np.max(matrix)}")

            # Always use log magnitude with color scale for ATA matrices
            matrix_abs = np.abs(matrix)
            nonzero_mask = matrix_abs > 0

            # Create log magnitude visualization
            if np.any(nonzero_mask):
                nonzero_values = matrix_abs[nonzero_mask]
                min_nonzero = np.min(nonzero_values)
                max_nonzero = np.max(nonzero_values)

                logger.debug(f"Value range: {min_nonzero:.2e} to {max_nonzero:.2e}")

                # Use a reasonable threshold to avoid log issues
                # Set floor at 1e-6 of the minimum non-zero value to distinguish from true zeros
                threshold = max(min_nonzero * 1e-6, 1e-15)
                matrix_clamped = np.where(matrix_abs < threshold, 0, matrix_abs)  # Keep zeros as zeros

                # Debug threshold and values
                original_nonzeros = np.sum(matrix_abs > 0)
                after_threshold_nonzeros = np.sum(matrix_clamped > 0)
                logger.debug(f"Threshold: {threshold:.2e}")
                logger.debug(f"Non-zeros before threshold: {original_nonzeros}, after: {after_threshold_nonzeros}")

                # Apply log10 transformation only to non-zero values
                log_matrix = np.zeros_like(matrix_abs, dtype=np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_values = np.log10(matrix_clamped)
                    # Only update non-zero positions
                    valid_mask = np.isfinite(log_values) & (matrix_clamped > 0)
                    log_matrix[valid_mask] = log_values[valid_mask]

                # Get log range for normalization (excluding zeros)
                if np.any(valid_mask):
                    log_min = np.min(log_matrix[valid_mask])
                    log_max = np.max(log_matrix[valid_mask])
                    logger.debug(f"Log range: {log_min:.2f} to {log_max:.2f}")

                    if log_max > log_min:
                        # Normalize log values to [0, 1] for non-zero elements
                        normalized_log = np.zeros_like(log_matrix)
                        normalized_log[valid_mask] = (log_matrix[valid_mask] - log_min) / (log_max - log_min)

                        # Create RGB color mapping: Red (low) -> Blue (high)
                        # ALL non-zero values get minimum intensity of 0.2, only true zeros are black
                        rgb_image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)

                        # Count non-zero elements for debugging
                        total_nonzero = np.sum(valid_mask)
                        logger.debug(f"Processing {total_nonzero} non-zero elements out of {matrix.size} total")

                        # Vectorized approach for better performance and debugging
                        # Get all normalized log values for non-zero elements
                        nonzero_vals = normalized_log[valid_mask]  # [0, 1] range

                        # Apply minimum intensity floor to ALL non-zero values
                        intensity_floor = 0.1  # 10% minimum intensity (was too high at 20%)

                        # Debug normalized log value distribution
                        logger.debug(
                            f"Normalized log values: min={np.min(nonzero_vals):.3f}, max={np.max(nonzero_vals):.3f}, mean={np.mean(nonzero_vals):.3f}"
                        )

                        # Check if values are clustered (causing purple issue)
                        percentiles = [10, 25, 50, 75, 90]
                        pct_vals = np.percentile(nonzero_vals, percentiles)
                        logger.debug(f"Percentiles {percentiles}: {pct_vals}")

                        # Calculate bandwidth and diagonal analysis
                        rows, cols = np.where(valid_mask)
                        if len(rows) > 0:
                            distances_from_diagonal = rows - cols  # Signed distance (positive = above diagonal)
                            max_distance = np.max(np.abs(distances_from_diagonal))  # Maximum absolute distance
                            min_offset = np.min(distances_from_diagonal)
                            max_offset = np.max(distances_from_diagonal)

                            # Count actual non-zero diagonals
                            unique_offsets = np.unique(distances_from_diagonal)
                            num_nonzero_diagonals = len(unique_offsets)

                            # DIA storage would need this many diagonals
                            dia_storage_diagonals = max_offset - min_offset + 1

                            diagonal_elements = np.sum(distances_from_diagonal == 0)
                            near_diagonal = np.sum(np.abs(distances_from_diagonal) <= 10)

                            logger.debug("Matrix bandwidth analysis:")
                            logger.debug(f"  Max distance from diagonal: {max_distance}")
                            logger.debug(f"  Diagonal offset range: {min_offset} to {max_offset}")
                            logger.debug(f"  Actual non-zero diagonals: {num_nonzero_diagonals}")
                            logger.debug(f"  DIA storage would need: {dia_storage_diagonals} diagonals")
                            logger.debug(f"  Main diagonal elements: {diagonal_elements}")
                            logger.debug(
                                f"  Near diagonal (≤10): {near_diagonal}/{len(rows)} ({near_diagonal/len(rows)*100:.1f}%)"
                            )
                            logger.debug(f"  Overall sparsity: {(len(rows) / matrix.size) * 100:.2f}% non-zero")
                        else:
                            max_distance = 0
                            num_nonzero_diagonals = 0
                            logger.debug("No non-zero elements found")

                        # Color mapping: Use aggressive spreading if values are concentrated
                        val_range = np.max(nonzero_vals) - np.min(nonzero_vals)
                        if val_range < 0.1:  # Very concentrated values
                            logger.debug("Values are very concentrated, using histogram equalization")
                            # Use histogram equalization to spread colors more evenly
                            sorted_indices = np.argsort(nonzero_vals)
                            color_vals = np.zeros_like(nonzero_vals)
                            color_vals[sorted_indices] = np.linspace(0, 1, len(nonzero_vals))
                        elif val_range < 0.5:  # Somewhat concentrated
                            logger.debug("Values are concentrated, using power transform")
                            color_vals = np.power(nonzero_vals, 0.3)  # More aggressive than sqrt
                        else:
                            logger.debug("Values are well distributed, using sqrt transform")
                            color_vals = np.power(nonzero_vals, 0.5)  # Square root

                        logger.debug(
                            f"After color transform: min={np.min(color_vals):.3f}, max={np.max(color_vals):.3f}"
                        )

                        # Simple red-to-blue mapping with high brightness and strong contrast
                        # Red component: bright for low values, dim for high values
                        r_vals = np.where(
                            color_vals < 0.5,
                            0.3 + 0.7 * (1.0 - 2 * color_vals),  # Range from 1.0 (lowest) to 0.3 (middle)
                            0.1,  # Very low red for high values
                        )

                        # Blue component: dim for low values, bright for high values
                        b_vals = np.where(
                            color_vals > 0.5,
                            0.3 + 0.7 * (2 * color_vals - 1.0),  # Range from 0.3 (middle) to 1.0 (highest)
                            0.1,  # Very low blue for low values
                        )

                        # Green component: moderate values for better overall brightness
                        # Provide some green to make the image brighter overall
                        g_vals = 0.2 + 0.4 * (1.0 - np.abs(2 * color_vals - 1.0))  # Peak at middle values

                        # Debug color component ranges before conversion
                        logger.debug("Color components before conversion:")
                        logger.debug(f"  Red: min={np.min(r_vals):.3f}, max={np.max(r_vals):.3f}")
                        logger.debug(f"  Green: min={np.min(g_vals):.3f}, max={np.max(g_vals):.3f}")
                        logger.debug(f"  Blue: min={np.min(b_vals):.3f}, max={np.max(b_vals):.3f}")

                        # Convert to 0-255 and assign to RGB image
                        r_uint8 = np.clip(r_vals * 255, 0, 255).astype(np.uint8)
                        g_uint8 = np.clip(g_vals * 255, 0, 255).astype(np.uint8)
                        b_uint8 = np.clip(b_vals * 255, 0, 255).astype(np.uint8)

                        rgb_image[valid_mask, 0] = r_uint8  # Red
                        rgb_image[valid_mask, 1] = g_uint8  # Green
                        rgb_image[valid_mask, 2] = b_uint8  # Blue

                        # Debug final uint8 ranges
                        logger.debug("Final uint8 ranges:")
                        logger.debug(f"  Red: min={np.min(r_uint8)}, max={np.max(r_uint8)}")
                        logger.debug(f"  Green: min={np.min(g_uint8)}, max={np.max(g_uint8)}")
                        logger.debug(f"  Blue: min={np.min(b_uint8)}, max={np.max(b_uint8)}")

                        # Debug: Check color distribution
                        unique_colors = np.unique(rgb_image.reshape(-1, 3), axis=0)
                        logger.debug(f"Generated {len(unique_colors)} unique colors")
                        logger.debug(f"Sample colors: {unique_colors[:5]}")
                        black_pixels = np.sum(np.all(rgb_image == [0, 0, 0], axis=2))
                        colored_pixels = matrix.size - black_pixels
                        logger.debug(
                            f"Black pixels (zeros): {black_pixels}, Colored pixels (non-zeros): {colored_pixels}"
                        )

                        # Verify our mapping worked
                        if colored_pixels != total_nonzero:
                            logger.warning(f"Mismatch: Expected {total_nonzero} colored pixels, got {colored_pixels}")

                        # Additional debug for intensity range
                        if colored_pixels > 0:
                            non_black_mask = ~np.all(rgb_image == [0, 0, 0], axis=2)
                            min_intensity = np.min(rgb_image[non_black_mask])
                            max_intensity = np.max(rgb_image[non_black_mask])
                            logger.debug(f"Non-zero pixel intensity range: {min_intensity} to {max_intensity}")
                            expected_min = int(intensity_floor * 255)
                            logger.debug(f"Expected minimum intensity: {expected_min}")

                        # else: pixels remain black (0,0,0) for true zeros

                        logger.debug(
                            f"RGB image range: R[{np.min(rgb_image[:,:,0])}-{np.max(rgb_image[:,:,0])}], "
                            f"G[{np.min(rgb_image[:,:,1])}-{np.max(rgb_image[:,:,1])}], "
                            f"B[{np.min(rgb_image[:,:,2])}-{np.max(rgb_image[:,:,2])}]"
                        )
                    else:
                        # Flat matrix - use single color
                        rgb_image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)
                        rgb_image[valid_mask] = [128, 0, 0]  # Dark red
                else:
                    # All zeros
                    rgb_image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)
            else:
                # All zeros
                rgb_image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)

            # Create PIL image in RGB mode
            img = Image.fromarray(rgb_image, mode="RGB")
            logger.debug(f"PIL RGB image created: size={img.size}, mode={img.mode}")

            # Resize if matrix is too large for practical display
            original_size = img.size
            if img.size[0] > 1000 or img.size[1] > 1000:
                # Calculate new size maintaining aspect ratio
                max_size = 1000
                ratio = min(max_size / img.size[0], max_size / img.size[1])
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                # Use newer PIL syntax with fallback
                try:
                    img = img.resize(new_size, Image.LANCZOS)
                except AttributeError:
                    # Fallback for older PIL versions
                    img = img.resize(new_size, Image.ANTIALIAS)
                logger.debug(f"Resized from {original_size} to {img.size}")

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer_bytes = buffer.getvalue()
            logger.debug(f"PNG buffer size: {len(buffer_bytes)} bytes")

            img_str = base64.b64encode(buffer_bytes).decode()
            logger.debug(f"Base64 string length: {len(img_str)}")

            data_url = f"data:image/png;base64,{img_str}"
            logger.debug(f"Data URL length: {len(data_url)}")

            return data_url

        except Exception as e:
            logger.warning(f"Failed to convert matrix to image: {e}")
            return ""

    def _find_image_pairs(self) -> List[Dict[str, str]]:
        """Find source/optimized image pairs."""
        pairs = []

        if not self.source_images_dir.exists():
            logger.warning(f"Source images directory does not exist: {self.source_images_dir}")
            return pairs

        # Find all source images
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        source_images = []

        for file_path in self.source_images_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                source_images.append(file_path)

        # Sort for consistent ordering
        source_images.sort()

        # For each source image, check if optimized version exists
        for source_path in source_images:
            source_name = source_path.name

            # Generate expected optimized filename
            stem = source_path.stem
            suffix = source_path.suffix
            optimized_name = f"{stem}_optimized{suffix}"
            optimized_path = self.optimized_images_dir / optimized_name

            # Check if we can generate optimized images (need required data and modules)
            can_generate = (
                FRAME_OPTIMIZER_AVAILABLE
                and DIAGONAL_ATA_AVAILABLE
                and self.mixed_tensor is not None
                and self.ata_inverse_data is not None
                and (
                    self.dia_matrix_data is not None
                    or self.dense_ata_data is not None
                    or self.symmetric_dia_data is not None
                )
            )

            # Create pair info
            pair_info = {
                "source_name": source_name,
                "source_path": f"/api/image/{source_name}",
                "optimized_name": optimized_name,
                "optimized_path": f"/api/image/{optimized_name}",
                "has_optimized": optimized_path.exists(),
                "can_generate": can_generate,
                "source_size": source_path.stat().st_size if source_path.exists() else 0,
                "optimized_size": optimized_path.stat().st_size if optimized_path.exists() else 0,
            }

            pairs.append(pair_info)

        logger.info(
            f"Found {len(pairs)} image pairs ({sum(1 for p in pairs if p['has_optimized'])} with optimized versions)"
        )
        return pairs

    def run(self, host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
        """Run the web server."""
        if not self.load_patterns():
            logger.error("Failed to load patterns")
            return

        logger.info(f"Starting diffusion pattern visualizer at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


# HTML Template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diffusion Pattern Visualizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 8px;
        }

        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        select, input, button {
            padding: 8px 12px;
            border: 1px solid #444;
            border-radius: 4px;
            background-color: #333;
            color: #fff;
        }

        button {
            background-color: #0066cc;
            cursor: pointer;
        }

        button:hover {
            background-color: #0052a3;
        }

        .metadata {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .metadata h3 {
            margin-top: 0;
            color: #4CAF50;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .pattern-card {
            background-color: #2a2a2a;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        .pattern-card:hover {
            background-color: #3a3a3a;
        }

        .pattern-card img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #444;
        }

        .pattern-info {
            margin-top: 8px;
            font-size: 12px;
            color: #ccc;
        }

        .pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin: 20px 0;
        }

        .pagination button {
            padding: 8px 16px;
        }

        .pagination button:disabled {
            background-color: #555;
            cursor: not-allowed;
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0, 0, 0, 0.8);
        }

        .modal-content {
            background-color: #2a2a2a;
            margin: 2% auto;
            padding: 20px;
            border-radius: 8px;
            width: 90%;
            max-width: 1200px;
            max-height: 90vh;
            overflow-y: auto;
        }

        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }

        .close:hover {
            color: #fff;
        }

        .pattern-detail {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .channel-images {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }

        .channel-image {
            text-align: center;
        }

        .channel-image img {
            width: 100%;
            max-width: 300px;
            border: 1px solid #444;
            border-radius: 4px;
        }

        .statistics {
            background-color: #333;
            padding: 15px;
            border-radius: 8px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        .loading {
            text-align: center;
            padding: 20px;
            color: #888;
        }

        .error {
            color: #ff6b6b;
            text-align: center;
            padding: 20px;
            background-color: #2a1a1a;
            border-radius: 8px;
            border: 1px solid #ff6b6b;
        }

        .format-toggle {
            background-color: #333;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .format-button {
            background-color: #444;
            color: #fff;
            border: none;
            padding: 8px 16px;
            margin-right: 10px;
            cursor: pointer;
            border-radius: 4px;
        }

        .format-button.active {
            background-color: #0066cc;
        }

        .ata-section {
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .ata-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #333;
            border-radius: 8px;
        }

        .ata-content {
            text-align: center;
        }

        .ata-matrix-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .ata-matrix-image {
            max-width: 100%;
            border: 1px solid #444;
            border-radius: 4px;
        }

        .ata-stats {
            background-color: #333;
            padding: 15px;
            border-radius: 8px;
        }

        .ata-rgb-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }

        .ata-channel-container {
            background-color: #333;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        .ata-channel-container h3 {
            margin: 0 0 15px 0;
            color: #4CAF50;
        }

        .ata-matrix-display img {
            max-width: 100%;
            border: 1px solid #444;
            border-radius: 4px;
        }

        @media (max-width: 1200px) {
            .ata-rgb-grid {
                grid-template-columns: 1fr;
            }
        }

        .ata-stats h3 {
            margin-top: 0;
            color: #4CAF50;
        }

        .images-section {
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .images-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #333;
            border-radius: 8px;
        }

        .images-controls input {
            flex: 1;
            max-width: 300px;
        }

        .images-content {
            text-align: center;
        }

        .image-pairs-grid {
            display: grid;
            gap: 20px;
        }

        .image-pair {
            background-color: #333;
            padding: 15px;
            border-radius: 8px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            align-items: start;
        }

        .image-pair.no-optimized {
            grid-template-columns: 1fr;
        }

        .image-container {
            text-align: center;
        }

        .image-container h4 {
            margin: 0 0 10px 0;
            color: #4CAF50;
        }

        .image-container img {
            max-width: 100%;
            max-height: 400px;
            border: 1px solid #444;
            border-radius: 4px;
        }

        .image-info {
            margin-top: 10px;
            font-size: 12px;
            color: #ccc;
        }

        .no-optimized-notice {
            text-align: center;
            color: #ff9800;
            font-style: italic;
            padding: 20px;
            background-color: #2a2a1a;
            border-radius: 4px;
        }

        .generate-btn {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 10px 0;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }

        .generate-btn:hover:not(:disabled) {
            background-color: #45a049;
        }

        .generate-btn:disabled {
            background-color: #666;
            cursor: not-allowed;
        }

        .regenerate-all-btn {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }

        .regenerate-all-btn:hover:not(:disabled) {
            background-color: #1976D2;
        }

        .regenerate-all-btn:disabled {
            background-color: #666;
            cursor: not-allowed;
        }

        .generation-status {
            margin-top: 10px;
        }

        .generation-status .loading {
            color: #2196F3;
            padding: 10px;
        }

        .generation-status .success {
            color: #4CAF50;
            padding: 10px;
            font-weight: bold;
        }

        .generation-status .error {
            color: #ff6b6b;
            padding: 10px;
            font-weight: bold;
            background-color: #2a1a1a;
            border-radius: 4px;
        }

        @media (max-width: 1200px) {
            .image-pair {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .controls {
                flex-direction: column;
                gap: 15px;
            }

            .grid {
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            }

            .pattern-detail {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌈 Diffusion Pattern Visualizer</h1>
            <p>Interactive viewer for LED diffusion patterns and source/optimized image comparisons</p>
        </div>

        <div id="metadata" class="metadata"></div>

        <div class="format-toggle">
            <button id="mixed-btn" class="format-button active" onclick="switchFormat('mixed')">LED Patterns</button>
            <button id="ata-btn" class="format-button" onclick="switchFormat('ata')">ATA Matrix</button>
            <button id="ata-inv-btn" class="format-button" onclick="switchFormat('ata_inverse')">ATA Inverse</button>
            <button id="images-btn" class="format-button" onclick="switchFormat('images')">Image Comparison</button>
        </div>

        <div class="controls">
            <div class="control-group">
                <label for="channel">Channel:</label>
                <select id="channel">
                    <option value="all">All (RGB)</option>
                    <option value="red">Red</option>
                    <option value="green">Green</option>
                    <option value="blue">Blue</option>
                </select>
            </div>

            <div class="control-group">
                <label for="order">Order:</label>
                <select id="order">
                    <option value="numerical">Numerical (LED ID)</option>
                    <option value="storage">Storage (Matrix Column)</option>
                </select>
            </div>

            <div class="control-group">
                <label for="per-page">Per page:</label>
                <select id="per-page">
                    <option value="25">25</option>
                    <option value="50" selected>50</option>
                    <option value="100">100</option>
                </select>
            </div>

            <button onclick="refresh()">Refresh</button>
        </div>

        <div class="pagination" id="pagination-top"></div>
        <div id="patterns" class="grid"></div>
        <div class="pagination" id="pagination-bottom"></div>

        <!-- ATA Matrix Visualization -->
        <div id="ata-section" class="ata-section" style="display: none;">
            <h2>A^T A Matrix Visualization (Log Magnitude)</h2>
            <div id="ata-content" class="ata-rgb-grid">
                <div class="ata-channel-container">
                    <h3>Red Channel</h3>
                    <div id="ata-red" class="ata-matrix-display"></div>
                </div>
                <div class="ata-channel-container">
                    <h3>Green Channel</h3>
                    <div id="ata-green" class="ata-matrix-display"></div>
                </div>
                <div class="ata-channel-container">
                    <h3>Blue Channel</h3>
                    <div id="ata-blue" class="ata-matrix-display"></div>
                </div>
            </div>
        </div>

        <!-- ATA Inverse Matrix Visualization -->
        <div id="ata-inverse-section" class="ata-section" style="display: none;">
            <h2>A^T A Inverse Matrix Visualization (Log Magnitude)</h2>
            <div id="ata-inverse-content" class="ata-rgb-grid">
                <div class="ata-channel-container">
                    <h3>Red Channel</h3>
                    <div id="ata-inv-red" class="ata-matrix-display"></div>
                </div>
                <div class="ata-channel-container">
                    <h3>Green Channel</h3>
                    <div id="ata-inv-green" class="ata-matrix-display"></div>
                </div>
                <div class="ata-channel-container">
                    <h3>Blue Channel</h3>
                    <div id="ata-inv-blue" class="ata-matrix-display"></div>
                </div>
            </div>
        </div>

        <!-- Image Comparison Section -->
        <div id="images-section" class="images-section" style="display: none;">
            <h2>Source vs Optimized Image Comparison</h2>
            <div class="images-controls">
                <button onclick="loadImagePairs()">Refresh Image List</button>
                <button onclick="regenerateAllPatterns()" class="regenerate-all-btn">Regenerate All Patterns</button>
                <label for="image-filter">Filter:</label>
                <input type="text" id="image-filter" placeholder="Filter by filename..." onInput="filterImages()">
            </div>
            <div id="images-content" class="images-content">
                <div class="loading">Click Refresh to load image pairs</div>
            </div>
        </div>
    </div>

    <!-- Modal for pattern details -->
    <div id="pattern-modal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <div id="pattern-detail-content"></div>
        </div>
    </div>

    <script>
        let currentPage = 0;
        let currentChannel = 'all';
        let currentOrder = 'numerical';
        let currentFormat = 'csc';
        let currentATAChannel = 0;
        let currentATAInvChannel = 0;
        let totalPages = 0;
        let perPage = 50;
        let metadata = null;

        // Initialize
        window.onload = function() {
            loadMetadata();
            loadPatterns();

            // Setup modal
            const modal = document.getElementById('pattern-modal');
            const closeBtn = document.getElementsByClassName('close')[0];

            closeBtn.onclick = function() {
                modal.style.display = 'none';
            }

            window.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = 'none';
                }
            }

            // Setup controls
            document.getElementById('channel').addEventListener('change', function() {
                currentChannel = this.value;
                currentPage = 0;
                loadPatterns();
            });

            document.getElementById('order').addEventListener('change', function() {
                currentOrder = this.value;
                currentPage = 0;
                loadPatterns();
            });

            document.getElementById('per-page').addEventListener('change', function() {
                perPage = parseInt(this.value);
                currentPage = 0;
                loadPatterns();
            });
        };

        function switchFormat(format) {
            currentFormat = format;
            currentPage = 0;

            // Update button styles
            document.getElementById('mixed-btn').classList.toggle('active', format === 'mixed');
            document.getElementById('ata-btn').classList.toggle('active', format === 'ata');
            document.getElementById('ata-inv-btn').classList.toggle('active', format === 'ata_inverse');
            document.getElementById('images-btn').classList.toggle('active', format === 'images');

            // Show/hide appropriate sections
            document.getElementById('patterns').style.display =
                format === 'mixed' ? 'grid' : 'none';
            document.getElementById('pagination-top').style.display =
                format === 'mixed' ? 'flex' : 'none';
            document.getElementById('pagination-bottom').style.display =
                format === 'mixed' ? 'flex' : 'none';
            document.querySelector('.controls').style.display =
                format === 'mixed' ? 'flex' : 'none';
            document.getElementById('ata-section').style.display =
                format === 'ata' ? 'block' : 'none';
            document.getElementById('ata-inverse-section').style.display =
                format === 'ata_inverse' ? 'block' : 'none';
            document.getElementById('images-section').style.display =
                format === 'images' ? 'block' : 'none';

            if (format === 'mixed') {
                loadPatterns();
            } else if (format === 'ata') {
                loadAllATAMatrices();
            } else if (format === 'ata_inverse') {
                loadAllATAInverseMatrices();
            }
        }

        function loadMetadata() {
            fetch('/api/metadata')
                .then(response => response.json())
                .then(data => {
                    metadata = data;
                    displayMetadata(data);

                    // Update order control visibility
                    const orderSelect = document.getElementById('order');
                    orderSelect.style.display = data.supports_storage_order ? 'block' : 'none';
                    if (!data.supports_storage_order) {
                        orderSelect.parentElement.style.display = 'none';
                    }

                    // Update ATA button visibility based on available data
                    const ataBtn = document.getElementById('ata-btn');
                    const ataInvBtn = document.getElementById('ata-inv-btn');
                    ataBtn.style.display = (data.dia_matrix_info || data.dense_ata_info || data.symmetric_dia_info) ? 'inline-block' : 'none';
                    ataInvBtn.style.display = data.ata_inverse_info ? 'inline-block' : 'none';

                })
                .catch(error => {
                    console.error('Error loading metadata:', error);
                    document.getElementById('metadata').innerHTML =
                        '<div class="error">Failed to load metadata: ' + error + '</div>';
                });
        }

        function displayMetadata(data) {
            const metadataDiv = document.getElementById('metadata');

            let html = '<h3>Pattern Information</h3>';
            html += '<div class="stats-grid">';
            html += `<div><strong>LED Count:</strong> ${data.led_count}</div>`;
            html += `<div><strong>Dimensions:</strong> ${data.frame_width} × ${data.frame_height}</div>`;
            html += `<div><strong>Channels:</strong> ${data.channels}</div>`;
            const format_text = 'Mixed Tensor';
            html += `<div><strong>Format:</strong> ${format_text}</div>`;

            if (data.mixed_tensor_info) {
                const info = data.mixed_tensor_info;
                const tensor_dims = `${info.channels} × ${info.batch_size} × ${info.height} × ${info.width}`;
                html += `<div><strong>Mixed Tensor:</strong> ${tensor_dims}</div>`;
                html += `<div><strong>Block Size:</strong> ${info.block_size} × ${info.block_size}</div>`;
                html += `<div><strong>Mixed Memory:</strong> ${info.memory_mb.toFixed(1)} MB</div>`;
                html += `<div><strong>Total Blocks:</strong> ${info.total_blocks.toLocaleString()}</div>`;
                if (info.dtype) {
                    html += `<div><strong>Mixed Tensor Type:</strong> ${info.dtype}</div>`;
                }
                if (info.sparse_values_dtype) {
                    html += `<div><strong>Sparse Values Type:</strong> ${info.sparse_values_dtype}</div>`;
                }
            }

            // Add ATA matrix information based on format
            if (data.ata_matrix_format) {
                html += `<div><strong>ATA Format:</strong> ${data.ata_matrix_format.toUpperCase()}</div>`;
            }

            if (data.dia_matrix_info) {
                const info = data.dia_matrix_info;
                html += `<div><strong>DIA Matrix:</strong> ${info.led_count} LEDs</div>`;
                html += `<div><strong>DIA Bandwidth:</strong> ${info.bandwidth}</div>`;
                html += `<div><strong>DIA Diagonals:</strong> ${info.k_diagonals}</div>`;
                html += `<div><strong>DIA Memory:</strong> ${info.memory_mb.toFixed(1)} MB</div>`;
                if (info.dtype) {
                    html += `<div><strong>DIA Data Type:</strong> ${info.dtype}</div>`;
                }
                if (info.storage_dtype) {
                    html += `<div><strong>DIA Storage Type:</strong> ${info.storage_dtype}</div>`;
                }
                if (info.output_dtype) {
                    html += `<div><strong>DIA Output Type:</strong> ${info.output_dtype}</div>`;
                }
            }

            if (data.dense_ata_info) {
                const info = data.dense_ata_info;
                html += `<div><strong>Dense Matrix:</strong> ${info.led_count} LEDs</div>`;
                html += `<div><strong>Dense Shape:</strong> ${info.matrix_shape.join(' × ')}</div>`;
                html += `<div><strong>Dense Memory:</strong> ${info.memory_mb.toFixed(1)} MB</div>`;
                if (info.storage_dtype) {
                    html += `<div><strong>Dense Storage Type:</strong> ${info.storage_dtype}</div>`;
                }
                if (info.output_dtype) {
                    html += `<div><strong>Dense Output Type:</strong> ${info.output_dtype}</div>`;
                }
            }

            if (data.symmetric_dia_info) {
                const info = data.symmetric_dia_info;
                html += `<div><strong>Symmetric DIA Matrix:</strong> ${info.led_count} LEDs</div>`;
                html += `<div><strong>Symmetric DIA Bandwidth:</strong> ${info.bandwidth}</div>`;
                html += `<div><strong>Symmetric DIA Upper Diagonals:</strong> ${info.k_upper}</div>`;
                html += `<div><strong>Symmetric DIA Memory:</strong> ${info.memory_mb.toFixed(1)} MB</div>`;
                html += `<div><strong>Symmetric DIA Sparsity:</strong> ${info.sparsity.toFixed(2)}%</div>`;
                html += `<div><strong>Symmetric DIA Non-zeros:</strong> ${info.nnz.toLocaleString()}</div>`;
                if (info.memory_reduction_percent) {
                    html += `<div><strong>Memory Reduction:</strong> ${info.memory_reduction_percent.toFixed(1)}%</div>`;
                }
                if (info.dtype) {
                    html += `<div><strong>Symmetric DIA Data Type:</strong> ${info.dtype}</div>`;
                }
                if (info.output_dtype) {
                    html += `<div><strong>Symmetric DIA Output Type:</strong> ${info.output_dtype}</div>`;
                }
            }

            if (data.ata_inverse_info) {
                const info = data.ata_inverse_info;
                html += `<div><strong>ATA Inverse:</strong> ${info.shape[0]} × ${info.shape[1]} × ${info.shape[2]}</div>`;
                html += `<div><strong>ATA Inv Memory:</strong> ${info.memory_mb.toFixed(1)} MB</div>`;
                html += `<div><strong>ATA Inv Type:</strong> ${info.dtype}</div>`;
            }

            html += '</div>';

            metadataDiv.innerHTML = html;
        }

        function loadPatterns() {
            const patternsDiv = document.getElementById('patterns');
            patternsDiv.innerHTML = '<div class="loading">Loading patterns...</div>';

            const params = new URLSearchParams({
                page: currentPage,
                per_page: perPage,
                channel: currentChannel,
                order: currentOrder,
                format: currentFormat
            });

            fetch(`/api/patterns?${params}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    displayPatterns(data);
                    setupPagination(data);
                })
                .catch(error => {
                    console.error('Error loading patterns:', error);
                    patternsDiv.innerHTML =
                        '<div class="error">Failed to load patterns: ' + error + '</div>';
                });
        }

        function displayPatterns(data) {
            const patternsDiv = document.getElementById('patterns');

            if (data.patterns.length === 0) {
                patternsDiv.innerHTML = '<div class="loading">No patterns to display</div>';
                return;
            }

            let html = '';
            data.patterns.forEach(pattern => {
                html += `
                    <div class="pattern-card" onclick="showPatternDetail(${pattern.led_id})">
                        <img src="${pattern.thumbnail}" alt="COL ${pattern.spatial_idx} LED ${pattern.physical_led_id}">
                        <div class="pattern-info">
                            <div><strong>COL ${pattern.spatial_idx} LED ${pattern.physical_led_id}</strong></div>
                            <div>Max: ${pattern.max_intensity.toFixed(3)}</div>
                            <div>CM: (${pattern.center_of_mass[0].toFixed(1)},
                                     ${pattern.center_of_mass[1].toFixed(1)})</div>
                        </div>
                    </div>
                `;
            });

            patternsDiv.innerHTML = html;
        }

        function setupPagination(data) {
            totalPages = data.total_pages;
            currentPage = data.page;

            const paginationHTML = `
                <button onclick="goToPage(0)" ${currentPage === 0 ? 'disabled' : ''}>First</button>
                <button onclick="goToPage(${currentPage - 1})" ${currentPage === 0 ? 'disabled' : ''}>Previous</button>
                <span>Page ${currentPage + 1} of ${totalPages} (${data.total_leds} LEDs, ${data.format_label})</span>
                <button onclick="goToPage(${currentPage + 1})"
                        ${currentPage >= totalPages - 1 ? 'disabled' : ''}>Next</button>
                <button onclick="goToPage(${totalPages - 1})"
                        ${currentPage >= totalPages - 1 ? 'disabled' : ''}>Last</button>
            `;

            document.getElementById('pagination-top').innerHTML = paginationHTML;
            document.getElementById('pagination-bottom').innerHTML = paginationHTML;
        }

        function goToPage(page) {
            if (page >= 0 && page < totalPages) {
                currentPage = page;
                loadPatterns();
            }
        }

        function showPatternDetail(ledId) {
            const modal = document.getElementById('pattern-modal');
            const content = document.getElementById('pattern-detail-content');

            content.innerHTML = '<div class="loading">Loading pattern details...</div>';
            modal.style.display = 'block';

            fetch(`/api/pattern/${ledId}?order=${currentOrder}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    displayPatternDetail(data);
                })
                .catch(error => {
                    console.error('Error loading pattern detail:', error);
                    content.innerHTML =
                        '<div class="error">Failed to load pattern details: ' + error + '</div>';
                });
        }

        function displayPatternDetail(data) {
            const content = document.getElementById('pattern-detail-content');

            let html = `<h2>COL ${data.spatial_idx} LED ${data.physical_led_id} - Pattern Details</h2>`;

            // Format tabs
            html += '<div class="format-toggle">';
            Object.keys(data.formats).forEach(format => {
                const isActive = format === 'mixed' ? 'active' : '';
                html += `<button class="format-button ${isActive}"
                         onclick="showFormatTab('${format}')">${format.toUpperCase()}</button>`;
            });
            html += '</div>';

            // Format content
            Object.entries(data.formats).forEach(([format, formatData]) => {
                const displayStyle = format === 'mixed' ? 'block' : 'none';
                html += `<div id="format-${format}" style="display: ${displayStyle}">`;
                html += '<div class="pattern-detail">';

                // Left side - images
                html += '<div>';
                html += '<h3>Composite RGB</h3>';
                html += `<img src="${formatData.composite.image}"
                         style="width: 100%; max-width: 400px; border: 1px solid #444;">`;

                html += '<h3>Individual Channels</h3>';
                html += '<div class="channel-images">';
                Object.entries(formatData.channels).forEach(([channel, channelData]) => {
                    html += `
                        <div class="channel-image">
                            <h4>${channel.charAt(0).toUpperCase() + channel.slice(1)}</h4>
                            <img src="${channelData.image}" alt="${channel}">
                        </div>
                    `;
                });
                html += '</div></div>';

                // Right side - statistics
                html += '<div><h3>Statistics</h3>';
                html += '<div class="statistics">';
                html += '<h4>Composite</h4>';
                html += '<div class="stats-grid">';
                Object.entries(formatData.composite.statistics).forEach(([key, value]) => {
                    const displayValue = Array.isArray(value) ?
                        `(${value.map(v => v.toFixed(1)).join(', ')})` :
                        (typeof value === 'number' ? value.toFixed(4) : value);
                    html += `<div><strong>${key.replace('_', ' ')}:</strong> ${displayValue}</div>`;
                });
                html += '</div>';

                Object.entries(formatData.channels).forEach(([channel, channelData]) => {
                    html += `<h4>${channel.charAt(0).toUpperCase() + channel.slice(1)} Channel</h4>`;
                    html += '<div class="stats-grid">';
                    Object.entries(channelData.statistics).forEach(([key, value]) => {
                        const displayValue = Array.isArray(value) ?
                            `(${value.map(v => v.toFixed(1)).join(', ')})` :
                            (typeof value === 'number' ? value.toFixed(4) : value);
                        html += `<div><strong>${key.replace('_', ' ')}:</strong> ${displayValue}</div>`;
                    });
                    html += '</div>';
                });

                html += '</div></div></div></div>';
            });

            content.innerHTML = html;
        }

        function showFormatTab(format) {
            // Hide all format tabs
            document.querySelectorAll('[id^="format-"]').forEach(tab => {
                tab.style.display = 'none';
            });

            // Show selected tab
            document.getElementById(`format-${format}`).style.display = 'block';

            // Update button states
            document.querySelectorAll('.format-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
        }

        function refresh() {
            loadMetadata();
            loadPatterns();
        }

        let imagePairs = [];

        function loadImagePairs() {
            const content = document.getElementById('images-content');

            content.innerHTML = '<div class="loading">Loading image pairs...</div>';

            fetch('/api/image_pairs')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    imagePairs = data.pairs;
                    displayImagePairs(imagePairs);
                })
                .catch(error => {
                    console.error('Error loading image pairs:', error);
                    content.innerHTML = '<div class="error">Failed to load image pairs: ' + error + '</div>';
                });
        }

        function displayImagePairs(pairs) {
            const content = document.getElementById('images-content');

            if (pairs.length === 0) {
                content.innerHTML = '<div class="loading">No images found in images/source directory</div>';
                return;
            }

            let html = '<div class="image-pairs-grid">';

            pairs.forEach(pair => {
                const hasOptimized = pair.has_optimized;
                const pairClass = hasOptimized ? 'image-pair' : 'image-pair no-optimized';

                html += `<div class="${pairClass}">`;

                // Source image
                html += '<div class="image-container">';
                html += '<h4>Source Image</h4>';
                html += `<img src="${pair.source_path}" alt="${pair.source_name}" loading="lazy">`;
                html += '<div class="image-info">';
                html += `<div><strong>File:</strong> ${pair.source_name}</div>`;
                html += `<div><strong>Size:</strong> ${formatFileSize(pair.source_size)}</div>`;
                html += '</div></div>';

                // Optimized image (if exists)
                if (hasOptimized) {
                    html += '<div class="image-container">';
                    html += '<h4>Optimized LED Version</h4>';
                    html += `<img src="${pair.optimized_path}" alt="${pair.optimized_name}" loading="lazy">`;
                    html += '<div class="image-info">';
                    html += `<div><strong>File:</strong> ${pair.optimized_name}</div>`;
                    html += `<div><strong>Size:</strong> ${formatFileSize(pair.optimized_size)}</div>`;
                    html += '</div></div>';
                } else {
                    html += '<div class="no-optimized-notice">';
                    html += '<p>No optimized version available</p>';
                    if (pair.can_generate) {
                        html += `<button class="generate-btn" onclick="generateOptimizedImage('${pair.source_name}', this)">Generate Optimized Version</button>`;
                        html += '<div class="generation-status" style="display: none;"></div>';
                    } else {
                        html += '<p>Cannot generate - missing optimization data</p>';
                        html += '<p>Run: <code>python tools/batch_image_optimizer.py --pattern-file patterns.npz</code></p>';
                    }
                    html += '</div>';
                }

                html += '</div>';
            });

            html += '</div>';

            content.innerHTML = html;
        }

        function filterImages() {
            const filterText = document.getElementById('image-filter').value.toLowerCase();

            if (!imagePairs.length) {
                return;
            }

            const filteredPairs = imagePairs.filter(pair => {
                return pair.source_name.toLowerCase().includes(filterText);
            });

            displayImagePairs(filteredPairs);
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 B';

            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));

            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }

        function generateOptimizedImage(sourceName, buttonElement) {
            // Disable button and show loading state
            buttonElement.disabled = true;
            buttonElement.textContent = 'Generating...';

            const statusDiv = buttonElement.parentElement.querySelector('.generation-status');
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '<div class="loading">Running frame optimization...</div>';

            // Make POST request to generate optimized image
            fetch(`/api/generate_optimized/${encodeURIComponent(sourceName)}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }

                // Success - show success message
                statusDiv.innerHTML = '<div class="success">✓ Optimized image generated successfully!</div>';

                // Refresh the image pairs display after a short delay
                setTimeout(() => {
                    loadImagePairs();
                }, 1000);
            })
            .catch(error => {
                console.error('Failed to generate optimized image:', error);

                // Show error message
                statusDiv.innerHTML = `<div class="error">✗ Error: ${error.message}</div>`;

                // Re-enable button
                buttonElement.disabled = false;
                buttonElement.textContent = 'Generate Optimized Version';
            });
        }

        function regenerateAllPatterns() {
            const button = document.querySelector('.regenerate-all-btn');

            // Disable button and show loading state
            button.disabled = true;
            button.textContent = 'Regenerating...';

            // Show progress in images content area
            const contentDiv = document.getElementById('images-content');
            contentDiv.innerHTML = '<div class="loading">Regenerating all optimized patterns...</div>';

            // Make POST request to regenerate all patterns
            fetch('/api/regenerate_all_patterns', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }

                // Show success message with summary
                const summaryHtml = `
                    <div class="success" style="padding: 20px; background-color: #1b5e20; color: #4caf50; border-radius: 4px; margin-bottom: 20px;">
                        <h3>✅ Bulk Regeneration Complete!</h3>
                        <p>Total Images: ${data.total}</p>
                        <p>Successful: ${data.successful}</p>
                        <p>Failed: ${data.failed}</p>
                    </div>
                `;

                contentDiv.innerHTML = summaryHtml;

                // Re-enable button
                button.disabled = false;
                button.textContent = 'Regenerate All Patterns';

                // Refresh the image pairs display after a short delay
                setTimeout(() => {
                    loadImagePairs();
                }, 2000);
            })
            .catch(error => {
                console.error('Failed to regenerate all patterns:', error);

                // Show error message
                contentDiv.innerHTML = `<div class="error" style="padding: 20px; background-color: #5e1b1b; color: #f44336; border-radius: 4px;">✗ Error: ${error.message}</div>`;

                // Re-enable button
                button.disabled = false;
                button.textContent = 'Regenerate All Patterns';
            });
        }

        function loadAllATAMatrices() {
            const channels = ['red', 'green', 'blue'];

            channels.forEach((channelName, channelIdx) => {
                const container = document.getElementById(`ata-${channelName}`);
                container.innerHTML = '<div class="loading">Loading...</div>';

                fetch(`/api/ata_matrix/${channelIdx}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            throw new Error(data.error);
                        }
                        container.innerHTML = `<img src="${data.matrix_image}" alt="${channelName.toUpperCase()} ATA Matrix">`;
                    })
                    .catch(error => {
                        console.error(`Error loading ${channelName} ATA matrix:`, error);
                        container.innerHTML = `<div class="error">Failed to load: ${error}</div>`;
                    });
            });
        }

        function loadAllATAInverseMatrices() {
            const channels = ['red', 'green', 'blue'];

            channels.forEach((channelName, channelIdx) => {
                const container = document.getElementById(`ata-inv-${channelName}`);
                container.innerHTML = '<div class="loading">Loading...</div>';

                fetch(`/api/ata_inverse/${channelIdx}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            throw new Error(data.error);
                        }
                        container.innerHTML = `<img src="${data.matrix_image}" alt="${channelName.toUpperCase()} ATA Inverse Matrix">`;
                    })
                    .catch(error => {
                        console.error(`Error loading ${channelName} ATA inverse matrix:`, error);
                        container.innerHTML = `<div class="error">Failed to load: ${error}</div>`;
                    });
            });
        }


    </script>
</body>
</html>
"""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize LED diffusion patterns")
    parser.add_argument(
        "--patterns",
        type=str,
        help="Path to patterns file (.npz) in new nested format",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Set up logging - force debug temporarily
    logging.basicConfig(
        level=logging.DEBUG,  # Force debug to see normalization steps
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create visualizer
    visualizer = DiffusionPatternVisualizer(patterns_file=args.patterns)

    # Run the server
    visualizer.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
