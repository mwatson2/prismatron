#!/usr/bin/env python3
"""
Diffusion Pattern Visualization Tool.

This tool creates a web interface to visualize LED diffusion patterns using
the new nested storage format with LEDDiffusionCSCMatrix and SingleBlockMixedSparseTensor.

Features:
1. Grid view of all LED patterns
2. Individual LED/channel navigation
3. Support for both CSC matrix and mixed tensor formats
4. Interactive controls and pattern comparison
5. Comprehensive pattern statistics

Supported format:
- New nested format: NPZ with diffusion_matrix (LEDDiffusionCSCMatrix),
  mixed_tensor (SingleBlockMixedSparseTensor), and metadata

Usage:
    python visualize_diffusion_patterns.py --patterns patterns.npz \\
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
from flask import Flask, jsonify, render_template_string, request, send_from_directory

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import constants directly to avoid relative import issues
FRAME_HEIGHT = 480
FRAME_WIDTH = 800
LED_COUNT = 2600

# Import specific modules directly to avoid __init__.py issues
sys.path.append(str(Path(__file__).parent.parent / "src" / "utils"))
from led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from single_block_sparse_tensor import SingleBlockMixedSparseTensor

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

        # Wrapper objects for the new format
        self.diffusion_matrix: Optional[LEDDiffusionCSCMatrix] = None
        self.mixed_tensor: Optional[SingleBlockMixedSparseTensor] = None

        # Cached dense patterns for visualization
        self.dense_patterns_csc: Optional[np.ndarray] = None
        self.dense_patterns_mixed: Optional[np.ndarray] = None

        # Metadata and LED info
        self.metadata: Dict = {}
        self.led_positions: Optional[np.ndarray] = None
        self.led_spatial_mapping: Optional[Dict] = None
        self.dense_ata_data: Optional[Dict] = None

        # Flask app
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)  # Reduce Flask logging

        # Setup routes
        self._setup_routes()

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
            if not all(key in data for key in ["diffusion_matrix", "mixed_tensor", "metadata"]):
                logger.error("File does not contain the new nested format")
                logger.error("Expected keys: diffusion_matrix, mixed_tensor, metadata")
                logger.error(f"Found keys: {list(data.keys())}")
                logger.error("Please regenerate patterns with the updated generate_synthetic_patterns.py")
                return False

            logger.info("Detected new nested format, loading wrapper objects...")
            return self._load_nested_format(data)

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def _load_nested_format(self, data: np.lib.npyio.NpzFile) -> bool:
        """Load patterns from new nested format."""
        try:
            # Load LEDDiffusionCSCMatrix
            logger.info("Loading LEDDiffusionCSCMatrix...")
            diffusion_matrix_dict = data["diffusion_matrix"].item()
            self.diffusion_matrix = LEDDiffusionCSCMatrix.from_dict(diffusion_matrix_dict)
            logger.info(f"Loaded diffusion matrix: {self.diffusion_matrix}")

            # Load SingleBlockMixedSparseTensor
            logger.info("Loading SingleBlockMixedSparseTensor...")
            mixed_tensor_dict = data["mixed_tensor"].item()
            self.mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
            logger.info(f"Loaded mixed tensor: {self.mixed_tensor}")

            # Load metadata
            self.metadata = data["metadata"].item() if "metadata" in data else {}

            # Load LED positions and spatial mapping
            self.led_positions = data.get("led_positions", None)
            self.led_spatial_mapping = data.get("led_spatial_mapping", None)
            if self.led_spatial_mapping is not None and hasattr(self.led_spatial_mapping, "item"):
                self.led_spatial_mapping = self.led_spatial_mapping.item()

            # Load dense A^T@A data if available
            self.dense_ata_data = data.get("dense_ata", None)
            if self.dense_ata_data is not None and hasattr(self.dense_ata_data, "item"):
                self.dense_ata_data = self.dense_ata_data.item()

            # Convert to dense patterns for visualization using new methods
            logger.info("Converting to dense patterns for visualization...")
            self.dense_patterns_csc = self.diffusion_matrix.to_dense_patterns()
            self.dense_patterns_mixed = self.mixed_tensor.to_dense_patterns()

            logger.info(f"CSC dense patterns shape: {self.dense_patterns_csc.shape}")
            logger.info(f"Mixed tensor dense patterns shape: {self.dense_patterns_mixed.shape}")

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
            """Get pattern metadata using wrapper classes."""
            if not self.diffusion_matrix or not self.mixed_tensor:
                return jsonify({"error": "No patterns loaded"}), 404

            # Get actual LED count from wrapper objects
            led_count_csc = self.diffusion_matrix.led_count
            led_count_mixed = self.mixed_tensor.batch_size

            # Use CSC matrix as authoritative source
            actual_led_count = led_count_csc

            metadata_response = {
                "metadata": self.metadata,
                "led_count": actual_led_count,
                "frame_width": self.diffusion_matrix.width,
                "frame_height": self.diffusion_matrix.height,
                "channels": self.diffusion_matrix.channels,
                "patterns_loaded": True,
                "has_both_formats": True,  # Always true for new format
                "supports_storage_order": self.led_spatial_mapping is not None,
            }

            # Add CSC matrix info using wrapper methods
            if self.diffusion_matrix:
                summary = self.diffusion_matrix.get_pattern_summary()
                memory_info = self.diffusion_matrix.memory_info()

                metadata_response["csc_info"] = {
                    "matrix_shape": summary["matrix_shape"],
                    "nnz": summary["nnz_total"],
                    "sparsity_percent": summary["sparsity_ratio"] * 100,
                    "memory_mb": memory_info["total_mb"],
                    "led_count": summary["led_count"],
                    "channels": summary["channels"],
                }

            # Add mixed tensor info using wrapper methods
            if self.mixed_tensor:
                block_summary = self.mixed_tensor.get_block_summary()
                memory_info = self.mixed_tensor.memory_info()

                metadata_response["mixed_tensor_info"] = {
                    "batch_size": block_summary["batch_size"],
                    "channels": block_summary["channels"],
                    "height": self.mixed_tensor.height,
                    "width": self.mixed_tensor.width,
                    "block_size": block_summary["block_size"],
                    "total_blocks": block_summary["total_blocks"],
                    "memory_mb": memory_info["total_mb"],
                }

            # Add dense A^T@A info if available
            if self.dense_ata_data:
                metadata_response["dense_ata_info"] = {
                    "shape": list(self.dense_ata_data["dense_ata_matrices"].shape),
                    "memory_mb": self.dense_ata_data["dense_ata_matrices"].nbytes / (1024 * 1024),
                    "computation_time": self.dense_ata_data["dense_ata_computation_time"],
                }

            return jsonify(metadata_response)

        @self.app.route("/api/patterns")
        def get_patterns():
            """Get pattern list with thumbnails using cached dense patterns."""
            if self.dense_patterns_csc is None or self.dense_patterns_mixed is None:
                return jsonify({"error": "No patterns loaded"}), 404

            try:
                page = int(request.args.get("page", 0))
                per_page = int(request.args.get("per_page", 50))
                channel = request.args.get("channel", "all")
                order = request.args.get("order", "numerical")  # "numerical" or "storage"
                format_type = request.args.get("format", "csc")  # "csc" or "mixed"

                # Use CSC matrix as authoritative for LED count
                actual_led_count = self.diffusion_matrix.led_count

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
                    reverse_mapping = {
                        matrix_idx: physical_id for physical_id, matrix_idx in self.led_spatial_mapping.items()
                    }
                    # Sort by physical ID for display, but keep track of spatial index for array access
                    display_info = [
                        {"spatial_idx": i, "display_id": reverse_mapping.get(i, i)} for i in range(actual_led_count)
                    ]
                    # Sort by display_id (physical LED ID) for numerical order
                    display_info.sort(key=lambda x: x["display_id"])

                # Get the LEDs for this page
                page_info = display_info[start_idx:end_idx]

                # Select pattern data based on format parameter
                if format_type == "mixed":
                    patterns_to_use = self.dense_patterns_mixed
                    format_label = "Mixed Tensor"
                else:
                    patterns_to_use = self.dense_patterns_csc
                    format_label = "CSC Matrix"

                for info in page_info:
                    spatial_idx = info["spatial_idx"]
                    display_id = info["display_id"]
                    if channel == "all":
                        # Create composite RGB image - patterns are (height, width, 3)
                        rgb_pattern = patterns_to_use[spatial_idx]  # Use spatial index for array access
                    else:
                        # Single channel - extract the specific channel
                        channel_idx = {"red": 0, "green": 1, "blue": 2}.get(channel, 0)
                        single_channel = patterns_to_use[spatial_idx, :, :, channel_idx]  # (height, width)
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
                            "led_id": display_id,  # For compatibility with existing click handlers
                            "spatial_idx": spatial_idx,  # Column index in matrix
                            "physical_led_id": physical_led_id,  # Original LED numbering
                            "thumbnail": thumbnail,
                            "max_intensity": float(np.max(rgb_pattern)),
                            "center_of_mass": self._calculate_center_of_mass(rgb_pattern).tolist(),
                        }
                    )

                return jsonify(
                    {
                        "patterns": patterns,
                        "page": page,
                        "per_page": per_page,
                        "total_leds": actual_led_count,
                        "total_pages": (actual_led_count + per_page - 1) // per_page,
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
            if self.dense_patterns_csc is None or self.dense_patterns_mixed is None:
                return jsonify({"error": "No patterns loaded"}), 404

            # Get ordering mode to interpret led_id correctly
            order = request.args.get("order", "numerical")

            actual_led_count = self.diffusion_matrix.led_count

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
                    "led_id": display_id,
                    "spatial_idx": spatial_idx,
                    "physical_led_id": physical_led_id,
                    "channels": {},
                    "formats": {},
                }

                channel_names = ["red", "green", "blue"]

                # Process both CSC and mixed tensor formats
                for format_name, patterns_array in [
                    ("csc", self.dense_patterns_csc),
                    ("mixed", self.dense_patterns_mixed),
                ]:
                    format_data = {"channels": {}}

                    for ch_idx, ch_name in enumerate(channel_names):
                        pattern = patterns_array[spatial_idx, :, :, ch_idx]

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
                    rgb_pattern = patterns_array[spatial_idx]  # Already in (height, width, 3) format

                    format_data["composite"] = {
                        "image": self._create_full_image(rgb_pattern),
                        "statistics": {
                            "max_intensity": float(np.max(rgb_pattern)),
                            "center_of_mass": self._calculate_center_of_mass(rgb_pattern).tolist(),
                        },
                    }

                    pattern_data["formats"][format_name] = format_data

                return jsonify(pattern_data)

            except Exception as e:
                logger.error(f"Failed to get pattern detail: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return jsonify({"error": str(e)}), 500

    def _create_thumbnail(self, pattern: np.ndarray, size: Tuple[int, int]) -> str:
        """Create base64 thumbnail from pattern."""
        try:
            if not PIL_AVAILABLE:
                return ""

            # Normalize pattern to 0-255
            if len(pattern.shape) == 3:
                # RGB pattern
                normalized = np.clip(pattern * 255, 0, 255).astype(np.uint8)
            else:
                # Single channel - convert to RGB
                normalized = np.clip(pattern * 255, 0, 255).astype(np.uint8)
                normalized = np.stack([normalized] * 3, axis=-1)

            # Create PIL image
            img = Image.fromarray(normalized, mode="RGB")

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

            # Normalize to 0-255
            normalized = np.clip(pattern * 255, 0, 255).astype(np.uint8)

            # Create PIL image
            img = Image.fromarray(normalized, mode="RGB")

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

            # Normalize to 0-255
            normalized = np.clip(pattern * 255, 0, 255).astype(np.uint8)

            # Create PIL image
            img = Image.fromarray(normalized, mode="L")

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
                        self.diffusion_matrix.width // 2,
                        self.diffusion_matrix.height // 2,
                    ]
                )

            y_indices, x_indices = np.indices(intensity.shape)
            x_cm = np.sum(x_indices * intensity) / total_mass
            y_cm = np.sum(y_indices * intensity) / total_mass

            return np.array([x_cm, y_cm])

        except Exception:
            return np.array([self.diffusion_matrix.width // 2, self.diffusion_matrix.height // 2])

    def run(self, host: str = "127.0.0.1", port: int = 8080, debug: bool = False):
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
            <p>Interactive viewer for LED diffusion patterns (CSC Matrix & Mixed Tensor formats)</p>
        </div>

        <div id="metadata" class="metadata"></div>

        <div class="format-toggle">
            <button id="csc-btn" class="format-button active" onclick="switchFormat('csc')">CSC Matrix</button>
            <button id="mixed-btn" class="format-button" onclick="switchFormat('mixed')">Mixed Tensor</button>
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
            document.getElementById('csc-btn').classList.toggle('active', format === 'csc');
            document.getElementById('mixed-btn').classList.toggle('active', format === 'mixed');

            loadPatterns();
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
            const format_text = data.has_both_formats ? 'CSC Matrix + Mixed Tensor' : 'Single Format';
            html += `<div><strong>Formats:</strong> ${format_text}</div>`;

            if (data.csc_info) {
                const rows = data.csc_info.matrix_shape[0];
                const cols = data.csc_info.matrix_shape[1];
                html += `<div><strong>CSC Matrix:</strong> ${rows} × ${cols}</div>`;
                html += `<div><strong>CSC Memory:</strong> ${data.csc_info.memory_mb.toFixed(1)} MB</div>`;
                html += `<div><strong>CSC Sparsity:</strong> ${data.csc_info.sparsity_percent.toFixed(2)}%</div>`;
                html += `<div><strong>CSC NNZ:</strong> ${data.csc_info.nnz.toLocaleString()}</div>`;
            }

            if (data.mixed_tensor_info) {
                const info = data.mixed_tensor_info;
                const tensor_dims = `${info.batch_size} × ${info.channels} × ${info.height} × ${info.width}`;
                html += `<div><strong>Mixed Tensor:</strong> ${tensor_dims}</div>`;
                html += `<div><strong>Block Size:</strong> ${info.block_size} × ${info.block_size}</div>`;
                html += `<div><strong>Mixed Memory:</strong> ${info.memory_mb.toFixed(1)} MB</div>`;
                html += `<div><strong>Total Blocks:</strong> ${info.total_blocks.toLocaleString()}</div>`;
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
                const isActive = format === 'csc' ? 'active' : '';
                html += `<button class="format-button ${isActive}"
                         onclick="showFormatTab('${format}')">${format.toUpperCase()}</button>`;
            });
            html += '</div>';

            // Format content
            Object.entries(data.formats).forEach(([format, formatData]) => {
                const displayStyle = format === 'csc' ? 'block' : 'none';
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
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create visualizer
    visualizer = DiffusionPatternVisualizer(patterns_file=args.patterns)

    # Run the server
    visualizer.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
