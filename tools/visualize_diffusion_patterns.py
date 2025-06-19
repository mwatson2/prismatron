#!/usr/bin/env python3
"""
Diffusion Pattern Visualization Tool.

This tool creates a web interface to visualize diffusion patterns with:
1. Grid view of all LED patterns
2. Individual LED/channel navigation
3. Support for both captured and synthetic patterns
4. Interactive controls and filtering

Usage:
    python visualize_diffusion_patterns.py --patterns captured_patterns.npz \\
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

from const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT

logger = logging.getLogger(__name__)


class DiffusionPatternVisualizer:
    """Web-based diffusion pattern visualization tool with sparse matrix support."""

    def __init__(self, patterns_file: Optional[str] = None, use_synthetic: bool = True):
        """
        Initialize visualizer.

        Args:
            patterns_file: Path to captured patterns file (.npz)
            use_synthetic: Generate synthetic patterns if no file provided
        """
        self.patterns_file = patterns_file
        self.use_synthetic = use_synthetic

        # Pattern data (dense format for visualization)
        self.diffusion_patterns: Optional[np.ndarray] = None
        self.metadata: Dict = {}

        # Sparse matrix data
        self.is_sparse_format = False
        self.sparse_matrix: Optional[sp.csc_matrix] = None
        self.led_positions: Optional[np.ndarray] = None
        self.led_spatial_mapping: Optional[Dict] = None

        # Flask app
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)  # Reduce Flask logging

        # Setup routes
        self._setup_routes()

    def load_patterns(self) -> bool:
        """Load diffusion patterns from file (dense or sparse format) or generate synthetic ones."""
        try:
            if self.patterns_file and Path(self.patterns_file).exists():
                logger.info(f"Loading patterns from {self.patterns_file}")

                # Check if this is a sparse format file
                if self.patterns_file.endswith("_matrix.npz"):
                    return self._load_sparse_patterns()
                else:
                    # Try to auto-detect sparse format
                    base_path = self.patterns_file.replace(".npz", "")
                    matrix_path = f"{base_path}_matrix.npz"
                    mapping_path = f"{base_path}_mapping.npz"

                    if Path(matrix_path).exists() and Path(mapping_path).exists():
                        logger.info(
                            "Detected sparse format files, loading sparse matrix..."
                        )
                        return self._load_sparse_patterns_from_base(base_path)
                    else:
                        return self._load_captured_patterns()

            elif self.use_synthetic:
                logger.error("Synthetic pattern generation moved to separate tool.")
                logger.error(
                    "Generate patterns first with: python tools/generate_synthetic_patterns.py"
                )
                return False
            else:
                logger.error(
                    "No patterns file provided and synthetic patterns disabled"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            return False

    def _load_captured_patterns(self) -> bool:
        """Load captured diffusion patterns from .npz file (dense format)."""
        try:
            data = np.load(self.patterns_file, allow_pickle=True)

            # Try different possible key names for patterns
            pattern_keys = ["diffusion_patterns", "patterns"]
            patterns = None
            for key in pattern_keys:
                if key in data:
                    patterns = data[key]
                    logger.info(f"Found patterns with key: {key}")
                    break

            if patterns is None:
                logger.error(
                    f"No pattern data found. Available keys: {list(data.keys())}"
                )
                return False

            # Handle different pattern formats
            if patterns.ndim == 4:
                if patterns.shape[1] == 3:  # (led_count, 3, height, width) - CHW format
                    logger.info("Converting from CHW to HWC format")
                    self.diffusion_patterns = np.transpose(
                        patterns, (0, 2, 3, 1)
                    )  # -> (led_count, height, width, 3)
                elif (
                    patterns.shape[3] == 3
                ):  # (led_count, height, width, 3) - HWC format
                    self.diffusion_patterns = patterns
                else:
                    logger.error(f"Unsupported pattern shape: {patterns.shape}")
                    return False
            else:
                logger.error(f"Unsupported pattern dimensions: {patterns.ndim}")
                return False

            self.metadata = data["metadata"].item() if "metadata" in data else {}
            self.is_sparse_format = False

            logger.info(f"Loaded dense patterns: {self.diffusion_patterns.shape}")
            logger.info(f"Metadata: {self.metadata}")

            return True

        except Exception as e:
            logger.error(f"Failed to load captured patterns: {e}")
            return False

    def _load_sparse_patterns(self) -> bool:
        """Load sparse diffusion patterns from _matrix.npz file."""
        try:
            # Extract base path from matrix file
            base_path = self.patterns_file.replace("_matrix.npz", "")
            return self._load_sparse_patterns_from_base(base_path)

        except Exception as e:
            logger.error(f"Failed to load sparse patterns: {e}")
            return False

    def _load_sparse_patterns_from_base(self, base_path: str) -> bool:
        """Load sparse diffusion patterns from base path."""
        try:
            matrix_path = f"{base_path}_matrix.npz"
            mapping_path = f"{base_path}_mapping.npz"

            if not Path(matrix_path).exists():
                logger.error(f"Sparse matrix file not found: {matrix_path}")
                return False

            if not Path(mapping_path).exists():
                logger.error(f"Sparse mapping file not found: {mapping_path}")
                return False

            # Load sparse matrix
            logger.info(f"Loading sparse matrix from {matrix_path}")
            self.sparse_matrix = sp.load_npz(matrix_path)

            # Load spatial mapping and metadata
            logger.info(f"Loading spatial mapping from {mapping_path}")
            mapping_data = np.load(mapping_path, allow_pickle=True)
            self.led_spatial_mapping = mapping_data["led_spatial_mapping"].item()
            self.led_positions = mapping_data["led_positions"]
            self.metadata = (
                mapping_data["metadata"].item() if "metadata" in mapping_data else {}
            )

            # Convert sparse matrix to dense patterns for visualization
            logger.info("Converting sparse matrix to dense format for visualization...")
            self.diffusion_patterns = self._sparse_to_dense_patterns()
            self.is_sparse_format = True

            logger.info(f"Loaded sparse patterns: {self.diffusion_patterns.shape}")
            logger.info(f"Matrix shape: {self.sparse_matrix.shape}")
            logger.info(
                f"Matrix sparsity: {self.sparse_matrix.nnz / (self.sparse_matrix.shape[0] * self.sparse_matrix.shape[1]) * 100:.3f}%"
            )
            logger.info(f"Metadata: {self.metadata}")

            return True

        except Exception as e:
            logger.error(f"Failed to load sparse patterns from {base_path}: {e}")
            return False

    def _sparse_to_dense_patterns(self) -> np.ndarray:
        """
        Convert sparse CSC matrix back to dense diffusion patterns for visualization.

        Returns:
            Dense patterns array (led_count, height, width, 3)
        """
        try:
            led_count = self.sparse_matrix.shape[1]

            # Initialize dense patterns array
            patterns = np.zeros(
                (led_count, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
            )

            # Create reverse spatial mapping: matrix_index -> physical_led_id
            reverse_mapping = {
                matrix_idx: physical_id
                for physical_id, matrix_idx in self.led_spatial_mapping.items()
            }

            # Convert sparse matrix back to dense patterns
            logger.info("Converting sparse matrix columns to dense patterns...")
            for physical_led_id in range(led_count):
                # Get the matrix column index for this physical LED
                matrix_idx = self.led_spatial_mapping.get(
                    physical_led_id, physical_led_id
                )
                if matrix_idx >= led_count:
                    logger.warning(
                        f"Matrix index {matrix_idx} out of range for physical LED {physical_led_id}, using LED ID"
                    )
                    matrix_idx = physical_led_id

                # Get the LED column from sparse matrix
                led_column = self.sparse_matrix[:, matrix_idx].toarray().flatten()

                # Reshape from flattened RGB to (height, width, 3)
                # The sparse matrix uses channel-separate blocks format
                total_pixels = FRAME_HEIGHT * FRAME_WIDTH * 3
                if len(led_column) != total_pixels:
                    logger.warning(
                        f"Column length {len(led_column)} != expected {total_pixels}"
                    )
                    led_column = np.pad(
                        led_column, (0, max(0, total_pixels - len(led_column)))
                    )[:total_pixels]

                # Reshape from channel-separate blocks format to (height, width, 3) format
                try:
                    # The sparse matrix uses channel-separate blocks format:
                    # - First block: R channel data (pixels_per_channel elements)
                    # - Second block: G channel data (pixels_per_channel elements)
                    # - Third block: B channel data (pixels_per_channel elements)
                    pixels_per_channel = FRAME_HEIGHT * FRAME_WIDTH

                    if len(led_column) == pixels_per_channel * 3:
                        # Extract each channel block
                        r_block = led_column[:pixels_per_channel]
                        g_block = led_column[
                            pixels_per_channel : 2 * pixels_per_channel
                        ]
                        b_block = led_column[
                            2 * pixels_per_channel : 3 * pixels_per_channel
                        ]

                        # Reshape each channel to (height, width)
                        r_channel = r_block.reshape(FRAME_HEIGHT, FRAME_WIDTH)
                        g_channel = g_block.reshape(FRAME_HEIGHT, FRAME_WIDTH)
                        b_channel = b_block.reshape(FRAME_HEIGHT, FRAME_WIDTH)

                        # Stack channels to create (height, width, 3)
                        pattern_hwc = np.stack(
                            [r_channel, g_channel, b_channel], axis=2
                        )

                        # Convert to uint8 and store
                        patterns[physical_led_id] = (pattern_hwc * 255).astype(np.uint8)
                    else:
                        logger.warning(
                            f"LED {matrix_idx}: Column length {len(led_column)} != expected {pixels_per_channel * 3}"
                        )
                        # Fallback to zeros
                        patterns[physical_led_id] = np.zeros(
                            (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
                        )

                except ValueError as ve:
                    logger.warning(f"Reshape failed for LED {matrix_idx}: {ve}")
                    # Fallback to zeros
                    patterns[physical_led_id] = np.zeros(
                        (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
                    )

                # Progress reporting
                if (matrix_idx + 1) % max(1, led_count // 10) == 0:
                    logger.info(
                        f"Converted {matrix_idx + 1}/{led_count} patterns to dense format..."
                    )

            logger.info(
                f"Successfully converted sparse matrix to dense patterns: {patterns.shape}"
            )
            return patterns

        except Exception as e:
            logger.error(f"Failed to convert sparse to dense patterns: {e}")
            import traceback

            logger.error(traceback.format_exc())
            # Return empty patterns array on error
            return np.zeros(
                (
                    led_count if "led_count" in locals() else LED_COUNT,
                    FRAME_HEIGHT,
                    FRAME_WIDTH,
                    3,
                ),
                dtype=np.uint8,
            )

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Main visualization page."""
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/api/metadata")
        def get_metadata():
            """Get pattern metadata."""
            metadata_response = {
                "metadata": self.metadata,
                "led_count": LED_COUNT,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "patterns_loaded": self.diffusion_patterns is not None,
                "is_sparse_format": self.is_sparse_format,
                "supports_storage_order": self.is_sparse_format
                and self.led_spatial_mapping is not None,
            }

            # Add sparse matrix info if available
            if self.is_sparse_format and self.sparse_matrix is not None:
                metadata_response["sparse_info"] = {
                    "matrix_shape": list(self.sparse_matrix.shape),
                    "nnz": self.sparse_matrix.nnz,
                    "sparsity_percent": self.sparse_matrix.nnz
                    / (self.sparse_matrix.shape[0] * self.sparse_matrix.shape[1])
                    * 100,
                    "memory_mb": self.sparse_matrix.data.nbytes / (1024 * 1024),
                }

            return jsonify(metadata_response)

        @self.app.route("/api/patterns")
        def get_patterns():
            """Get pattern list with thumbnails."""
            if self.diffusion_patterns is None:
                return jsonify({"error": "No patterns loaded"}), 404

            try:
                page = int(request.args.get("page", 0))
                per_page = int(request.args.get("per_page", 50))
                channel = request.args.get("channel", "all")
                order = request.args.get(
                    "order", "numerical"
                )  # "numerical" or "storage"

                patterns = []
                start_idx = page * per_page
                end_idx = min(start_idx + per_page, LED_COUNT)

                # Create the LED order list based on the requested ordering
                if (
                    order == "storage"
                    and self.is_sparse_format
                    and self.led_spatial_mapping
                ):
                    # Storage order: show LEDs in the order they appear in the sparse matrix columns
                    # Create reverse mapping: matrix_column -> physical_led_id
                    reverse_mapping = {
                        matrix_idx: physical_id
                        for physical_id, matrix_idx in self.led_spatial_mapping.items()
                    }
                    led_order = [reverse_mapping.get(i, i) for i in range(LED_COUNT)]
                else:
                    # Numerical order: show LEDs in physical ID order (0, 1, 2, ...)
                    led_order = list(range(LED_COUNT))

                # Get the LEDs for this page
                page_leds = led_order[start_idx:end_idx]

                for led_idx in page_leds:
                    if channel == "all":
                        # Create composite RGB image - patterns are (height, width, 3)
                        rgb_pattern = self.diffusion_patterns[
                            led_idx
                        ]  # Already in HWC format
                    else:
                        # Single channel - extract the specific channel
                        channel_idx = {"red": 0, "green": 1, "blue": 2}.get(channel, 0)
                        single_channel = self.diffusion_patterns[
                            led_idx, :, :, channel_idx
                        ]  # (height, width)
                        rgb_pattern = np.stack([single_channel] * 3, axis=-1)

                    # Create thumbnail
                    thumbnail = self._create_thumbnail(rgb_pattern, size=(150, 90))

                    patterns.append(
                        {
                            "led_id": led_idx,
                            "thumbnail": thumbnail,
                            "max_intensity": float(np.max(rgb_pattern)),
                            "center_of_mass": self._calculate_center_of_mass(
                                rgb_pattern
                            ).tolist(),
                        }
                    )

                return jsonify(
                    {
                        "patterns": patterns,
                        "page": page,
                        "per_page": per_page,
                        "total_leds": LED_COUNT,
                        "total_pages": (LED_COUNT + per_page - 1) // per_page,
                        "order": order,
                        "supports_storage_order": self.is_sparse_format
                        and self.led_spatial_mapping is not None,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to generate patterns: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/pattern/<int:led_id>")
        def get_pattern_detail(led_id):
            """Get detailed view of specific LED pattern."""
            if self.diffusion_patterns is None:
                return jsonify({"error": "No patterns loaded"}), 404

            if led_id >= LED_COUNT:
                return jsonify({"error": "Invalid LED ID"}), 400

            try:
                pattern_data = {"led_id": led_id, "channels": {}}

                channel_names = ["red", "green", "blue"]

                for ch_idx, ch_name in enumerate(channel_names):
                    pattern = self.diffusion_patterns[led_id, :, :, ch_idx]

                    # Full resolution image
                    full_image = self._pattern_to_base64(pattern)

                    # Statistics
                    stats = {
                        "max_intensity": float(np.max(pattern)),
                        "min_intensity": float(np.min(pattern)),
                        "mean_intensity": float(np.mean(pattern)),
                        "std_intensity": float(np.std(pattern)),
                        "center_of_mass": self._calculate_center_of_mass(
                            pattern
                        ).tolist(),
                    }

                    pattern_data["channels"][ch_name] = {
                        "image": full_image,
                        "statistics": stats,
                    }

                # Create composite RGB view
                rgb_pattern = self.diffusion_patterns[
                    led_id
                ]  # Already in (height, width, 3) format

                pattern_data["composite"] = {
                    "image": self._create_full_image(rgb_pattern),
                    "statistics": {
                        "max_intensity": float(np.max(rgb_pattern)),
                        "center_of_mass": self._calculate_center_of_mass(
                            rgb_pattern
                        ).tolist(),
                    },
                }

                return jsonify(pattern_data)

            except Exception as e:
                logger.error(f"Failed to get pattern detail: {e}")
                return jsonify({"error": str(e)}), 500

    def _create_thumbnail(self, pattern: np.ndarray, size: Tuple[int, int]) -> str:
        """Create base64 encoded thumbnail."""
        try:
            if not PIL_AVAILABLE:
                return ""

            # Pattern is already uint8 (0-255), no normalization needed
            normalized = pattern.astype(np.uint8)

            # Create PIL image
            img = Image.fromarray(normalized)
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
        """Create base64 encoded full resolution image."""
        try:
            if not PIL_AVAILABLE:
                return ""

            # Handle different input shapes
            if len(pattern.shape) == 2:
                # Single channel - convert to RGB (pattern is already uint8)
                rgb_array = np.stack([pattern] * 3, axis=-1)
            else:
                # Multi-channel (pattern is already uint8)
                rgb_array = pattern

            # Create PIL image
            img = Image.fromarray(rgb_array)

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

            # Pattern is already uint8 (0-255)
            # Create PIL image
            img = Image.fromarray(pattern, mode="L")

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
                return np.array([FRAME_WIDTH // 2, FRAME_HEIGHT // 2])

            y_indices, x_indices = np.indices(intensity.shape)
            x_cm = np.sum(x_indices * intensity) / total_mass
            y_cm = np.sum(y_indices * intensity) / total_mass

            return np.array([x_cm, y_cm])

        except Exception:
            return np.array([FRAME_WIDTH // 2, FRAME_HEIGHT // 2])

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

        select, button, input {
            padding: 8px 15px;
            border: 1px solid #555;
            border-radius: 4px;
            background-color: #333;
            color: #fff;
        }

        button:hover {
            background-color: #444;
            cursor: pointer;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }

        .pattern-card {
            background-color: #2a2a2a;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        .pattern-card:hover {
            background-color: #3a3a3a;
        }

        .pattern-card img {
            width: 100%;
            border-radius: 4px;
            margin-bottom: 8px;
        }

        .pattern-info {
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

        .loading {
            text-align: center;
            padding: 50px;
            font-size: 18px;
            color: #888;
        }

        .error {
            text-align: center;
            padding: 50px;
            font-size: 18px;
            color: #ff6b6b;
        }

        .metadata {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .metadata h3 {
            margin-top: 0;
            color: #4a9eff;
        }

        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }

        .metadata-item {
            display: flex;
            justify-content: space-between;
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
        }

        .modal-content {
            background-color: #2a2a2a;
            margin: 2% auto;
            padding: 20px;
            border-radius: 8px;
            width: 90%;
            max-width: 1200px;
            max-height: 90%;
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
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .channel-view {
            text-align: center;
        }

        .channel-view h4 {
            color: #4a9eff;
            margin-bottom: 10px;
        }

        .channel-view img {
            width: 100%;
            max-width: 400px;
            border-radius: 4px;
            margin-bottom: 10px;
        }

        .stats-table {
            font-size: 12px;
            color: #ccc;
            text-align: left;
        }

        .stats-table tr {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Diffusion Pattern Visualizer</h1>
            <p>Interactive visualization of LED diffusion patterns</p>
        </div>

        <div id="metadata" class="metadata" style="display: none;">
            <h3>Pattern Metadata</h3>
            <div id="metadata-content" class="metadata-grid"></div>
        </div>

        <div class="controls">
            <div class="control-group">
                <label>Channel:</label>
                <select id="channelSelect">
                    <option value="all">RGB Composite</option>
                    <option value="red">Red Channel</option>
                    <option value="green">Green Channel</option>
                    <option value="blue">Blue Channel</option>
                </select>
            </div>

            <div class="control-group" id="orderGroup" style="display: none;">
                <label>Order:</label>
                <select id="orderSelect">
                    <option value="numerical">Numerical (0, 1, 2...)</option>
                    <option value="storage">Storage (Spatial)</option>
                </select>
            </div>

            <div class="control-group">
                <label>Per Page:</label>
                <select id="perPageSelect">
                    <option value="25">25</option>
                    <option value="50" selected>50</option>
                    <option value="100">100</option>
                </select>

                <button onclick="refreshPatterns()">Refresh</button>
            </div>
        </div>

        <div class="pagination" id="pagination-top"></div>

        <div id="content">
            <div class="loading">Loading patterns...</div>
        </div>

        <div class="pagination" id="pagination-bottom"></div>
    </div>

    <!-- Pattern Detail Modal -->
    <div id="patternModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <h2 id="modalTitle">Pattern Detail</h2>
            <div id="modalContent"></div>
        </div>
    </div>

    <script>
        let currentPage = 0;
        let currentChannel = 'all';
        let currentPerPage = 50;
        let currentOrder = 'numerical';
        let totalPages = 0;
        let supportsStorageOrder = false;

        // Load metadata and initial patterns
        document.addEventListener('DOMContentLoaded', function() {
            loadMetadata();
            loadPatterns();

            // Setup event listeners
            document.getElementById('channelSelect').addEventListener('change', function() {
                currentChannel = this.value;
                currentPage = 0;
                loadPatterns();
            });

            document.getElementById('orderSelect').addEventListener('change', function() {
                currentOrder = this.value;
                currentPage = 0;
                loadPatterns();
            });

            document.getElementById('perPageSelect').addEventListener('change', function() {
                currentPerPage = parseInt(this.value);
                currentPage = 0;
                loadPatterns();
            });
        });

        async function loadMetadata() {
            try {
                const response = await fetch('/api/metadata');
                const data = await response.json();

                if (data.metadata) {
                    displayMetadata(data);
                }

                // Show/hide storage order toggle based on support
                supportsStorageOrder = data.supports_storage_order || false;
                const orderGroup = document.getElementById('orderGroup');
                if (supportsStorageOrder) {
                    orderGroup.style.display = 'flex';
                } else {
                    orderGroup.style.display = 'none';
                }
            } catch (error) {
                console.error('Failed to load metadata:', error);
            }
        }

        function displayMetadata(data) {
            const metadataDiv = document.getElementById('metadata');
            const contentDiv = document.getElementById('metadata-content');

            const metadata = data.metadata;
            let html = '';

            // Display key metadata items
            const items = [
                ['LED Count', data.led_count],
                ['Frame Size', `${data.frame_width} × ${data.frame_height}`],
                ['Data Type', metadata.data_type || 'captured'],
                ['Patterns Loaded', data.patterns_loaded ? 'Yes' : 'No'],
                ['Format', data.is_sparse_format ? 'Sparse Matrix' : 'Dense Array']
            ];

            // Add sparse matrix info if available
            if (data.sparse_info) {
                items.push(['Matrix Shape', `${data.sparse_info.matrix_shape[0]} × ${data.sparse_info.matrix_shape[1]}`]);
                items.push(['Non-zero Entries', data.sparse_info.nnz.toLocaleString()]);
                items.push(['Sparsity', `${data.sparse_info.sparsity_percent.toFixed(3)}%`]);
                items.push(['Matrix Memory', `${data.sparse_info.memory_mb.toFixed(1)} MB`]);
                items.push(['Spatial Ordering', data.supports_storage_order ? 'Available' : 'Not Available']);
            }

            if (metadata.capture_timestamp) {
                items.push(['Capture Time', new Date(metadata.capture_timestamp * 1000).toLocaleString()]);
            }

            items.forEach(([key, value]) => {
                html += `<div class="metadata-item"><span>${key}:</span><span>${value}</span></div>`;
            });

            contentDiv.innerHTML = html;
            metadataDiv.style.display = 'block';
        }

        async function loadPatterns() {
            const contentDiv = document.getElementById('content');
            contentDiv.innerHTML = '<div class="loading">Loading patterns...</div>';

            try {
                const url = `/api/patterns?page=${currentPage}&per_page=${currentPerPage}&channel=${currentChannel}&order=${currentOrder}`;
                const response = await fetch(url);
                const data = await response.json();

                if (data.error) {
                    contentDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                    return;
                }

                totalPages = data.total_pages;
                displayPatterns(data.patterns);
                updatePagination();

            } catch (error) {
                console.error('Failed to load patterns:', error);
                contentDiv.innerHTML = '<div class="error">Failed to load patterns</div>';
            }
        }

        function displayPatterns(patterns) {
            const contentDiv = document.getElementById('content');
            let html = '<div class="grid">';

            patterns.forEach(pattern => {
                const intensity = pattern.max_intensity.toFixed(3);
                const centerX = pattern.center_of_mass[0].toFixed(1);
                const centerY = pattern.center_of_mass[1].toFixed(1);

                html += `
                    <div class="pattern-card" onclick="showPatternDetail(${pattern.led_id})">
                        <img src="${pattern.thumbnail}" alt="LED ${pattern.led_id}" />
                        <div class="pattern-info">
                            <div>LED ${pattern.led_id}</div>
                            <div>Max: ${intensity}</div>
                            <div>Center: (${centerX}, ${centerY})</div>
                        </div>
                    </div>
                `;
            });

            html += '</div>';
            contentDiv.innerHTML = html;
        }

        function updatePagination() {
            const paginationHTML = createPaginationHTML();
            document.getElementById('pagination-top').innerHTML = paginationHTML;
            document.getElementById('pagination-bottom').innerHTML = paginationHTML;
        }

        function createPaginationHTML() {
            if (totalPages <= 1) return '';

            let html = '';

            // Previous button
            if (currentPage > 0) {
                html += `<button onclick="changePage(${currentPage - 1})">Previous</button>`;
            }

            // Page numbers
            const startPage = Math.max(0, currentPage - 2);
            const endPage = Math.min(totalPages - 1, currentPage + 2);

            if (startPage > 0) {
                html += `<button onclick="changePage(0)">1</button>`;
                if (startPage > 1) html += '<span>...</span>';
            }

            for (let i = startPage; i <= endPage; i++) {
                const isActive = i === currentPage ? 'style="background-color: #4a9eff;"' : '';
                html += `<button ${isActive} onclick="changePage(${i})">${i + 1}</button>`;
            }

            if (endPage < totalPages - 1) {
                if (endPage < totalPages - 2) html += '<span>...</span>';
                html += `<button onclick="changePage(${totalPages - 1})">${totalPages}</button>`;
            }

            // Next button
            if (currentPage < totalPages - 1) {
                html += `<button onclick="changePage(${currentPage + 1})">Next</button>`;
            }

            return html;
        }

        function changePage(page) {
            currentPage = page;
            loadPatterns();
        }

        function refreshPatterns() {
            loadPatterns();
        }

        async function showPatternDetail(ledId) {
            const modal = document.getElementById('patternModal');
            const title = document.getElementById('modalTitle');
            const content = document.getElementById('modalContent');

            title.textContent = `LED ${ledId} - Diffusion Pattern Detail`;
            content.innerHTML = '<div class="loading">Loading pattern detail...</div>';
            modal.style.display = 'block';

            try {
                const response = await fetch(`/api/pattern/${ledId}`);
                const data = await response.json();

                if (data.error) {
                    content.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                    return;
                }

                displayPatternDetail(data);

            } catch (error) {
                console.error('Failed to load pattern detail:', error);
                content.innerHTML = '<div class="error">Failed to load pattern detail</div>';
            }
        }

        function displayPatternDetail(data) {
            const content = document.getElementById('modalContent');
            let html = '<div class="pattern-detail">';

            // Composite view
            if (data.composite) {
                html += `
                    <div class="channel-view">
                        <h4>RGB Composite</h4>
                        <img src="${data.composite.image}" alt="RGB Composite" />
                        <table class="stats-table">
                            <tr><td>Max Intensity:</td><td>${data.composite.statistics.max_intensity.toFixed(3)}</td></tr>
                            <tr><td>Center of Mass:</td><td>(${data.composite.statistics.center_of_mass[0].toFixed(1)}, ${data.composite.statistics.center_of_mass[1].toFixed(1)})</td></tr>
                        </table>
                    </div>
                `;
            }

            // Individual channels
            const channels = ['red', 'green', 'blue'];
            const channelColors = ['#ff6b6b', '#51cf66', '#4dabf7'];

            channels.forEach((channel, index) => {
                if (data.channels[channel]) {
                    const channelData = data.channels[channel];
                    const stats = channelData.statistics;

                    html += `
                        <div class="channel-view">
                            <h4 style="color: ${channelColors[index]}">${channel.charAt(0).toUpperCase() + channel.slice(1)} Channel</h4>
                            <img src="${channelData.image}" alt="${channel} channel" />
                            <table class="stats-table">
                                <tr><td>Max Intensity:</td><td>${stats.max_intensity.toFixed(3)}</td></tr>
                                <tr><td>Mean Intensity:</td><td>${stats.mean_intensity.toFixed(3)}</td></tr>
                                <tr><td>Std Deviation:</td><td>${stats.std_intensity.toFixed(3)}</td></tr>
                                <tr><td>Center of Mass:</td><td>(${stats.center_of_mass[0].toFixed(1)}, ${stats.center_of_mass[1].toFixed(1)})</td></tr>
                            </table>
                        </div>
                    `;
                }
            });

            html += '</div>';
            content.innerHTML = html;
        }

        function closeModal() {
            document.getElementById('patternModal').style.display = 'none';
        }

        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('patternModal');
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize diffusion patterns in web interface"
    )
    parser.add_argument(
        "--patterns",
        help="Path to diffusion patterns file (.npz). Supports both dense format and sparse matrix format (auto-detects sparse files)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server")
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="[DEPRECATED] Use pre-generated patterns from generate_synthetic_patterns.py instead",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check PIL availability
    if not PIL_AVAILABLE:
        logger.warning("PIL/Pillow not available - image generation will be disabled")
        logger.warning("Install with: pip install Pillow")

    # Create visualizer - synthetic generation now deprecated
    if not args.patterns:
        logger.error("No patterns file provided.")
        logger.error(
            "Generate patterns first with: python tools/generate_synthetic_patterns.py [--sparse]"
        )
        logger.error(
            "For sparse format: python tools/generate_synthetic_patterns.py --sparse --output test_patterns.npz"
        )
        return 1

    visualizer = DiffusionPatternVisualizer(
        patterns_file=args.patterns, use_synthetic=False
    )

    try:
        # Run web server
        visualizer.run(host=args.host, port=args.port, debug=args.debug)

    except KeyboardInterrupt:
        logger.info("Server stopped by user")

    except Exception as e:
        logger.error(f"Server failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
