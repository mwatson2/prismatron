#!/usr/bin/env python3
"""
Diffusion Pattern Visualization Tool.

This tool creates a web interface to visualize diffusion patterns with:
1. Grid view of all LED patterns
2. Individual LED/channel navigation
3. Support for both captured and synthetic patterns
4. Interactive controls and filtering

Usage:
    python visualize_diffusion_patterns.py --patterns captured_patterns.npz --host 0.0.0.0 --port 8080
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
    """Web-based diffusion pattern visualization tool."""

    def __init__(self, patterns_file: Optional[str] = None, use_synthetic: bool = True):
        """
        Initialize visualizer.

        Args:
            patterns_file: Path to captured patterns file (.npz)
            use_synthetic: Generate synthetic patterns if no file provided
        """
        self.patterns_file = patterns_file
        self.use_synthetic = use_synthetic

        # Pattern data
        self.diffusion_patterns: Optional[np.ndarray] = None
        self.metadata: Dict = {}

        # Flask app
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)  # Reduce Flask logging

        # Setup routes
        self._setup_routes()

    def load_patterns(self) -> bool:
        """Load diffusion patterns from file or generate synthetic ones."""
        try:
            if self.patterns_file and Path(self.patterns_file).exists():
                logger.info(f"Loading patterns from {self.patterns_file}")
                return self._load_captured_patterns()
            elif self.use_synthetic:
                logger.info("Generating synthetic diffusion patterns")
                return self._generate_synthetic_patterns()
            else:
                logger.error(
                    "No patterns file provided and synthetic patterns disabled"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            return False

    def _load_captured_patterns(self) -> bool:
        """Load captured diffusion patterns from .npz file."""
        try:
            data = np.load(self.patterns_file, allow_pickle=True)

            self.diffusion_patterns = data["diffusion_patterns"]
            self.metadata = data["metadata"].item() if "metadata" in data else {}

            logger.info(f"Loaded patterns: {self.diffusion_patterns.shape}")
            logger.info(f"Metadata: {self.metadata}")

            return True

        except Exception as e:
            logger.error(f"Failed to load captured patterns: {e}")
            return False

    def _generate_synthetic_patterns(self) -> bool:
        """Generate synthetic diffusion patterns for visualization."""
        try:
            logger.info("Generating synthetic diffusion patterns...")

            # Create synthetic patterns with realistic LED diffusion
            # Using uint8 to save memory: 3200×3×480×800×1 = ~3.5GB vs 14GB for float32
            self.diffusion_patterns = np.zeros(
                (LED_COUNT, 3, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8
            )

            # Generate random LED positions
            np.random.seed(42)  # Reproducible positions
            led_positions = np.random.randint(
                0, min(FRAME_WIDTH, FRAME_HEIGHT), size=(LED_COUNT, 2)
            )

            for led_idx in range(LED_COUNT):
                x_center, y_center = led_positions[led_idx]

                # Ensure positions are within frame
                x_center = min(x_center, FRAME_WIDTH - 1)
                y_center = min(y_center, FRAME_HEIGHT - 1)

                for channel in range(3):  # R, G, B
                    # Create Gaussian diffusion pattern
                    pattern = self._create_gaussian_pattern(
                        x_center,
                        y_center,
                        sigma_x=np.random.uniform(20, 60),
                        sigma_y=np.random.uniform(20, 60),
                        intensity=np.random.uniform(80, 255),
                    )

                    self.diffusion_patterns[led_idx, channel] = pattern

            # Create metadata
            self.metadata = {
                "led_count": LED_COUNT,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "channels": 3,
                "data_type": "synthetic",
                "generation_timestamp": time.time(),
            }

            logger.info(
                f"Generated synthetic patterns: {self.diffusion_patterns.shape}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to generate synthetic patterns: {e}")
            return False

    def _create_gaussian_pattern(
        self,
        x_center: int,
        y_center: int,
        sigma_x: float,
        sigma_y: float,
        intensity: float,
    ) -> np.ndarray:
        """Create a 2D Gaussian diffusion pattern."""
        # Create coordinate grids
        x = np.arange(FRAME_WIDTH)
        y = np.arange(FRAME_HEIGHT)
        X, Y = np.meshgrid(x, y)

        # Calculate Gaussian
        pattern = intensity * np.exp(
            -(
                (X - x_center) ** 2 / (2 * sigma_x**2)
                + (Y - y_center) ** 2 / (2 * sigma_y**2)
            )
        )

        # Clip to valid uint8 range and convert
        return np.clip(pattern, 0, 255).astype(np.uint8)

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Main visualization page."""
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/api/metadata")
        def get_metadata():
            """Get pattern metadata."""
            return jsonify(
                {
                    "metadata": self.metadata,
                    "led_count": LED_COUNT,
                    "frame_width": FRAME_WIDTH,
                    "frame_height": FRAME_HEIGHT,
                    "patterns_loaded": self.diffusion_patterns is not None,
                }
            )

        @self.app.route("/api/patterns")
        def get_patterns():
            """Get pattern list with thumbnails."""
            if self.diffusion_patterns is None:
                return jsonify({"error": "No patterns loaded"}), 404

            try:
                page = int(request.args.get("page", 0))
                per_page = int(request.args.get("per_page", 50))
                channel = request.args.get("channel", "all")

                patterns = []
                start_idx = page * per_page
                end_idx = min(start_idx + per_page, LED_COUNT)

                for led_idx in range(start_idx, end_idx):
                    if channel == "all":
                        # Create composite RGB image
                        rgb_pattern = np.stack(
                            [
                                self.diffusion_patterns[led_idx, 0],  # R
                                self.diffusion_patterns[led_idx, 1],  # G
                                self.diffusion_patterns[led_idx, 2],  # B
                            ],
                            axis=-1,
                        )
                    else:
                        # Single channel
                        channel_idx = {"red": 0, "green": 1, "blue": 2}.get(channel, 0)
                        single_channel = self.diffusion_patterns[led_idx, channel_idx]
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
                    pattern = self.diffusion_patterns[led_id, ch_idx]

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
                rgb_pattern = np.stack(
                    [
                        self.diffusion_patterns[led_id, 0],
                        self.diffusion_patterns[led_id, 1],
                        self.diffusion_patterns[led_id, 2],
                    ],
                    axis=-1,
                )

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
        let totalPages = 0;

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
                ['Patterns Loaded', data.patterns_loaded ? 'Yes' : 'No']
            ];

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
                const url = `/api/patterns?page=${currentPage}&per_page=${currentPerPage}&channel=${currentChannel}`;
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
    parser.add_argument("--patterns", help="Path to diffusion patterns file (.npz)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server")
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Disable synthetic pattern generation",
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

    # Create visualizer
    visualizer = DiffusionPatternVisualizer(
        patterns_file=args.patterns, use_synthetic=not args.no_synthetic
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
