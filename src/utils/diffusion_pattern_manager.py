"""
Diffusion Pattern Manager - Unified pattern generation, loading, and saving.

This module provides a unified interface for working with LED diffusion patterns,
supporting both synthetic generation and real capture workflows. It standardizes
on planar format (3, H, W) throughout and handles A^T @ A precomputation.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class DiffusionPatternManager:
    """
    Unified manager for LED diffusion patterns supporting generation, loading, and saving.

    Key features:
    - Standardizes on planar format (3, H, W) for frames
    - Incrementally builds sparse matrix A during generation
    - Precomputes dense A^T @ A for optimization
    - Supports both synthetic and captured patterns
    - Provides unified load/save interface

    Format specifications:
    - Frames: (3, height, width) - planar RGB
    - Diffusion tensor A: (led_count, 3, height, width)
    - Sparse matrix A: (pixels*3, led_count*3) block diagonal
    - Dense ATA: (led_count, led_count, 3) precomputed
    - LED output: (3, led_count) - planar RGB
    """

    def __init__(self, led_count: int, frame_height: int = 640, frame_width: int = 800):
        """
        Initialize diffusion pattern manager.

        Args:
            led_count: Number of LEDs in the display
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
        """
        self.led_count = led_count
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.pixels_per_frame = frame_height * frame_width

        # Pattern data storage
        self._diffusion_tensor = None  # (led_count, 3, height, width)
        self._sparse_matrices = None  # Dict with CSC matrices
        self._dense_ata = None  # (led_count, led_count, 3)
        self._led_positions = None  # (led_count, 2)
        self._led_spatial_mapping = None  # Dict mapping

        # Generation state
        self._generation_complete = False
        self._accumulated_patterns = []

        logger.info(
            f"DiffusionPatternManager initialized: {led_count} LEDs, {frame_height}×{frame_width} frames"
        )

    @property
    def is_loaded(self) -> bool:
        """Check if patterns are loaded and ready."""
        return (
            self._sparse_matrices is not None
            and self._dense_ata is not None
            and self._led_positions is not None
        )

    @property
    def is_generation_complete(self) -> bool:
        """Check if pattern generation is complete."""
        return self._generation_complete

    def get_frame_shape(self) -> Tuple[int, int, int]:
        """Get expected frame shape in planar format."""
        return (3, self.frame_height, self.frame_width)

    def get_led_output_shape(self) -> Tuple[int, int]:
        """Get LED output shape in planar format."""
        return (3, self.led_count)

    def start_pattern_generation(self, led_positions: Optional[np.ndarray] = None):
        """
        Start incremental pattern generation process.

        Args:
            led_positions: Optional (led_count, 2) array of LED positions
        """
        logger.info("Starting pattern generation...")
        self._accumulated_patterns = []
        self._generation_complete = False

        if led_positions is not None:
            self._led_positions = np.array(led_positions)
            assert self._led_positions.shape == (
                self.led_count,
                2,
            ), f"LED positions must be ({self.led_count}, 2), got {self._led_positions.shape}"
        else:
            # Generate default positions
            self._generate_default_led_positions()

        # Initialize spatial mapping
        self._led_spatial_mapping = {i: i for i in range(self.led_count)}

        logger.info(f"Pattern generation started for {self.led_count} LEDs")

    def add_led_pattern(self, led_id: int, diffusion_frame: np.ndarray):
        """
        Add diffusion pattern for a single LED.

        Args:
            led_id: LED identifier (0 to led_count-1)
            diffusion_frame: Diffusion pattern in planar format (3, height, width)
        """
        assert (
            0 <= led_id < self.led_count
        ), f"LED ID {led_id} out of range [0, {self.led_count})"
        assert (
            diffusion_frame.shape == self.get_frame_shape()
        ), f"Frame shape must be {self.get_frame_shape()}, got {diffusion_frame.shape}"

        # Store pattern for this LED
        pattern_data = {
            "led_id": led_id,
            "pattern": diffusion_frame.copy(),
            "timestamp": time.time(),
        }
        self._accumulated_patterns.append(pattern_data)

        logger.debug(
            f"Added pattern for LED {led_id}, total patterns: {len(self._accumulated_patterns)}"
        )

    def finalize_pattern_generation(self, sparse_format: str = "csc") -> Dict[str, any]:
        """
        Finalize pattern generation and compute A^T @ A.

        Args:
            sparse_format: Sparse matrix format ('csc', 'csr', or 'coo')

        Returns:
            Dictionary with generation statistics
        """
        if not self._accumulated_patterns:
            raise ValueError("No patterns accumulated - call add_led_pattern() first")

        logger.info("Finalizing pattern generation...")
        start_time = time.time()

        # Build diffusion tensor from accumulated patterns
        self._build_diffusion_tensor()

        # Create sparse matrices
        sparse_stats = self._create_sparse_matrices(sparse_format)

        # Compute dense A^T @ A
        ata_stats = self._compute_dense_ata()

        # Mark generation as complete
        self._generation_complete = True

        generation_time = time.time() - start_time

        stats = {
            "generation_time": generation_time,
            "led_count": self.led_count,
            "patterns_processed": len(self._accumulated_patterns),
            "sparse_stats": sparse_stats,
            "ata_stats": ata_stats,
            "frame_shape": self.get_frame_shape(),
            "led_output_shape": self.get_led_output_shape(),
        }

        logger.info(f"Pattern generation completed in {generation_time:.2f}s")
        return stats

    def save_patterns(
        self, filepath: Union[str, Path], include_diffusion_tensor: bool = False
    ):
        """
        Save patterns to file.

        Args:
            filepath: Output file path (.npz format)
            include_diffusion_tensor: Whether to save full diffusion tensor (large)
        """
        if not self.is_generation_complete and not self.is_loaded:
            raise ValueError("No patterns to save - generate or load patterns first")

        filepath = Path(filepath)
        logger.info(f"Saving patterns to {filepath}")

        # Prepare data for saving
        save_data = {
            "led_count": self.led_count,
            "frame_height": self.frame_height,
            "frame_width": self.frame_width,
            "led_positions": self._led_positions,
            "led_spatial_mapping": self._led_spatial_mapping,
            "format_version": "2.0_planar",  # Mark as new planar format
        }

        # Add sparse matrices
        if self._sparse_matrices:
            for name, matrix in self._sparse_matrices.items():
                save_data[f"sparse_{name}_data"] = matrix.data
                save_data[f"sparse_{name}_indices"] = matrix.indices
                save_data[f"sparse_{name}_indptr"] = matrix.indptr
                save_data[f"sparse_{name}_shape"] = matrix.shape

        # Add dense A^T @ A
        if self._dense_ata is not None:
            save_data["dense_ata"] = self._dense_ata

        # Optionally add full diffusion tensor
        if include_diffusion_tensor and self._diffusion_tensor is not None:
            save_data["diffusion_tensor"] = self._diffusion_tensor

        # Save to file
        np.savez_compressed(filepath, **save_data)

        file_size_mb = filepath.stat().st_size / (1024**2)
        logger.info(f"Patterns saved to {filepath} ({file_size_mb:.1f}MB)")

    def load_patterns(self, filepath: Union[str, Path]) -> Dict[str, any]:
        """
        Load patterns from file.

        Args:
            filepath: Input file path (.npz format)

        Returns:
            Dictionary with loading statistics
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Pattern file not found: {filepath}")

        logger.info(f"Loading patterns from {filepath}")
        start_time = time.time()

        # Load data
        data = np.load(filepath, allow_pickle=True)

        # Check format version
        format_version = data.get("format_version", "1.0_interleaved")
        if format_version == "2.0_planar":
            stats = self._load_planar_format(data)
        else:
            # Convert from old interleaved format
            stats = self._load_and_convert_interleaved_format(data)

        loading_time = time.time() - start_time
        stats["loading_time"] = loading_time

        logger.info(f"Patterns loaded in {loading_time:.2f}s")
        return stats

    def get_sparse_matrices(self) -> Dict[str, sp.spmatrix]:
        """Get sparse matrices for optimization."""
        if self._sparse_matrices is None:
            raise ValueError(
                "Sparse matrices not available - generate or load patterns first"
            )
        return self._sparse_matrices.copy()

    def get_dense_ata(self) -> np.ndarray:
        """Get dense A^T @ A matrices."""
        if self._dense_ata is None:
            raise ValueError(
                "Dense A^T @ A not available - generate or load patterns first"
            )
        return self._dense_ata.copy()

    def get_led_positions(self) -> np.ndarray:
        """Get LED positions."""
        if self._led_positions is None:
            raise ValueError("LED positions not available")
        return self._led_positions.copy()

    def _generate_default_led_positions(self):
        """Generate default LED positions for synthetic patterns."""
        # Create random positions within frame bounds
        np.random.seed(42)  # Reproducible
        self._led_positions = np.random.rand(self.led_count, 2)
        self._led_positions[:, 0] *= self.frame_width  # X coordinates
        self._led_positions[:, 1] *= self.frame_height  # Y coordinates

        logger.debug(f"Generated default LED positions for {self.led_count} LEDs")

    def _build_diffusion_tensor(self):
        """Build diffusion tensor from accumulated patterns."""
        logger.info("Building diffusion tensor from accumulated patterns...")

        # Initialize tensor
        self._diffusion_tensor = np.zeros(
            (self.led_count, 3, self.frame_height, self.frame_width), dtype=np.float32
        )

        # Fill tensor from accumulated patterns or loaded patterns
        if self._accumulated_patterns:
            # Use accumulated patterns (during generation)
            for pattern_data in self._accumulated_patterns:
                led_id = pattern_data["led_id"]
                pattern = pattern_data["pattern"]
                self._diffusion_tensor[led_id] = pattern
        elif self._patterns:
            # Use loaded patterns (from converted data)
            for led_id, pattern in self._patterns.items():
                self._diffusion_tensor[led_id] = pattern
        else:
            raise ValueError("No patterns available to build diffusion tensor")

        logger.info(f"Diffusion tensor built: {self._diffusion_tensor.shape}")

    def _create_sparse_matrices(self, sparse_format: str) -> Dict[str, any]:
        """Create sparse matrices from diffusion tensor."""
        logger.info(f"Creating sparse matrices in {sparse_format} format...")

        if self._diffusion_tensor is None:
            raise ValueError("Diffusion tensor not built")

        # Reshape diffusion tensor for sparse matrix creation
        # From: (led_count, 3, height, width)
        # To: (pixels*3, led_count*3) block diagonal

        A_matrices = {}

        # Create separate matrices for each channel
        for channel, channel_name in enumerate(["r", "g", "b"]):
            # Extract channel data: (led_count, height, width)
            channel_data = self._diffusion_tensor[:, channel, :, :]

            # Reshape to (pixels, led_count)
            A_channel = channel_data.reshape(self.led_count, -1).T

            # Convert to sparse format
            if sparse_format == "csc":
                A_sparse = sp.csc_matrix(A_channel, dtype=np.float32)
            elif sparse_format == "csr":
                A_sparse = sp.csr_matrix(A_channel, dtype=np.float32)
            elif sparse_format == "coo":
                A_sparse = sp.coo_matrix(A_channel, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported sparse format: {sparse_format}")

            A_matrices[f"A_{channel_name}"] = A_sparse

        # Create combined block diagonal matrix
        A_combined = sp.block_diag(
            [A_matrices["A_r"], A_matrices["A_g"], A_matrices["A_b"]],
            format=sparse_format,
        )
        A_matrices["A_combined"] = A_combined

        self._sparse_matrices = A_matrices

        # Calculate statistics
        total_nnz = sum(matrix.nnz for matrix in A_matrices.values())

        # Calculate memory usage (handle different sparse formats)
        total_bytes = 0
        for matrix in A_matrices.values():
            total_bytes += matrix.data.nbytes
            if hasattr(matrix, "indices"):  # CSC/CSR
                total_bytes += matrix.indices.nbytes
                total_bytes += matrix.indptr.nbytes
            elif hasattr(matrix, "row"):  # COO
                total_bytes += matrix.row.nbytes
                total_bytes += matrix.col.nbytes

        total_size_mb = total_bytes / (1024**2)

        stats = {
            "format": sparse_format,
            "matrices_created": len(A_matrices),
            "total_nnz": total_nnz,
            "total_size_mb": total_size_mb,
            "combined_shape": A_combined.shape,
            "combined_density": A_combined.nnz
            / (A_combined.shape[0] * A_combined.shape[1]),
        }

        logger.info(
            f"Sparse matrices created: {total_nnz:,} NNZ, {total_size_mb:.1f}MB"
        )
        return stats

    def _compute_dense_ata(self) -> Dict[str, any]:
        """Compute dense A^T @ A matrices."""
        logger.info("Computing dense A^T @ A matrices...")

        if self._sparse_matrices is None:
            raise ValueError("Sparse matrices not created")

        start_time = time.time()

        # Compute A^T @ A for each channel separately
        ata_channels = []

        for channel_name in ["A_r", "A_g", "A_b"]:
            A_channel = self._sparse_matrices[channel_name]

            # Compute A^T @ A (results in led_count × led_count dense matrix)
            ATA_channel = A_channel.T @ A_channel
            ATA_dense = ATA_channel.toarray().astype(np.float32)

            ata_channels.append(ATA_dense)

        # Stack into (led_count, led_count, 3) tensor
        self._dense_ata = np.stack(ata_channels, axis=2)

        computation_time = time.time() - start_time

        # Calculate statistics
        ata_memory_mb = self._dense_ata.nbytes / (1024**2)
        ata_density = np.count_nonzero(self._dense_ata) / self._dense_ata.size

        stats = {
            "computation_time": computation_time,
            "ata_shape": self._dense_ata.shape,
            "ata_memory_mb": ata_memory_mb,
            "ata_density": ata_density,
        }

        logger.info(
            f"Dense A^T @ A computed: {self._dense_ata.shape}, {ata_memory_mb:.1f}MB, {computation_time:.2f}s"
        )
        return stats

    def _load_planar_format(self, data) -> Dict[str, any]:
        """Load patterns from new planar format."""
        # Load basic parameters
        self.led_count = int(data["led_count"])
        self.frame_height = int(data["frame_height"])
        self.frame_width = int(data["frame_width"])
        self._led_positions = data["led_positions"]
        self._led_spatial_mapping = data["led_spatial_mapping"].item()

        # Load sparse matrices
        self._sparse_matrices = {}
        matrix_names = ["A_r", "A_g", "A_b", "A_combined"]

        for name in matrix_names:
            if f"sparse_{name}_data" in data:
                matrix = sp.csc_matrix(
                    (
                        data[f"sparse_{name}_data"],
                        data[f"sparse_{name}_indices"],
                        data[f"sparse_{name}_indptr"],
                    ),
                    shape=tuple(data[f"sparse_{name}_shape"]),
                )
                self._sparse_matrices[name] = matrix

        # Load dense A^T @ A
        if "dense_ata" in data:
            self._dense_ata = data["dense_ata"]

        # Load diffusion tensor if available
        if "diffusion_tensor" in data:
            self._diffusion_tensor = data["diffusion_tensor"]

        stats = {
            "format": "planar_2.0",
            "led_count": self.led_count,
            "frame_shape": self.get_frame_shape(),
            "matrices_loaded": len(self._sparse_matrices),
            "has_dense_ata": self._dense_ata is not None,
            "has_diffusion_tensor": self._diffusion_tensor is not None,
        }

        return stats

    def _load_and_convert_interleaved_format(self, data) -> Dict[str, any]:
        """Load and convert patterns from old interleaved format."""
        logger.info("Converting from interleaved to planar format...")

        # Import required modules
        import scipy.sparse as sp

        # Load sparse matrix components
        matrix_data = data["matrix_data"]
        matrix_indices = data["matrix_indices"]
        matrix_indptr = data["matrix_indptr"]
        matrix_shape = tuple(data["matrix_shape"])

        logger.info(f"Loading sparse matrix with shape {matrix_shape}")

        # Reconstruct the sparse matrix (CSC format)
        A_combined = sp.csc_matrix(
            (matrix_data, matrix_indices, matrix_indptr), shape=matrix_shape
        )

        # The matrix shape is (pixels, led_count * 3)
        # Where pixels = frame_height * frame_width = 480 * 800 = 384000
        # And led_count * 3 = 1000 * 3 = 3000
        pixels = matrix_shape[0]
        led_count_times_3 = matrix_shape[1]
        led_count = led_count_times_3 // 3

        if pixels != self.frame_height * self.frame_width:
            raise ValueError(
                f"Matrix pixels {pixels} != expected "
                f"{self.frame_height * self.frame_width}"
            )

        if led_count != self.led_count:
            raise ValueError(
                f"Matrix LED count {led_count} != expected {self.led_count}"
            )

        logger.info(f"Matrix format: {pixels} pixels, {led_count} LEDs")

        # Extract RGB channel matrices
        # The matrix structure is: columns 0-999 are R channels, 1000-1999 are G channels, 2000-2999 are B channels
        # Each LED's diffusion pattern spans the full frame (all pixels) for each channel
        logger.info("Extracting RGB channel patterns...")

        # Convert sparse matrices to dense and reshape to planar format
        # For each LED, we need to convert from (H*W) vector to (H, W) and then to planar
        self._patterns = {}

        logger.info("Converting LED patterns to planar format...")
        for led_id in range(led_count):
            # Extract diffusion pattern for this LED from each channel
            # Red channel: column led_id (0-999)
            pattern_r_vec = A_combined[:, led_id].toarray().flatten()
            # Green channel: column led_id + led_count (1000-1999)
            pattern_g_vec = A_combined[:, led_id + led_count].toarray().flatten()
            # Blue channel: column led_id + 2*led_count (2000-2999)
            pattern_b_vec = A_combined[:, led_id + 2 * led_count].toarray().flatten()

            # Reshape each channel from flat vector to (H, W)
            pattern_r = pattern_r_vec.reshape(self.frame_height, self.frame_width)
            pattern_g = pattern_g_vec.reshape(self.frame_height, self.frame_width)
            pattern_b = pattern_b_vec.reshape(self.frame_height, self.frame_width)

            # Combine into planar format (3, H, W)
            pattern_planar = np.stack([pattern_r, pattern_g, pattern_b], axis=0).astype(
                np.float32
            )

            self._patterns[led_id] = pattern_planar

            if led_id % 100 == 0:
                logger.debug(f"Converted LED {led_id}/{led_count}")

        logger.info(f"Converted {len(self._patterns)} LED patterns to planar format")

        # Load LED positions and mapping if available
        if "led_positions" in data:
            self._led_positions = data["led_positions"]
            logger.info(f"Loaded {len(self._led_positions)} LED positions")

        if "led_spatial_mapping" in data:
            mapping_data = data["led_spatial_mapping"]
            if hasattr(mapping_data, "item"):
                self._led_spatial_mapping = mapping_data.item()
            else:
                self._led_spatial_mapping = dict(mapping_data)
            logger.info(
                f"Loaded LED spatial mapping with {len(self._led_spatial_mapping)} entries"
            )

        # Build diffusion tensor from loaded patterns
        logger.info("Building diffusion tensor from loaded patterns...")
        self._build_diffusion_tensor()

        # Create sparse matrices and dense ATA for compatibility
        logger.info("Creating sparse matrices from loaded data...")
        sparse_stats = self._create_sparse_matrices("csc")

        logger.info("Computing dense A^T*A matrices...")
        ata_stats = self._compute_dense_ata()

        # Update metadata
        metadata = data.get("metadata", {})
        if hasattr(metadata, "item"):
            metadata = metadata.item()

        return {
            "led_count": led_count,
            "pattern_count": len(self._patterns),
            "frame_dimensions": (self.frame_height, self.frame_width),
            "format": "planar",
            "converted_from": "sparse_interleaved",
            "original_matrix_shape": matrix_shape,
            "metadata": metadata,
            "sparse_stats": sparse_stats,
            "ata_stats": ata_stats,
        }
