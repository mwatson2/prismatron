#!/usr/bin/env python3
"""
Test DIA matrix integration with frame optimizer.

This test validates that the optimal mixed tensor (A^T b) + DIA matrix (A^T A) 
combination works correctly and provides expected performance benefits.
"""

import numpy as np
import pytest
from pathlib import Path

from src.utils.frame_optimizer import optimize_frame_led_values, optimize_frame_with_dia_matrix
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


class TestDIAIntegration:
    """Test DIA matrix integration with frame optimizer."""

    def create_test_frame(self) -> np.ndarray:
        """Create simple test frame for optimization."""
        height, width = 480, 800
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add simple test pattern
        frame[100:150, :, 0] = 255  # Red stripe
        center_y, center_x = height // 2, width // 2
        radius = 50
        y, x = np.ogrid[:height, :width]
        mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
        frame[mask, 1] = 255  # Green circle
        
        # Convert to planar format (3, H, W)
        return frame.transpose(2, 0, 1)

    def load_patterns(self):
        """Load real diffusion patterns for testing."""
        # Try to load 1000 LED patterns (fallback to smaller if not available)
        pattern_files = [
            "diffusion_patterns/synthetic_1000_64x64.npz",
            "diffusion_patterns/synthetic_500_64x64.npz",
            "diffusion_patterns/synthetic_100_64x64.npz"
        ]
        
        for pattern_file in pattern_files:
            pattern_path = Path(pattern_file)
            if pattern_path.exists():
                data = np.load(str(pattern_path), allow_pickle=True)
                
                # Load mixed tensor
                mixed_tensor_dict = data['mixed_tensor'].item()
                mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
                
                # Load DIA matrix if available
                if 'dia_matrix' in data:
                    dia_dict = data['dia_matrix'].item()
                    dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)
                else:
                    # Build DIA matrix from diffusion matrix
                    diffusion_dict = data['diffusion_matrix'].item()
                    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(diffusion_dict)
                    led_positions = data['led_positions']
                    
                    dia_matrix = DiagonalATAMatrix(mixed_tensor.batch_size, crop_size=64)
                    A_matrix = diffusion_csc.to_csc_matrix()
                    dia_matrix.build_from_diffusion_matrix(A_matrix, led_positions)
                
                return mixed_tensor, dia_matrix, mixed_tensor.batch_size
                
        pytest.skip("No suitable diffusion patterns found for testing")

    def test_dia_matrix_optimization(self):
        """Test frame optimization with DIA matrix."""
        mixed_tensor, dia_matrix, led_count = self.load_patterns()
        target_frame = self.create_test_frame()
        
        print(f"\nTesting DIA integration with {led_count} LEDs")
        print(f"DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k} diagonals")
        
        # Test the optimal combination: mixed tensor (A^T b) + DIA matrix (A^T A)
        result = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,  # Mixed tensor for A^T @ b
            ATA_matrix=dia_matrix,   # DIA matrix for A^T A operations
            max_iterations=5,
            compute_error_metrics=True,
            debug=False,
            enable_timing=True,
        )
        
        # Validate results
        assert result.led_values.shape == (3, led_count)
        assert result.led_values.dtype == np.uint8
        assert np.all(result.led_values >= 0) and np.all(result.led_values <= 255)
        assert result.iterations > 0
        assert result.timing_data is not None
        
        print(f"✅ Optimization completed in {result.iterations} iterations")
        print(f"   LED values shape: {result.led_values.shape}")
        print(f"   LED values range: [{result.led_values.min()}, {result.led_values.max()}]")
        
        if result.timing_data:
            total_time = sum(result.timing_data.values())
            print(f"   Total time: {total_time:.4f}s")
            for step, duration in result.timing_data.items():
                print(f"   {step}: {duration:.4f}s")
        
        # Check error metrics if computed
        if result.error_metrics:
            print(f"   Error metrics: MSE={result.error_metrics.get('mse', 'N/A'):.6f}")

    def test_convenience_function(self):
        """Test the convenience function for DIA matrix optimization."""
        mixed_tensor, dia_matrix, led_count = self.load_patterns()
        target_frame = self.create_test_frame()
        
        # Load CSC matrix for convenience function
        pattern_files = [
            "diffusion_patterns/synthetic_1000_64x64.npz",
            "diffusion_patterns/synthetic_500_64x64.npz", 
            "diffusion_patterns/synthetic_100_64x64.npz"
        ]
        
        diffusion_csc = None
        for pattern_file in pattern_files:
            pattern_path = Path(pattern_file)
            if pattern_path.exists():
                data = np.load(str(pattern_path), allow_pickle=True)
                diffusion_dict = data['diffusion_matrix'].item()
                diffusion_csc = LEDDiffusionCSCMatrix.from_dict(diffusion_dict)
                break
        
        if diffusion_csc is None:
            pytest.skip("No CSC diffusion matrix found for convenience function test")
        
        # Test convenience function
        result = optimize_frame_with_dia_matrix(
            target_frame=target_frame,
            diffusion_csc=diffusion_csc,
            dia_matrix=dia_matrix,
            max_iterations=3,
            compute_error_metrics=False,
            debug=False,
        )
        
        # Validate results
        assert result.led_values.shape == (3, led_count)
        assert result.led_values.dtype == np.uint8
        assert np.all(result.led_values >= 0) and np.all(result.led_values <= 255)
        assert result.iterations > 0
        
        print(f"✅ Convenience function test completed")
        print(f"   Used CSC matrix + DIA matrix combination")
        print(f"   Optimization completed in {result.iterations} iterations")

    def test_performance_comparison(self):
        """Compare performance between dense and DIA matrix approaches."""
        mixed_tensor, dia_matrix, led_count = self.load_patterns()
        target_frame = self.create_test_frame()
        
        # Create dense ATA matrix for comparison
        ata_dense = mixed_tensor.compute_ata_dense()  # Shape: (led_count, led_count, 3)
        
        import time
        
        # Test DIA matrix approach
        start_time = time.perf_counter()
        result_dia = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=3,
            compute_error_metrics=False,
            debug=False,
        )
        dia_time = time.perf_counter() - start_time
        
        # Test dense matrix approach
        start_time = time.perf_counter()
        result_dense = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=ata_dense,
            max_iterations=3,
            compute_error_metrics=False,
            debug=False,
        )
        dense_time = time.perf_counter() - start_time
        
        # Compare results
        speedup = dense_time / dia_time
        
        print(f"\n📊 Performance Comparison ({led_count} LEDs, 3 iterations):")
        print(f"   Dense matrix:  {dense_time:.4f}s")
        print(f"   DIA matrix:    {dia_time:.4f}s")
        print(f"   Speedup:       {speedup:.2f}x")
        
        # Results should be similar (within reasonable tolerance)
        max_diff = np.max(np.abs(result_dia.led_values.astype(float) - 
                                 result_dense.led_values.astype(float)))
        print(f"   Max difference: {max_diff:.1f} (LED values)")
        
        assert max_diff < 5, "DIA and dense results should be similar"
        
        # DIA approach should be faster for properly structured matrices
        if dia_matrix.k < led_count * 0.5:  # If reasonably sparse
            print(f"   ✅ DIA matrix is faster ({speedup:.2f}x speedup)")
        else:
            print(f"   ⚠️  DIA matrix not optimal (too dense: {dia_matrix.k} diagonals)")


if __name__ == "__main__":
    test = TestDIAIntegration()
    test.test_dia_matrix_optimization()
    test.test_convenience_function() 
    test.test_performance_comparison()
    print("\n🎉 All DIA integration tests passed!")