#!/usr/bin/env python3
"""
Debug A^T@A matrix differences between mixed tensor and CSC.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_ata_matrices():
    """Debug A^T@A matrix differences."""
    logger.info("=== Debugging A^T@A Matrix Differences ===")
    
    # Load clipped pattern data
    data = np.load('diffusion_patterns/synthetic_1000_with_ata_clipped.npz', allow_pickle=True)
    
    # Load CSC matrix
    matrix_shape = tuple(data['matrix_shape'])
    matrix_data = data['matrix_data']
    matrix_indices = data['matrix_indices'] 
    matrix_indptr = data['matrix_indptr']
    clipped_csc = sp.csc_matrix((matrix_data, matrix_indices, matrix_indptr), shape=matrix_shape)
    
    # Load precomputed A^T@A matrices (these were computed from the clipped CSC)
    precomputed_ata = data['dense_ata_matrices']  # Shape: (1000, 1000, 3)
    
    logger.info(f"Precomputed A^T@A shape: {precomputed_ata.shape}")
    logger.info(f"CSC matrix shape: {clipped_csc.shape}")
    
    # Recompute A^T@A from CSC to verify
    logger.info("Recomputing A^T@A from CSC matrix...")
    
    # Extract RGB channel matrices
    A_r = clipped_csc[:, 0::3]  # Red channels
    A_g = clipped_csc[:, 1::3]  # Green channels
    A_b = clipped_csc[:, 2::3]  # Blue channels
    
    # Compute A^T@A for each channel
    ata_r_computed = (A_r.T @ A_r).toarray().astype(np.float32)
    ata_g_computed = (A_g.T @ A_g).toarray().astype(np.float32)
    ata_b_computed = (A_b.T @ A_b).toarray().astype(np.float32)
    
    logger.info(f"Computed A^T@A shapes: R={ata_r_computed.shape}, G={ata_g_computed.shape}, B={ata_b_computed.shape}")
    
    # Compare with precomputed matrices
    diff_r = np.abs(precomputed_ata[:, :, 0] - ata_r_computed)
    diff_g = np.abs(precomputed_ata[:, :, 1] - ata_g_computed)
    diff_b = np.abs(precomputed_ata[:, :, 2] - ata_b_computed)
    
    logger.info(f"A^T@A differences:")
    logger.info(f"  Red channel - Max: {diff_r.max():.6e}, Mean: {diff_r.mean():.6e}")
    logger.info(f"  Green channel - Max: {diff_g.max():.6e}, Mean: {diff_g.mean():.6e}")
    logger.info(f"  Blue channel - Max: {diff_b.max():.6e}, Mean: {diff_b.mean():.6e}")
    
    # Check if matrices are identical
    ata_identical = (diff_r.max() < 1e-4 and diff_g.max() < 1e-4 and diff_b.max() < 1e-4)
    
    if ata_identical:
        logger.info("✅ A^T@A matrices are identical - problem is elsewhere")
    else:
        logger.error("❌ A^T@A matrices differ - this could cause optimization differences")
        return False
    
    # Check matrix properties
    logger.info("\n=== A^T@A Matrix Properties ===")
    logger.info(f"Red channel A^T@A:")
    logger.info(f"  Diagonal range: [{np.diag(ata_r_computed).min():.3f}, {np.diag(ata_r_computed).max():.3f}]")
    logger.info(f"  Condition number: {np.linalg.cond(ata_r_computed):.2e}")
    
    logger.info(f"Green channel A^T@A:")
    logger.info(f"  Diagonal range: [{np.diag(ata_g_computed).min():.3f}, {np.diag(ata_g_computed).max():.3f}]")
    logger.info(f"  Condition number: {np.linalg.cond(ata_g_computed):.2e}")
    
    logger.info(f"Blue channel A^T@A:")
    logger.info(f"  Diagonal range: [{np.diag(ata_b_computed).min():.3f}, {np.diag(ata_b_computed).max():.3f}]")
    logger.info(f"  Condition number: {np.linalg.cond(ata_b_computed):.2e}")
    
    # Now let's check if the MIXED TENSOR is using the SAME A^T@A computation
    # The issue might be that mixed tensor and CSC are using different approaches
    logger.info("\n=== Checking Mixed Tensor A^T@A Usage ===")
    
    # Both should use the SAME precomputed A^T@A matrices from the file
    # Let's verify this is happening in the optimizer
    logger.info("Both mixed tensor and CSC modes should use the same precomputed A^T@A matrices.")
    logger.info("The difference must be in the A^T@b calculation or optimization loop.")
    
    return True


if __name__ == "__main__":
    success = debug_ata_matrices()
    sys.exit(0 if success else 1)