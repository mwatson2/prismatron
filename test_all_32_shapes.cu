/*
 * Test all possible 32x8x16 WMMA parameter combinations
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" {

__global__ void test_all_32_shapes() {
    // Try 32x8x16 with different parameter orders
    
    // Attempt 1: Standard interpretation
    // wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> a1;
    // wmma::fragment<wmma::matrix_b, 16, 8, 16, half, wmma::col_major> b1;  // This failed
    
    // Attempt 2: Maybe it's m=32, n=8, k=16 but different matrix shapes
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> a2;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> b2;
    wmma::fragment<wmma::accumulator, 32, 32, 16, float> c2;
    
    // Attempt 3: Check if 32x8x16 means different dimensions
    // Maybe the second parameter is the inner K dimension?
    
    wmma::fill_fragment(c2, 0.0f);
}

}