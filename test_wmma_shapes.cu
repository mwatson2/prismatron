/*
 * Test which WMMA shapes are actually supported on this GPU
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" {

__global__ void test_wmma_shapes() {
    // Test 16x16x16 (known to work)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_16_16_16;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_16_16_16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_16_16_16;
    
    // Test 8x32x16: A[8x32] @ B[32x16] = C[8x16]
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> a_8_32_16;
    wmma::fragment<wmma::matrix_b, 32, 16, 16, half, wmma::col_major> b_32_16_16;
    wmma::fragment<wmma::accumulator, 8, 16, 16, float> c_8_16_16;
    
    // Test 32x8x16: A[32x8] @ B[8x16] = C[32x16]
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> a_32_8_16;
    wmma::fragment<wmma::matrix_b, 8, 16, 16, half, wmma::col_major> b_8_16_16;
    wmma::fragment<wmma::accumulator, 32, 16, 16, float> c_32_16_16;
}

}