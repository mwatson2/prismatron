/*
 * Test 32x8x16 WMMA shapes
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" {

__global__ void test_32x8x16_wmma() {
    // Test 32x8x16: A[32x16] @ B[16x8] = C[32x8]
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> a_32_16;
    wmma::fragment<wmma::matrix_b, 16, 8, 16, half, wmma::col_major> b_16_8;
    wmma::fragment<wmma::accumulator, 32, 8, 16, float> c_32_8;
    
    wmma::fill_fragment(c_32_8, 0.0f);
}

}