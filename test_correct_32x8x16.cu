/*
 * Test correct 32x8x16 WMMA syntax
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" {

__global__ void test_correct_32x8x16() {
    // For 32x8x16 WMMA: A[32x16] @ B[16x8] = C[32x8]
    // All fragments use the same (M, N, K) = (32, 8, 16)
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 32, 8, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Test basic operations
    __shared__ half shared_a[32 * 16];
    __shared__ half shared_b[16 * 16];  // B is 16x8, padded to 16x16
    __shared__ float shared_c[32 * 16]; // C is 32x8, stored in 32x16
    
    // These should compile if the fragment declarations are correct
    wmma::load_matrix_sync(a_frag, shared_a, 16);  // A is 32x16, ldm=16
    wmma::load_matrix_sync(b_frag, shared_b, 16);  // B is 16x8, ldm=16 (padded)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(shared_c, c_frag, 16, wmma::mem_row_major);  // C is 32x8, ldm=16
}

}