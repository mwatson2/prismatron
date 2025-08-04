/*
 * Test basic WMMA shapes step by step
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" {

__global__ void test_basic_wmma() {
    // 16x16x16 - definitely supported
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_16;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_16;
    
    wmma::fill_fragment(c_16, 0.0f);
}

}