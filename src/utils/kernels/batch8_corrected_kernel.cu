/*
 * 8-Frame Batch WMMA Kernel with Corrected Vertical Pair Processing
 *
 * Uses 32x8x16 WMMA operations with FP32 input/output, half precision internally
 *
 * Key Architecture:
 * - ATA matrix divided into 32x16 fragments A_{i,j} (vertical pairs of 16x16 blocks)
 * - Input tensor divided into 16x8 fragments B_j
 * - WMMA(A_{i,j}, B_j) → accumulate into output[i*32:(i+1)*32, :]
 * - Sum over j, skipping empty block pairs
 * - FP32 data converted to half for WMMA, results accumulated in FP32
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" {

__global__ void batch8_symmetric_block_pair_multiply_wmma(
    const float* block_data,      // (channels, block_diag_count, 16, 16)
    const int* block_offsets,     // (block_diag_count,)
    const float* input_batch,     // (8, 3, leds)
    float* output_batch,          // (8, 3, leds)
    int batch_size,               // Always 8
    int channels,                 // 3 (RGB)
    int led_blocks,              // Number of 16x16 blocks per dimension
    int block_diag_count,        // Number of diagonal bands
    int leds                     // LED count (multiple of 32)
) {
    // Grid: (channels, led_blocks/2) - each block processes one 32x16 vertical pair (i index)
    // Block: 32 threads (1 warp for WMMA)

    int channel_idx = blockIdx.x;
    int block_pair_i = blockIdx.y;  // i index: which 32x16 vertical pair (0, 1, 2, ...)
    int lane_id = threadIdx.x % 32;

    // Early exit if out of bounds
    if (channel_idx >= channels || block_pair_i * 2 >= led_blocks) {
        return;
    }

    // Calculate the two block row indices in this vertical pair A_{i,j}
    int block_row_top = block_pair_i * 2;      // Even row (0, 2, 4, ...)
    int block_row_bottom = block_row_top + 1;  // Odd row (1, 3, 5, ...)

    // WMMA fragments for 32x8x16 operations with half precision
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::row_major> b_frag;  // Fixed: should be row_major to match loading
    wmma::fragment<wmma::accumulator, 32, 8, 16, float> c_frag;

    // Shared memory for processing
    __shared__ half shared_a[32 * 16];     // Vertical pair A_{i,j}: 32x16 (half for WMMA)
    __shared__ half shared_b[16 * 8];      // Input slice B_j: 16x8 (half for WMMA)

    // Accumulator for final results
    __shared__ float block_results[32][8];  // [output_led][batch_frame]

    // Initialize accumulator
    if (lane_id < 32) {
        for (int b = 0; b < 8; b++) {
            block_results[lane_id][b] = 0.0f;
        }
    }
    __syncwarp();

    // Sum over j: iterate through all possible block columns
    for (int block_col_j = 0; block_col_j < led_blocks; block_col_j++) {

        // Find the top and bottom blocks for this vertical pair A_{i,j}
        const float* top_block_ptr = nullptr;
        const float* bottom_block_ptr = nullptr;
        bool use_transpose_top = false;
        bool use_transpose_bottom = false;

        // Check each diagonal to find blocks (block_row_top, block_col_j) and (block_row_bottom, block_col_j)
        for (int diag_idx = 0; diag_idx < block_diag_count; diag_idx++) {
            int offset = block_offsets[diag_idx];

            // Top block: (block_row_top, block_col_j)
            if (top_block_ptr == nullptr) {
                if (block_row_top + offset == block_col_j) {
                    // Found in upper diagonal
                    top_block_ptr = block_data + (channel_idx * block_diag_count + diag_idx) * 16 * 16;
                    use_transpose_top = false;
                } else if (block_col_j + offset == block_row_top && offset > 0) {
                    // Found in transpose position (symmetric matrix)
                    top_block_ptr = block_data + (channel_idx * block_diag_count + diag_idx) * 16 * 16;
                    use_transpose_top = true;
                }
            }

            // Bottom block: (block_row_bottom, block_col_j)
            if (bottom_block_ptr == nullptr && block_row_bottom < led_blocks) {
                if (block_row_bottom + offset == block_col_j) {
                    // Found in upper diagonal
                    bottom_block_ptr = block_data + (channel_idx * block_diag_count + diag_idx) * 16 * 16;
                    use_transpose_bottom = false;
                } else if (block_col_j + offset == block_row_bottom && offset > 0) {
                    // Found in transpose position (symmetric matrix)
                    bottom_block_ptr = block_data + (channel_idx * block_diag_count + diag_idx) * 16 * 16;
                    use_transpose_bottom = true;
                }
            }
        }

        // Skip if both blocks are empty (not found in any diagonal)
        if (top_block_ptr == nullptr && bottom_block_ptr == nullptr) {
            continue;
        }

        // Load input tensor slice B_j (16x8)
        int elements_per_thread = (16 * 8 + 31) / 32;
        for (int elem = 0; elem < elements_per_thread; elem++) {
            int idx = lane_id + elem * 32;
            if (idx < 16 * 8) {
                int row = idx / 8;      // LED index within 16x8 block (0-15)
                int col = idx % 8;      // Batch frame index (0-7)
                int led_idx = block_col_j * 16 + row;

                float value = 0.0f;
                if (led_idx < leds) {
                    // Load from input_batch[col, channel_idx, led_idx]
                    // Tensor layout is (batch_size, channels, leds) = (8, 3, leds)
                    value = input_batch[col * channels * leds + channel_idx * leds + led_idx];
                }
                shared_b[idx] = __float2half(value);
            }
        }
        __syncwarp();

        // Load vertical pair A_{i,j} (32x16)
        elements_per_thread = (32 * 16 + 31) / 32;
        for (int elem = 0; elem < elements_per_thread; elem++) {
            int idx = lane_id + elem * 32;
            if (idx < 32 * 16) {
                int row = idx / 16;
                int col = idx % 16;
                float value = 0.0f;

                if (row < 16) {
                    // Top block (rows 0-15)
                    if (top_block_ptr != nullptr) {
                        if (use_transpose_top) {
                            // Load transpose: A^T[row][col] = A[col][row]
                            value = top_block_ptr[col * 16 + row];
                        } else {
                            // Load normal: A[row][col]
                            value = top_block_ptr[row * 16 + col];
                        }
                    }
                } else {
                    // Bottom block (rows 16-31)
                    if (bottom_block_ptr != nullptr) {
                        int local_row = row - 16;
                        if (use_transpose_bottom) {
                            // Load transpose
                            value = bottom_block_ptr[col * 16 + local_row];
                        } else {
                            // Load normal
                            value = bottom_block_ptr[local_row * 16 + col];
                        }
                    }
                }
                shared_a[idx] = __float2half(value);
            }
        }
        __syncwarp();

        // Perform 32x8x16 WMMA operation: A_{i,j}[32x16] @ B_j[16x8] → C_i[32x8]
        wmma::fill_fragment(c_frag, 0.0f);
        wmma::load_matrix_sync(a_frag, shared_a, 16);       // A: 32x16, leading dim 16
        wmma::load_matrix_sync(b_frag, shared_b, 8);        // B: 16x8, leading dim 8
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Store and accumulate results
        __shared__ float temp_c[32 * 8];
        wmma::store_matrix_sync(temp_c, c_frag, 8, wmma::mem_row_major);  // C: 32x8, leading dim 8
        __syncwarp();

        // Accumulate into block_results
        if (lane_id < 32) {
            for (int batch_col = 0; batch_col < 8; batch_col++) {
                block_results[lane_id][batch_col] += temp_c[lane_id * 8 + batch_col];
            }
        }
        __syncwarp();
    }

    // Store final accumulated results into output
    if (lane_id < 32) {
        for (int batch_col = 0; batch_col < 8; batch_col++) {
            int led_idx = block_pair_i * 32 + lane_id;  // 32 LEDs per vertical pair
            if (led_idx < leds) {
                // Store to output_batch[batch_col, channel_idx, led_idx]
                output_batch[batch_col * channels * leds + channel_idx * leds + led_idx] =
                    block_results[lane_id][batch_col];
            }
        }
    }
}


} // extern "C"
