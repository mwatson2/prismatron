/*
 * Corrected Batch MMA Kernel for symmetric block diagonal matrix multiplication.
 *
 * For single 16x16 block case:
 * - We want: result[batch_i] = A_matrix * input_vector[batch_i] for each batch_i
 * - WMMA computes: C = A * B where C[i,j] = sum_k(A[i,k] * B[k,j])
 * - Solution: Make B matrix where each column j is input_vector[batch_j]
 * - Then C[i,j] = sum_k(A[i,k] * input[batch_j,k]) = (A * input[batch_j])[i]
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" {

__global__ void batch_symmetric_block_dia_multiply_wmma(
    const float* block_data,      // Shape: (channels, block_diag_count, 16, 16)
    const int* block_offsets,     // Shape: (block_diag_count,)
    const float* input_batch,     // Shape: (batch_size, channels, padded_leds)
    float* output_batch,          // Shape: (batch_size, channels, padded_leds)
    int batch_size,
    int channels,
    int led_blocks,
    int block_diag_count,
    int padded_leds
) {
    // Grid configuration: (channels, led_blocks)
    // Block configuration: 32 threads (1 warp for WMMA)

    int channel_idx = blockIdx.x;
    int block_row = blockIdx.y;

    int lane_id = threadIdx.x % 32;

    // Early exit if out of bounds
    if (channel_idx >= channels || block_row >= led_blocks) {
        return;
    }

    // WMMA fragments for 16x16x16 operations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;  // col_major for input vectors as columns
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Shared memory for data conversion and intermediate results
    __shared__ half shared_a[16 * 16];
    __shared__ half shared_b[16 * 16];
    __shared__ float shared_c[16 * 16];

    // Accumulator for results across all block diagonals for each batch item
    __shared__ float batch_results[16][16];  // [output_position][batch_item]

    // Initialize accumulator
    if (lane_id < 16) {
        for (int b = 0; b < 16; b++) {
            batch_results[lane_id][b] = 0.0f;
        }
    }
    __syncwarp();

    // Process each block diagonal
    for (int block_diag_idx = 0; block_diag_idx < block_diag_count; block_diag_idx++) {
        int block_offset = block_offsets[block_diag_idx];
        int block_col = block_row + block_offset;

        if (block_col < led_blocks) {
            // Initialize WMMA accumulator
            wmma::fill_fragment(c_frag, 0.0f);

            // Load ATA block matrix A (16x16)
            const float* block_ptr = block_data +
                (channel_idx * block_diag_count + block_diag_idx) * 16 * 16;

            if (lane_id < 16) {
                for (int i = 0; i < 16; i++) {
                    shared_a[lane_id * 16 + i] = __float2half(block_ptr[lane_id * 16 + i]);
                }
            }
            __syncwarp();

            wmma::load_matrix_sync(a_frag, shared_a, 16);

            // Load batch input vectors as Matrix B (16x16) - CORRECT LAYOUT
            // Each COLUMN represents a different batch item's vector segment
            // B[vector_element][batch_idx] for proper A * B = C where C[i][j] = (A * input[j])[i]
            if (lane_id < 16) {
                for (int batch_col = 0; batch_col < 16; batch_col++) {
                    int batch_idx = batch_col;  // Column index maps to batch index
                    int vector_element = lane_id;  // Thread handles one vector element
                    int led_idx = block_col * 16 + vector_element;
                    float input_val = 0.0f;

                    if (batch_idx < batch_size && led_idx < padded_leds) {
                        input_val = input_batch[batch_idx * channels * padded_leds +
                                              channel_idx * padded_leds + led_idx];
                    }

                    // Store in column-major order: B[vector_element][batch_idx]
                    // Memory layout: shared_b[batch_col * 16 + vector_element] for col-major
                    shared_b[batch_col * 16 + vector_element] = __float2half(input_val);
                }
            }
            __syncwarp();

            wmma::load_matrix_sync(b_frag, shared_b, 16);

            // Perform WMMA: C = A * B
            // C[i,j] = sum_k(A[i,k] * B[k,j]) = sum_k(A[i,k] * input[batch_j,k])
            // This gives us C[i,j] = (A * input_vector[batch_j])[i]
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            // Store result matrix C (16x16) and accumulate
            wmma::store_matrix_sync(shared_c, c_frag, 16, wmma::mem_row_major);
            __syncwarp();

            // Accumulate results: C[output_pos][batch_idx] contains result for batch_idx at output_pos
            if (lane_id < 16) {
                for (int batch_col = 0; batch_col < 16; batch_col++) {
                    if (batch_col < batch_size) {
                        batch_results[lane_id][batch_col] += shared_c[lane_id * 16 + batch_col];
                    }
                }
            }

            // Handle symmetric contribution for off-diagonal blocks
            if (block_offset > 0) {
                // Symmetric contribution: result[block_col] += A^T * input[block_row]

                // Load input vectors from block_row position for all batch items
                if (lane_id < 16) {
                    for (int batch_col = 0; batch_col < 16; batch_col++) {
                        int batch_idx = batch_col;
                        int vector_element = lane_id;
                        int led_idx = block_row * 16 + vector_element;  // From block_row position
                        float sym_input_val = 0.0f;

                        if (batch_idx < batch_size && led_idx < padded_leds) {
                            sym_input_val = input_batch[batch_idx * channels * padded_leds +
                                                      channel_idx * padded_leds + led_idx];
                        }

                        // Store in column-major order for Matrix B
                        shared_b[batch_col * 16 + vector_element] = __float2half(sym_input_val);
                    }
                }
                __syncwarp();

                // Load transposed ATA block (A^T)
                if (lane_id < 16) {
                    for (int i = 0; i < 16; i++) {
                        // Transpose: A^T[lane_id][i] = A[i][lane_id]
                        shared_a[lane_id * 16 + i] = __float2half(block_ptr[i * 16 + lane_id]);
                    }
                }
                __syncwarp();

                // Compute A^T * input_vectors
                wmma::fill_fragment(c_frag, 0.0f);
                wmma::load_matrix_sync(a_frag, shared_a, 16);
                wmma::load_matrix_sync(b_frag, shared_b, 16);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

                // Store symmetric results directly to output
                wmma::store_matrix_sync(shared_c, c_frag, 16, wmma::mem_row_major);
                __syncwarp();

                if (lane_id < 16) {
                    for (int batch_col = 0; batch_col < 16; batch_col++) {
                        int batch_idx = batch_col;
                        if (batch_idx < batch_size) {
                            float sym_result = shared_c[lane_id * 16 + batch_col];

                            // Add symmetric contribution to block_col position
                            int sym_led_idx = block_col * 16 + lane_id;
                            if (sym_led_idx < padded_leds) {
                                atomicAdd(&output_batch[batch_idx * channels * padded_leds +
                                                      channel_idx * padded_leds + sym_led_idx],
                                         sym_result);
                            }
                        }
                    }
                }
            }
        }
    }

    // Store final accumulated results for all batch items at this block_row
    if (lane_id < 16) {
        for (int batch_col = 0; batch_col < 16; batch_col++) {
            int batch_idx = batch_col;
            if (batch_idx < batch_size) {
                int led_idx = block_row * 16 + lane_id;
                if (led_idx < padded_leds) {
                    atomicAdd(&output_batch[batch_idx * channels * padded_leds +
                                          channel_idx * padded_leds + led_idx],
                             batch_results[lane_id][batch_col]);
                }
            }
        }
    }
}


} // extern "C"
