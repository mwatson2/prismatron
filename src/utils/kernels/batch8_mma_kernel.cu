/*
 * 8-Frame Batch MMA Kernel for symmetric block diagonal matrix multiplication.
 *
 * Key features:
 * - Processes 8 frames simultaneously using standard 16x16x16 WMMA operations
 * - Uses block pair processing to handle 8-frame inputs efficiently
 * - Each block processes 8 frames at a time using standard WMMA shapes
 * - Input tensor (8, 3, leds) is processed in 8-frame chunks
 * - Uses multiple 16x16x16 operations to achieve 8-frame batch processing
 *
 * Processing strategy:
 * For each 16x16 block Ai:
 * 1. Load 16x16 block matrix
 * 2. Load 8 input vectors as 16x8 matrix (pad to 16x16 with zeros)
 * 3. WMMA: 16x16 × 16x16 → 16x16 result
 * 4. Extract 8 relevant result vectors from 16x16 output
 * 5. Accumulate results and handle symmetric contributions
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" {

__global__ void batch8_symmetric_block_pair_multiply_wmma(
    const float* block_data,      // Shape: (channels, block_diag_count, 16, 16)
    const int* block_offsets,     // Shape: (block_diag_count,)
    const float* input_batch,     // Shape: (8, channels, padded_leds)
    float* output_batch,          // Shape: (8, channels, padded_leds)
    int batch_size,               // Always 8
    int channels,                 // 3 (RGB)
    int led_blocks,              // Number of 16x16 blocks per dimension
    int block_diag_count,        // Number of diagonal bands
    int padded_leds              // LED count padded to 16-boundary
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

    // WMMA fragments for 16x16x16 operations (standard supported shape)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Shared memory for data conversion and intermediate results
    __shared__ half shared_a[16 * 16];    // Block data (16x16)
    __shared__ half shared_b[16 * 16];    // Input tensor slice padded to 16x16
    __shared__ float shared_c[16 * 16];   // Results (16x16)

    // Accumulator for results across all block diagonals for each batch item
    __shared__ float batch_results[16][8];  // [output_position][batch_item] - 16 positions, 8 batches

    // Initialize accumulator
    if (lane_id < 16) {
        for (int b = 0; b < 8; b++) {
            batch_results[lane_id][b] = 0.0f;
        }
    }
    __syncwarp();

    // Process each block diagonal (simplified approach using standard 16x16x16 WMMA)
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

            // Load 8 input vectors as Matrix B (16x16) - pad with zeros for columns 8-15
            if (lane_id < 16) {
                for (int batch_col = 0; batch_col < 16; batch_col++) {
                    int vector_element = lane_id;
                    int led_idx = block_col * 16 + vector_element;
                    float input_val = 0.0f;

                    if (batch_col < 8 && led_idx < padded_leds) {
                        // Use actual input for first 8 columns
                        input_val = input_batch[batch_col * channels * padded_leds +
                                              channel_idx * padded_leds + led_idx];
                    }
                    // Columns 8-15 remain zero for padding

                    // Store in column-major order
                    shared_b[batch_col * 16 + vector_element] = __float2half(input_val);
                }
            }
            __syncwarp();

            wmma::load_matrix_sync(a_frag, shared_a, 16);
            wmma::load_matrix_sync(b_frag, shared_b, 16);

            // Perform WMMA: C = A * B
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            // Store result matrix C (16x16) and accumulate
            wmma::store_matrix_sync(shared_c, c_frag, 16, wmma::mem_row_major);
            __syncwarp();

            // Accumulate results: only use first 8 columns of result (8 batch items)
            if (lane_id < 16) {
                for (int batch_col = 0; batch_col < 8; batch_col++) {
                    batch_results[lane_id][batch_col] += shared_c[lane_id * 16 + batch_col];
                }
            }
            __syncwarp();

            // Handle symmetric contribution for off-diagonal blocks
            if (block_offset > 0) {
                // Transpose contribution: result[block_col] += A^T * input[block_row]
                // Load input vectors from block_row position
                if (lane_id < 16) {
                    for (int batch_col = 0; batch_col < 16; batch_col++) {
                        int vector_element = lane_id;
                        int led_idx = block_row * 16 + vector_element;
                        float input_val = 0.0f;

                        if (batch_col < 8 && led_idx < padded_leds) {
                            // Use actual input for first 8 columns
                            input_val = input_batch[batch_col * channels * padded_leds +
                                                  channel_idx * padded_leds + led_idx];
                        }

                        shared_b[batch_col * 16 + vector_element] = __float2half(input_val);
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
                    for (int batch_col = 0; batch_col < 8; batch_col++) {
                        float sym_result = shared_c[lane_id * 16 + batch_col];

                        // Add symmetric contribution to block_col position
                        int sym_led_idx = block_col * 16 + lane_id;
                        if (sym_led_idx < padded_leds) {
                            atomicAdd(&output_batch[batch_col * channels * padded_leds +
                                                  channel_idx * padded_leds + sym_led_idx],
                                     sym_result);
                        }
                    }
                }
            }
        }
    }

    // Store final accumulated results for all batch items at this block_row
    if (lane_id < 16) {
        for (int batch_col = 0; batch_col < 8; batch_col++) {
            if (batch_col < batch_size) {
                int led_idx = block_row * 16 + lane_id;
                if (led_idx < padded_leds) {
                    atomicAdd(&output_batch[batch_col * channels * padded_leds +
                                          channel_idx * padded_leds + led_idx],
                             batch_results[lane_id][batch_col]);
                }
            }
        }
    }
}

__global__ void batch8_symmetric_block_pair_multiply_wmma_optimized(
    const float* block_data,
    const int* block_offsets,
    const float* input_batch,
    float* output_batch,
    int batch_size,
    int channels,
    int led_blocks,
    int block_diag_count,
    int padded_leds
) {
    // Optimized version with improved memory access patterns
    // Grid configuration: (channels, led_blocks)  
    // Block configuration: 32 threads (1 warp, same as basic but with optimizations)
    
    int channel_idx = blockIdx.x;
    int block_row = blockIdx.y;
    int lane_id = threadIdx.x % 32;

    // Early exit if out of bounds
    if (channel_idx >= channels || block_row >= led_blocks) {
        return;
    }

    // WMMA fragments for 16x16x16 operations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Shared memory with better organization
    __shared__ half shared_a[16 * 16];
    __shared__ half shared_b[16 * 16];
    __shared__ float shared_c[16 * 16];
    __shared__ float batch_results[16][8];

    // Initialize accumulator
    if (lane_id < 16) {
        for (int b = 0; b < 8; b++) {
            batch_results[lane_id][b] = 0.0f;
        }
    }
    __syncwarp();

    // Process diagonals with optimized memory access patterns
    for (int block_diag_idx = 0; block_diag_idx < block_diag_count; block_diag_idx++) {
        int block_offset = block_offsets[block_diag_idx];
        int block_col = block_row + block_offset;

        if (block_col < led_blocks) {
            wmma::fill_fragment(c_frag, 0.0f);

            // Optimized block matrix loading with coalesced access
            const float* block_ptr = block_data +
                (channel_idx * block_diag_count + block_diag_idx) * 16 * 16;

            // Parallel loading - each thread loads multiple elements
            int elements_per_thread = (16 * 16 + 31) / 32;  // Ceiling division
            for (int elem = 0; elem < elements_per_thread; elem++) {
                int idx = lane_id + elem * 32;
                if (idx < 16 * 16) {
                    shared_a[idx] = __float2half(block_ptr[idx]);
                }
            }
            __syncwarp();

            // Optimized input loading with coalesced access
            for (int elem = 0; elem < elements_per_thread; elem++) {
                int idx = lane_id + elem * 32;
                if (idx < 16 * 16) {
                    int row = idx / 16;
                    int col = idx % 16;
                    int led_idx = block_col * 16 + row;
                    float input_val = 0.0f;

                    if (col < 8 && led_idx < padded_leds) {
                        // Coalesced memory access pattern
                        input_val = input_batch[col * channels * padded_leds +
                                              channel_idx * padded_leds + led_idx];
                    }
                    // Store in column-major order (like basic kernel)
                    shared_b[col * 16 + row] = __float2half(input_val);
                }
            }
            __syncwarp();

            // WMMA computation
            wmma::load_matrix_sync(a_frag, shared_a, 16);
            wmma::load_matrix_sync(b_frag, shared_b, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            // Store and accumulate results
            wmma::store_matrix_sync(shared_c, c_frag, 16, wmma::mem_row_major);
            __syncwarp();

            // Accumulate with better memory access pattern
            if (lane_id < 16) {
                for (int batch_col = 0; batch_col < 8; batch_col++) {
                    batch_results[lane_id][batch_col] += shared_c[lane_id * 16 + batch_col];
                }
            }
            __syncwarp();

            // Optimized symmetric contribution handling
            if (block_offset > 0) {
                // Load symmetric input vectors
                for (int elem = 0; elem < elements_per_thread; elem++) {
                    int idx = lane_id + elem * 32;
                    if (idx < 16 * 16) {
                        int row = idx / 16;
                        int col = idx % 16;
                        int led_idx = block_row * 16 + row;
                        float input_val = 0.0f;

                        if (col < 8 && led_idx < padded_leds) {
                            input_val = input_batch[col * channels * padded_leds +
                                                  channel_idx * padded_leds + led_idx];
                        }
                        // Store in column-major order (like basic kernel)
                        shared_b[col * 16 + row] = __float2half(input_val);
                    }
                }
                __syncwarp();

                // Load transposed matrix with coalesced access
                for (int elem = 0; elem < elements_per_thread; elem++) {
                    int idx = lane_id + elem * 32;
                    if (idx < 16 * 16) {
                        int row = idx / 16;
                        int col = idx % 16;
                        shared_a[idx] = __float2half(block_ptr[col * 16 + row]);
                    }
                }
                __syncwarp();

                // Symmetric WMMA computation
                wmma::fill_fragment(c_frag, 0.0f);
                wmma::load_matrix_sync(a_frag, shared_a, 16);
                wmma::load_matrix_sync(b_frag, shared_b, 16);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

                wmma::store_matrix_sync(shared_c, c_frag, 16, wmma::mem_row_major);
                __syncwarp();

                // Store symmetric results with optimized atomic operations
                if (lane_id < 16) {
                    for (int batch_col = 0; batch_col < 8; batch_col++) {
                        float sym_result = shared_c[lane_id * 16 + batch_col];
                        int sym_led_idx = block_col * 16 + lane_id;
                        
                        if (sym_led_idx < padded_leds) {
                            atomicAdd(&output_batch[batch_col * channels * padded_leds +
                                                  channel_idx * padded_leds + sym_led_idx],
                                     sym_result);
                        }
                    }
                }
            }
        }
    }

    // Store final results with optimized memory access
    if (lane_id < 16) {
        for (int batch_col = 0; batch_col < 8; batch_col++) {
            if (batch_col < batch_size) {
                int led_idx = block_row * 16 + lane_id;
                if (led_idx < padded_leds) {
                    atomicAdd(&output_batch[batch_col * channels * padded_leds +
                                          channel_idx * padded_leds + led_idx],
                             batch_results[lane_id][batch_col]);
                }
            }
        }
    }
}

} // extern "C"