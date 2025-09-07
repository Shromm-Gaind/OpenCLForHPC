//created by Shromm

#define BLOCK_SIZE 32
#define VEC_SIZE 16
#define TILE_DIM 4      // The dimension of the 2D tile computed by a single work-item.
#define WGS 8           // The Work-Group Size dimension (BLOCK_SIZE / TILE_DIM), which is 8.
#define TILE_DIM_16 2 // new to the larger threads per workgroup

__kernel void lu_row_update_kernel(__global float* A, const int k, const int n) {
    __local float L_panel[BLOCK_SIZE][BLOCK_SIZE];

    int local_id = get_local_id(0);

    // Cooperative load of the L_panel. Every work-item in a group helps.
    if (local_id < BLOCK_SIZE) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            L_panel[local_id][j] = A[(k + local_id) * n + (k + j)];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each work-item is responsible for one column of the output panel.
    int global_j = k + BLOCK_SIZE + get_global_id(0);
    if (global_j >= n) return;

    // Load column into private memory
    float x[BLOCK_SIZE];
    for(int i = 0; i < BLOCK_SIZE; i++) {
        x[i] = A[(k + i) * n + global_j];
    }

    // Perform forward substitution on private data
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float sum = 0.0f;
        for (int p = 0; p < i; p++) {
            sum += L_panel[i][p] * x[p];
        }
        // Unit diagonal for L is assumed
        x[i] = x[i] - sum;
    }

    // Write result back to global memory
    for(int i = 0; i < BLOCK_SIZE; i++) {
        A[(k + i) * n + global_j] = x[i];
    }
}

__kernel void lu_col_update_kernel(__global float* A, const int k, const int n) {
    __local float U_panel[BLOCK_SIZE][BLOCK_SIZE];

    int local_id = get_local_id(0);

    // Cooperative load of the U_panel.
    if (local_id < BLOCK_SIZE) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            U_panel[local_id][j] = A[(k + local_id) * n + (k + j)];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each work-item is responsible for one row of the output panel.
    int global_i = k + BLOCK_SIZE + get_global_id(0);
    if (global_i >= n) return;

    // Load row into private memory.
    float x[BLOCK_SIZE];
    for (int j = 0; j < BLOCK_SIZE; j++) {
        x[j] = A[global_i * n + (k + j)];
    }

    // Perform row-wise forward substitution to solve X * U = B
    for (int j = 0; j < BLOCK_SIZE; j++) {
        float sum = 0.0f;
        for (int p = 0; p < j; p++) {
            sum += x[p] * U_panel[p][j];
        }
        float pivot = U_panel[j][j];
        if (fabs(pivot) < 1e-9f) pivot = 1e-9f; // Safety check
        x[j] = (x[j] - sum) / pivot;
    }

    // Write result back to global memory.
    for (int j = 0; j < BLOCK_SIZE; j++) {
        A[global_i * n + (k + j)] = x[j];
    }
}

// This kernel combines three core optimization strategies for maximum performance:
//
// 1. TILING (SHARED MEMORY):
//    - A 32x32 block of the L panel (A_ik) and a 32x32 block of the U panel (A_kj) are
//      loaded into fast on-chip shared memory. This drastically reduces slow global
//      memory access during computation.
//
// 2. SQUARE WORK-GROUPS:
//    - This kernel is designed to be launched with a local work-group of (8, 8).
//    - Square work-groups are often more efficient for 2D problems like matrix multiplication.
//
// 3. REGISTER BLOCKING (The "MNN" Technique):
//    - Each of the 64 work-items in the work-group is responsible for computing
//      its own private 4x4 tile of the final 32x32 output block.
//    - These 16 float values are stored in 4 `float4` registers (`accum`).
//    - This technique maximizes the reuse of data loaded from shared memory, dramatically
//      increasing the arithmetic intensity (math vs. memory operations).

__kernel void gemm_update_kernel_register_blocked(__global float* A, const int k, const int n) {

    // --- 1. SETUP & INDEXING ---

    // Pointers for vectorized access (4 floats at a time)
    __global float4* A_vec = (__global float4*)A;
    const int n_vec = n / 4;

    // Local thread IDs within the 8x8 work-group
    const int local_row = get_local_id(1); // [0..7]
    const int local_col = get_local_id(0); // [0..7]

    // Top-left corner of the 32x32 destination tile this WORK-GROUP is responsible for
    const int dest_tile_row_start = k + BLOCK_SIZE + get_group_id(1) * BLOCK_SIZE;
    const int dest_tile_col_start_vec = (k + BLOCK_SIZE) / 4 + get_group_id(0) * (BLOCK_SIZE / 4);

    // Private per-thread accumulators for the 4x4 tile. This is register blocking.
    float4 accum[TILE_DIM];
    for(int i = 0; i < TILE_DIM; ++i) {
        accum[i] = (float4)(0.0f);
    }

    // Shared memory tiles
    __local float L_tile[BLOCK_SIZE][BLOCK_SIZE];
    __local float U_tile[BLOCK_SIZE][BLOCK_SIZE];


    // --- 2. COOPERATIVE LOAD FROM GLOBAL TO SHARED MEMORY ---
    // Each of the 64 threads loads a 4x4 patch of data from the global L and U panels
    // into the 32x32 shared memory tiles.

    const int l_panel_start_row = dest_tile_row_start;
    const int l_panel_start_col_vec = k / 4;
    const int u_panel_start_row = k;
    const int u_panel_start_col_vec = dest_tile_col_start_vec;

    for (int i = 0; i < TILE_DIM; i++) {
        // Each thread loads one float4 vector (4 floats) for its part of the L and U tiles
        float4 l_data = A_vec[(l_panel_start_row + local_row * TILE_DIM + i) * n_vec + l_panel_start_col_vec + local_col];
        float4 u_data = A_vec[(u_panel_start_row + local_row * TILE_DIM + i) * n_vec + u_panel_start_col_vec + local_col];

        // Store the loaded vectors into the correct place in shared memory
        *(__local float4*)&L_tile[local_row * TILE_DIM + i][local_col * 4] = l_data;
        *(__local float4*)&U_tile[local_row * TILE_DIM + i][local_col * 4] = u_data;
    }

    // Synchronize to ensure all data is loaded before computation begins.
    barrier(CLK_LOCAL_MEM_FENCE);


    // --- 3. COMPUTATION FROM SHARED MEMORY ---
    // This is the most performance-critical loop.
#pragma unroll
    for (int p = 0; p < BLOCK_SIZE; ++p) {

        // Fetch one column vector (4 floats) from the U tile.
        // All threads in the same local_row will fetch the same u_vec, but from different columns.
        float4 u_vec = *(__local float4*)&U_tile[p][local_col * TILE_DIM];

        // Fetch 4 separate scalar values from the L tile for the 4 rows this thread handles.
        // All threads in the same local_col will fetch the same l_vals.
        float l_vals[TILE_DIM];
        for(int i=0; i<TILE_DIM; ++i) {
            l_vals[i] = L_tile[local_row * TILE_DIM + i][p];
        }

        // Perform the dot product. The u_vec is REUSED 4 times here,
        // which is the entire point of register blocking.
        for(int i=0; i<TILE_DIM; ++i) {
            accum[i] = fma((float4)l_vals[i], u_vec, accum[i]);
        }
    }


    // --- 4. WRITE RESULT FROM REGISTERS TO GLOBAL MEMORY ---

    const int final_global_row_start = dest_tile_row_start + local_row * TILE_DIM;
    const int final_global_col_vec = dest_tile_col_start_vec + local_col;

    // Each thread writes its 4x4 tile (as 4 float4 vectors) back to the global matrix.
    // Boundary checks prevent writing outside the matrix on the final "ragged" edge tiles.
    for (int i = 0; i < TILE_DIM; i++) {
        if ((final_global_row_start + i < n) && (get_global_id(0) < get_num_groups(0) * WGS)) {
            A_vec[(final_global_row_start + i) * n_vec + final_global_col_vec] -= accum[i];
        }
    }
}
