
#define BLOCK_SIZE 32

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

__kernel void gemm_update_kernel(__global float* A, const int k, const int n) {
    int local_row = get_local_id(1);
    int local_col = get_local_id(0);
    int group_row = get_group_id(1);
    int group_col = get_group_id(0);

    // Top-left corner of the destination tile this work-group is responsible for
    int dest_tile_row_start = k + BLOCK_SIZE + group_row * BLOCK_SIZE;
    int dest_tile_col_start = k + BLOCK_SIZE + group_col * BLOCK_SIZE;

    __local float L_tile[BLOCK_SIZE][BLOCK_SIZE];
    __local float U_tile[BLOCK_SIZE][BLOCK_SIZE];

    // Load the tile from the L panel (A_ik)
    L_tile[local_row][local_col] = A[(dest_tile_row_start + local_row) * n + (k + local_col)];

    // Load the tile from the U panel (A_kj)
    U_tile[local_row][local_col] = A[(k + local_row) * n + (dest_tile_col_start + local_col)];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Standard local memory GEMM computation
    float accumulator = 0.0f;
    for (int p = 0; p < BLOCK_SIZE; ++p) {
        accumulator += L_tile[local_row][p] * U_tile[p][local_col];
    }

    // Calculate the final global memory address to write to
    int final_global_row = dest_tile_row_start + local_row;
    int final_global_col = dest_tile_col_start + local_col;

    // Boundary check and write result
    if (final_global_row < n && final_global_col < n) {
        A[final_global_row * n + final_global_col] -= accumulator;
    }
}