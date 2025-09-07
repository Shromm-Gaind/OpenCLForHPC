// Save this as kernels/my_gemm_kernel.cl

#define BLOCK_SIZE 32
#define TILE_DIM 4      // Each thread computes a 4x4 tile of C
#define WGS 8           // A work-group is 8x8 = 64 threads

__kernel void my_gemm_pure_register_blocked(const int M, const int N, const int K,
                                            __global const float* A,
                                            __global const float* B,
                                            __global float* C) {
    // --- 1. SETUP & INDEXING ---
    __global const float4* A_vec = (__global const float4*)A;
    __global const float4* B_vec = (__global const float4*)B;
    __global float4*       C_vec = (__global float4*)C;
    const int n_vec_A = K / 4;
    const int n_vec_B = N / 4;
    const int n_vec_C = N / 4;

    const int local_row = get_local_id(1); // [0..7]
    const int local_col = get_local_id(0); // [0..7]

    const int dest_tile_row_start = get_group_id(1) * BLOCK_SIZE;
    const int dest_tile_col_start_vec = get_group_id(0) * (BLOCK_SIZE / 4);

    float4 accum[TILE_DIM];
    for(int i = 0; i < TILE_DIM; ++i) {
        accum[i] = (float4)(0.0f);
    }

    __local float L_tile[BLOCK_SIZE][BLOCK_SIZE];
    __local float U_tile[BLOCK_SIZE][BLOCK_SIZE];

    // --- Outer loop to tile through the K dimension ---
    for (int tile_k = 0; tile_k < K; tile_k += BLOCK_SIZE) {

        // --- 2. COOPERATIVE LOAD FROM GLOBAL TO SHARED MEMORY ---
        const int l_panel_start_row = dest_tile_row_start;
        const int l_panel_start_col_vec = tile_k / 4;
        const int u_panel_start_row = tile_k;
        const int u_panel_start_col_vec = dest_tile_col_start_vec;

        for (int i = 0; i < TILE_DIM; i++) {
            float4 l_data = A_vec[(l_panel_start_row + local_row * TILE_DIM + i) * n_vec_A + l_panel_start_col_vec + local_col];
            float4 u_data = B_vec[(u_panel_start_row + local_row * TILE_DIM + i) * n_vec_B + u_panel_start_col_vec + local_col];

            *(__local float4*)&L_tile[local_row * TILE_DIM + i][local_col * 4] = l_data;
            *(__local float4*)&U_tile[local_row * TILE_DIM + i][local_col * 4] = u_data;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- 3. COMPUTATION FROM SHARED MEMORY ---
        #pragma unroll
        for (int p = 0; p < BLOCK_SIZE; ++p) {
            float4 u_vec = *(__local float4*)&U_tile[p][local_col * TILE_DIM];
            float l_vals[TILE_DIM];
            for(int i=0; i<TILE_DIM; ++i) {
                l_vals[i] = L_tile[local_row * TILE_DIM + i][p];
            }
            for(int i=0; i<TILE_DIM; ++i) {
                accum[i] = fma((float4)l_vals[i], u_vec, accum[i]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 4. WRITE RESULT FROM REGISTERS TO GLOBAL MEMORY ---
    const int final_global_row_start = dest_tile_row_start + local_row * TILE_DIM;
    const int final_global_col_vec = dest_tile_col_start_vec + local_col;

    for (int i = 0; i < TILE_DIM; i++) {
        if ((final_global_row_start + i < M) && (get_group_id(0) < (N / BLOCK_SIZE))) {
             C_vec[(final_global_row_start + i) * n_vec_C + final_global_col_vec] = accum[i];
        }
    }
}