// Save this as kernels/my_gemm_improved.cl

// --- TUNABLE PARAMETERS (passed in by the host during compilation) ---
// e.g., clBuildProgram(..., "-D MWG=32 -D NWG=32 -D KWG=32 -D MWI=4 -D NWI=4 -D WGS_X=8 -D WGS_Y=8")

// Save this as kernels/my_gemm_corrected.cl

// --- TUNABLE PARAMETERS ---
#ifndef MWG
#define MWG 128
#endif
#ifndef NWG
#define NWG 128
#endif
#ifndef KWG
#define KWG 16
#endif
#ifndef MWI
#define MWI 8
#endif
#ifndef NWI
#define NWI 8
#endif
#ifndef WGS_X
#define WGS_X 16 // Should be NWG / NWI
#endif
#ifndef WGS_Y
#define WGS_Y 16 // Should be MWG / MWI
#endif
#define PADDING 0 // Padding wasn't the issue, so it's removed for clarity

__kernel __attribute__((reqd_work_group_size(WGS_X, WGS_Y, 1)))
void my_gemm_improved(const int M, const int N, const int K,
                       const __global float* restrict A,
                       const __global float* restrict B, // IMPORTANT: Assumed to be transposed
                       __global float* restrict C) {

    // --- 1. INDEXING ---
    const int local_row = get_local_id(1); // [0..WGS_Y-1]
    const int local_col = get_local_id(0); // [0..WGS_X-1]
    const int group_start_row = get_group_id(1) * MWG;
    const int group_start_col = get_group_id(0) * NWG;
    const int tid = local_row * WGS_X + local_col;
    const int num_threads = WGS_X * WGS_Y;

    // --- 2. ALLOCATIONS ---
    float accum[MWI][NWI] = {{0.0f}};
    __local float a_tile[KWG][MWG];
    __local float b_tile[KWG][NWG];

    // --- 3. K-LOOP ---
    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += KWG) {

        // --- 3a. CORRECTED GLOBAL TO SHARED MEMORY LOAD ---
        // This pattern uses a single flat thread ID to correctly load tiles.
        // It's simple, robust, and performs well due to coalesced access patterns.
        for (int load_idx = tid; load_idx < KWG * MWG; load_idx += num_threads) {
            int k = load_idx / MWG;
            int m = load_idx % MWG;
            int load_row = group_start_row + m;
            int load_col = k_tile_start + k;
            if (load_row < M && load_col < K) {
                a_tile[k][m] = A[load_row * K + load_col];
            } else {
                a_tile[k][m] = 0.0f;
            }
        }
        for (int load_idx = tid; load_idx < KWG * NWG; load_idx += num_threads) {
            int k = load_idx / NWG;
            int n = load_idx % NWG;
            int load_row = group_start_col + n; // row in B_T is column in C
            int load_col = k_tile_start + k;    // col in B_T is K dimension
            if (load_row < N && load_col < K) {
                b_tile[k][n] = B[load_row * K + load_col]; // B is transposed: (N, K)
            } else {
                b_tile[k][n] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // --- 3b. COMPUTATION (Correct) ---
        // This part was algorithmically correct and remains.
#pragma unroll
        for (int p = 0; p < KWG; ++p) {
            float a_reg[MWI];
            float b_reg[NWI];
            for(int i=0; i<MWI; ++i) a_reg[i] = a_tile[p][local_row * MWI + i];
            for(int i=0; i<NWI; ++i) b_reg[i] = b_tile[p][local_col * NWI + i];

            for (int m = 0; m < MWI; ++m) {
                for (int n = 0; n < NWI; ++n) {
                    accum[m][n] = fma(a_reg[m], b_reg[n], accum[m][n]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 4. WRITE RESULT (Correct) ---
    for (int m = 0; m < MWI; ++m) {
        for (int n = 0; n < NWI; ++n) {
            int final_global_row = group_start_row + local_row * MWI + m;
            int final_global_col = group_start_col + local_col * NWI + n;
            if (final_global_row < M && final_global_col < N) {
                C[final_global_row * N + final_global_col] = accum[m][n];
            }
        }
    }
}