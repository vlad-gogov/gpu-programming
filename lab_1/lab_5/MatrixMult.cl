#define BLOCK_SIZE 16

__kernel void matrixMultBlockOpt(__global float *a, __global float *b, __global float *c, int n) {

    __local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];

    int local_i = get_local_id(1);
    int local_j = get_local_id(0);

    int global_i = get_global_id(1);
    int global_j = get_global_id(0);

    int blocks = n / BLOCK_SIZE;

    float sum = 0;
    for (int i = 0; i < blocks; i++) {
        A[local_i][local_j] = a[global_i * n + BLOCK_SIZE * i + local_j];
        B[local_i][local_j] = b[(BLOCK_SIZE * i + local_i) * n + global_j];
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (int l = 0; l < BLOCK_SIZE; l++)
            sum += A[local_i][l] * B[l][local_j];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    c[global_i * n + global_j] = sum;
}