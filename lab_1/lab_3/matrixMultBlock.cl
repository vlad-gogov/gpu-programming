#define BLOCK_SIZE 16

__kernel void matrixMultBlock(__global float *a, __global float *b, __global float *c, int m, int n, int k) {

    __local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    int blocks = n / BLOCK_SIZE;

    float sum = 0;
    for (int i = 0; i < blocks; i++) {
    A[local_i][local_j] = a[global_i * n + BLOCK_SIZE * i + local_j];
    B[local_i][local_j] = b[(BLOCK_SIZE * i + local_i) * k + global_j];
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int l = 0; l < BLOCK_SIZE; l++)
        sum += A[local_i][l] * B[l][local_j];
    barrier(CLK_GLOBAL_MEM_FENCE);
    }
    c[global_i * k + global_j] = sum;
}

__kernel void matrixMultBlockOpt(__global float *a, __global float *b, __global float *c, int m, int n, int k) {

    __local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    int blocks = n / BLOCK_SIZE;

    float sum = 0;
    for (int i = 0; i < blocks; i++) {
    A[local_j][local_i] = a[global_j * m + BLOCK_SIZE * i + local_i];
    B[local_j][local_i] = b[(BLOCK_SIZE * i + local_j) * n + global_i];
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int l = 0; l < BLOCK_SIZE; l++)
        sum += A[local_j][l] * B[l][local_i];
    barrier(CLK_GLOBAL_MEM_FENCE);
    }
    c[global_j * k + global_i] = sum;
}

__kernel void matrixMultBlockImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, int m, int n, int k) {

    __local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    int blocks = n / BLOCK_SIZE;

    float sum = 0;
    for (int i = 0; i < blocks; i++) {
    float x = read_imagef(a, (int2)(BLOCK_SIZE * i + local_i, global_j)).x;
    float y = read_imagef(b, (int2)(global_i, BLOCK_SIZE * i + local_j)).x;
    A[local_j][local_i] = x;
    B[local_j][local_i] = y;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = 0; l < BLOCK_SIZE; l++)
        sum += A[local_j][l] * B[l][local_i];
    barrier(CLK_GLOBAL_MEM_FENCE);
    }
    write_imagef(c, (int2)(global_i, global_j), sum);
}