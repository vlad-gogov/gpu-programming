__kernel void matrixMult(__global float *a, __global float *b, __global float *c, int m, int n, int k) {
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[global_j * n + i] * b[global_i + n * i];
    c[global_j * k + global_i] = sum;
}