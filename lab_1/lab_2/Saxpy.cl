__kernel void saxpy_gpu(int n, float a, __global float *x, int incx, __global float *y, int incy)
{
	int i = get_global_id(0);
	y[i * incy] += a * x[i * incx];
}