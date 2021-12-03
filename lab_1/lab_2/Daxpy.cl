#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void daxpy_gpu(int n, double a, __global double *x, int incx, __global double *y, int incy)
{
	int i = get_global_id(0);
	y[i * incy] += a * x[i * incx];
}