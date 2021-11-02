#include <CL/cl.h>
#include <omp.h>
#include <iostream>

#include "Axpy.h"
#include "Utils.h"

void saxpy(int n, float a, float* x, int incx, float* y, int incy) {
	for (int i = 0; i < n; i++)
		y[i * incy] += a * x[i * incx];
}
void daxpy(int n, double a, double* x, int incx, double* y, int incy) {
	for (int i = 0; i < n; i++)
		y[i * incy] += a * x[i * incx];
}

double saxpy_gpu(int n, float a, float* x, int incx, float* y, int incy, cl_device_id deviceId, size_t local_work_size) {
	cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
	cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);
	cl_int ret = 0;

	std::string source = readFile("Saxpy.cl");
	const char* strings[] = { source.c_str() };
	cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
	clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
	cl_kernel kernel = clCreateKernel(program, "saxpy_gpu", nullptr);

	cl_mem x_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * incx * sizeof(float), nullptr, nullptr);
	clEnqueueWriteBuffer(queue, x_mem, CL_TRUE, 0, n * incx * sizeof(float), x, 0, nullptr, nullptr);
	cl_mem y_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, n * incy * sizeof(float), nullptr, nullptr);
	clEnqueueWriteBuffer(queue, y_mem, CL_TRUE, 0, n * incy * sizeof(float), y, 0, nullptr, nullptr);

	ret = clSetKernelArg(kernel, 0, sizeof(int), &n);
	ret = clSetKernelArg(kernel, 1, sizeof(float), &a);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_mem);
	ret = clSetKernelArg(kernel, 3, sizeof(int), &incx);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_mem);
	ret = clSetKernelArg(kernel, 5, sizeof(int), &incy);

	size_t n_size = static_cast<size_t>(n);
	double t1 = omp_get_wtime();
	ret = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &n_size, &local_work_size, 0, nullptr, nullptr);
	ret = clFinish(queue);
	double t2 = omp_get_wtime();
	ret = clEnqueueReadBuffer(queue, y_mem, CL_TRUE, 0, n * incy * sizeof(float), y, 0, nullptr, nullptr);

	clReleaseMemObject(x_mem);
	clReleaseMemObject(y_mem);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);

	return t2 - t1;
}

double daxpy_gpu(int n, double a, double* x, int incx, double* y, int incy, cl_device_id deviceId, size_t local_work_size) {
	cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
	cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);
	cl_int ret = 0;

	std::string source = readFile("Daxpy.cl");
	const char* strings[] = { source.c_str() };
	cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
	clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
	cl_kernel kernel = clCreateKernel(program, "daxpy_gpu", nullptr);

	cl_mem x_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * incx * sizeof(double), nullptr, nullptr);
	ret = clEnqueueWriteBuffer(queue, x_mem, CL_TRUE, 0, n * incx * sizeof(double), x, 0, nullptr, nullptr);
	cl_mem y_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, n * incy * sizeof(double), nullptr, nullptr);
	ret = clEnqueueWriteBuffer(queue, y_mem, CL_TRUE, 0, n * incy * sizeof(double), y, 0, nullptr, nullptr);

	ret = clSetKernelArg(kernel, 0, sizeof(int), &n);
	ret = clSetKernelArg(kernel, 1, sizeof(double), &a);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_mem);
	ret = clSetKernelArg(kernel, 3, sizeof(int), &incx);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_mem);
	ret = clSetKernelArg(kernel, 5, sizeof(int), &incy);

	size_t n_size = static_cast<size_t>(n);
	double t1 = omp_get_wtime();
	ret = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &n_size, nullptr, 0, nullptr, nullptr);
	ret = clFinish(queue);
	double t2 = omp_get_wtime();
	ret = clEnqueueReadBuffer(queue, y_mem, CL_TRUE, 0, n * incy * sizeof(double), y, 0, nullptr, nullptr);

	clReleaseMemObject(x_mem);
	clReleaseMemObject(y_mem);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);

	return t2 - t1;
}

void saxpy_omp(int n, float a, float* x, int incx, float* y, int incy) {
#pragma omp parallel for
	for (int i = 0; i < n; i++)
		y[i * incy] += a * x[i * incx];
}
void daxpy_omp(int n, double a, double* x, int incx, double* y, int incy) {
#pragma omp parallel for
	for (int i = 0; i < n; i++)
		y[i * incy] += a * x[i * incx];
}
