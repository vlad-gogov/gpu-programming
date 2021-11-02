#pragma once

void saxpy(int n, float a, float* x, int incx, float* y, int incy);
void daxpy(int n, double a, double* x, int incx, double* y, int incy);

double saxpy_gpu(int n, float a, float* x, int incx, float* y, int incy, cl_device_id deviceId, size_t local_work_size);
double daxpy_gpu(int n, double a, double* x, int incx, double* y, int incy, cl_device_id deviceId, size_t local_work_size);

void saxpy_omp(int n, float a, float* x, int incx, float* y, int incy);
void daxpy_omp(int n, double a, double* x, int incx, double* y, int incy);