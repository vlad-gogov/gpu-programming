#include <CL/cl.h>
#include <omp.h>

#include <iostream>
#include <vector>

#include "Axpy.h"

// 0 - Intel(R) OpenCL HD Graphics
// 1 - OpenCL CUDA
// 2 - Intel(R) OpenCL

int main() {
	cl_uint platformCount = 0;
	clGetPlatformIDs(0, nullptr, &platformCount);
	cl_platform_id* platform = new cl_platform_id[platformCount];
	clGetPlatformIDs(platformCount, platform, nullptr);
	cl_uint deviceCount = 0;
	cl_device_id deviceId = 0;

	constexpr int n = 9'999'872;
	constexpr int incx = 3;
	constexpr int incy = 2;
	constexpr size_t size_x = static_cast<size_t>(n * incx);
	constexpr size_t size_y = static_cast<size_t>(n * incy);

	std::cout << "FLOAT " << std::endl;

	{
		float a = 3;
		clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &deviceCount);
		std::vector<float> x_float(size_x, 0);
		std::vector<float> y_float(size_y, 0);
		std::vector<float> y_float_result(size_y, 0);

		for (size_t i = 0; i < size_x; i += incx) {
			x_float[i] = 4.0f;
		}

		for (size_t i = 0; i < size_y; i += incy) {
			y_float[i] = 5.0f;
			y_float_result[i] = 5.0f;
		}
		double t1 = 0, t2 = 0, t = 0;

		std::vector<float> y_copy = y_float;
		/*
		t1 = omp_get_wtime();
		saxpy(n, a, x_float.data(), incx, y_copy.data(), incy);
		t2 = omp_get_wtime();
		std::cout << "Sequential time float: " << t2 - t1 << std::endl;
		std::cout << std::endl;
		y_copy = y_float;
		*/

		t1 = omp_get_wtime();
		saxpy(n, a, x_float.data(), incx, y_float_result.data(), incy);
		t2 = omp_get_wtime();
		std::cout << "Sequential time float: " << t2 - t1 << std::endl;
		std::cout << std::endl;

		for (size_t i = 8; i <= 256; i *= 2) {
			t = saxpy_gpu(n, a, x_float.data(), incx, y_copy.data(), incy, deviceId, i);
			std::cout << i << ".OpenCL GPU time float: " << t << ". ";
			if (y_copy == y_float_result)
				std::cout << "SUCCESS" << std::endl;
			y_copy = y_float;
		}
		std::cout << std::endl;

		clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &deviceCount);

		for (size_t i = 8; i <= 256; i *= 2) {
			t = saxpy_gpu(n, a, x_float.data(), incx, y_copy.data(), incy, deviceId, i);
			std::cout << i << ".OpenCL CPU time float: " << t << ". ";
			if (y_copy == y_float_result)
				std::cout << "SUCCESS" << std::endl;
			y_copy = y_float;
		}
		std::cout << std::endl;

		t1 = omp_get_wtime();
		saxpy_omp(n, a, x_float.data(), incx, y_copy.data(), incy);
		t2 = omp_get_wtime();
		std::cout << "OMP time float: " << t2 - t1 << ". ";
		if (y_copy == y_float_result)
			std::cout << "SUCCESS" << std::endl;
		y_copy = y_float;
	}

	std::cout << std::endl << "DOUBLE " << std::endl;

	{
		double a = 3;
		clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &deviceCount);
		std::vector<double> x_double(size_x, 0);
		std::vector<double> y_double(size_y, 0);
		std::vector<double> y_double_result(size_y, 0);

		for (size_t i = 0; i < size_x; i += incx) {
			x_double[i] = 4.0;
		}

		for (size_t i = 0; i < size_y; i += incy) {
			y_double[i] = 5.0;
			y_double_result[i] = 5.0;
		}
		double t1 = 0, t2 = 0, t = 0;

		t1 = omp_get_wtime();
		daxpy(n, a, x_double.data(), incx, y_double_result.data(), incy);
		t2 = omp_get_wtime();
		std::cout << "Sequential time double: " << t2 - t1 << std::endl;
		std::cout << std::endl;

		std::vector<double> y_copy = y_double;
		for (size_t i = 8; i <= 256; i *= 2) {
			t = daxpy_gpu(n, a, x_double.data(), incx, y_copy.data(), incy, deviceId, i);
			std::cout << i << ".OpenCL GPU time double: " << t << ". ";
			if (y_copy == y_double_result)
				std::cout << "SUCCESS" << std::endl;
			y_copy = y_double;
		}
		std::cout << std::endl;
		clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &deviceCount);

		for (size_t i = 8; i <= 256; i *= 2) {
			t = daxpy_gpu(n, a, x_double.data(), incx, y_copy.data(), incy, deviceId, 8);
			std::cout << i << ".OpenCL CPU time double: " << t << ". ";
			if (y_copy == y_double_result)
				std::cout << "SUCCESS" << std::endl;
			y_copy = y_double;
		}
		std::cout << std::endl;

		t1 = omp_get_wtime();
		daxpy_omp(n, a, x_double.data(), incx, y_copy.data(), incy);
		t2 = omp_get_wtime();
		std::cout << "OMP time doulbe: " << t2 - t1 << ". ";
		if (y_copy == y_double_result)
			std::cout << "SUCCESS" << std::endl;
		y_copy = y_double;
	}

	delete[] platform;
}
