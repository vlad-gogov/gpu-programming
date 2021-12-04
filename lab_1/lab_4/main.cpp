#include <CL/cl.h>
#include <omp.h>

#include <iostream>
#include <vector>

#include "jacobi.h"
#include "Utils.h"

// 0 - Intel(R) OpenCL HD Graphics
// 1 - OpenCL CUDA
// 2 - Intel(R) OpenCL

int main() {
	cl_uint platformCount = 0;
	clGetPlatformIDs(0, nullptr, &platformCount);
	cl_platform_id* platform = new cl_platform_id[platformCount];
	clGetPlatformIDs(platformCount, platform, nullptr);


	cl_uint deviceCount = 0;
	cl_device_id deviceIntelGPU = 0, deviceNvidiaGPU = 0, deviceIntelCPU = 0;
	char deviceName[128] = { 0 };

	clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_DEFAULT, 1, &deviceIntelCPU, &deviceCount);
	clGetDeviceInfo(deviceIntelCPU, CL_DEVICE_NAME, 128, deviceName, nullptr);
	std::cout << "CPU: " << deviceName << std::endl;

	std::cout << "GPU:" << std::endl;
	clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_DEFAULT, 1, &deviceIntelGPU, &deviceCount);
	clGetDeviceInfo(deviceIntelGPU, CL_DEVICE_NAME, 128, deviceName, nullptr);
	std::cout << "-" << deviceName << std::endl;

	clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_DEFAULT, 1, &deviceNvidiaGPU, &deviceCount);
	clGetDeviceInfo(deviceNvidiaGPU, CL_DEVICE_NAME, 128, deviceName, nullptr);
	std::cout << "-" << deviceName << std::endl;

	constexpr int n = 50 * 100;
	constexpr int count_iter = 1000;
	constexpr float epsilon = 1e-7;
	std::vector<float> A(n * n, 0);
	std::vector<float> b(n, 0);

	randomMatrix(A);
	randomMatrix(b);

	{
		std::random_device rd;
		std::mt19937 mersenne(rd());
		std::uniform_real_distribution<> urd(n * 5, n * 5 + 3.0);
		for (size_t i = 0; i < n; i++)
			A[i * n + i] = urd(mersenne);
	}

	std::cout << "Size matrix = " << n << ", Iterations = " << count_iter << ", e = " << epsilon << std::endl;

	std::cout << std::endl << "--------------------OpenCL CPU--------------------" << std::endl;

	{
		std::vector<float> x(n, 0);
		Results res = jacobi(A.data(), b.data(), x.data(), n, count_iter, epsilon, deviceIntelCPU);
		std::cout << "Full time: " << res.full_time << std::endl;
		std::cout << "Kernel time: " << res.kernel_time << std::endl;
		std::cout << "Count iter: " << res.count_iter << std::endl;
		std::cout << "Convergence: " << res.converg << std::endl;
		std::cout << "Achieved: " << accuracy(A.data(), b.data(), x.data(), n) << std::endl;
		std::cout << std::endl;
	}

	std::cout << std::endl << "--------------------OpenCL Nvidia GPU--------------------" << std::endl;

	{
		std::vector<float> x(n, 0);
		Results res = jacobi(A.data(), b.data(), x.data(), n, count_iter, epsilon, deviceNvidiaGPU);
		std::cout << "Full time: " << res.full_time << std::endl;
		std::cout << "Kernel time: " << res.kernel_time << std::endl;
		std::cout << "Count iter: " << res.count_iter << std::endl;
		std::cout << "Convergence: " << res.converg << std::endl;
		std::cout << "Achieved: " << accuracy(A.data(), b.data(), x.data(), n) << std::endl;
		std::cout << std::endl;
	}


	std::cout << std::endl << "--------------------OpenCL Intel GPU--------------------" << std::endl;

	{
		std::vector<float> x(n, 0);
		Results res = jacobi(A.data(), b.data(), x.data(), n, count_iter, epsilon, deviceIntelGPU);
		std::cout << "Full time: " << res.full_time << std::endl;
		std::cout << "Kernel time: " << res.kernel_time << std::endl;
		std::cout << "Count iter: " << res.count_iter << std::endl;
		std::cout << "Convergence: " << res.converg << std::endl;
		std::cout << "Achieved: " << accuracy(A.data(), b.data(), x.data(), n) << std::endl;
		std::cout << std::endl;
	}


	delete[] platform;
}


//t1 = omp_get_wtime();
//
//t2 = omp_get_wtime();
//std::cout << "Sequential time float: " << t2 - t1 << std::endl;
//std::cout << std::endl;


//for (size_t i = 0; i < 10; i++)
//	std::cout << C[i] << ", ";
//std::cout << std::endl;