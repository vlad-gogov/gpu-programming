#include <CL/cl.h>
#include <omp.h>

#include <iostream>
#include <vector>

#include "Mult.h"
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

	constexpr int m = 16 * 100;
	constexpr int n = 16 * 100;
	constexpr int k = 16 * 100;
	std::vector<float> A(m * n);
	std::vector<float> B(n * k);
	std::vector<float> cResult(m * k, 0);

	randomMatrix(A);
	randomMatrix(B);

#pragma omp parallel
	{
#pragma omp single
		std::cout << "Num threads: " << omp_get_num_threads() << std::endl;
	}

	{
		double t1 = omp_get_wtime();
		multMatrix(A.data(), B.data(), cResult.data(), m, n, k);
		double t2 = omp_get_wtime();
		std::cout << "Sequential: " << (t2 - t1) << std::endl;
	}

	{
		std::vector<float> C(m * k);
		double t1 = omp_get_wtime();
		multMatrixOmp(A.data(), B.data(), C.data(), m, n, k);
		double t2 = omp_get_wtime();
		std::cout << "OpenMP: " << (t2 - t1) << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	std::cout << std::endl << "--------------------OpenCL CPU--------------------" << std::endl;

	{
		std::vector<float> C(m * k);
		double time = multMatrix(A.data(), B.data(), C.data(), m, n, k, deviceIntelCPU);
		std::cout << "Matrix mult: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	{
		std::vector<float> C(m * k);
		double time = multMatrixBlock(A.data(), B.data(), C.data(), m, n, k, deviceIntelCPU);
		std::cout << "Matrix mult block: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	{
		std::vector<float> C(m * k);
		double time = multMatrixBlockOpt(A.data(), B.data(), C.data(), m, n, k, deviceIntelCPU);
		std::cout << "Matrix mult block opt: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	{
		std::vector<float> C(m * k);
		double time = multMatrixBlockImage(A.data(), B.data(), C.data(), m, n, k, deviceIntelCPU);
		std::cout << "Matrix mult block image: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	std::cout << std::endl << "--------------------OpenCL Nvidia GPU--------------------" << std::endl;

	{
		std::vector<float> C(m * k);
		double time = multMatrix(A.data(), B.data(), C.data(), m, n, k, deviceNvidiaGPU);
		std::cout << "Matrix mult: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	{
		std::vector<float> C(m * k);
		double time = multMatrixBlock(A.data(), B.data(), C.data(), m, n, k, deviceNvidiaGPU);
		std::cout << "Matrix mult block: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	{
		std::vector<float> C(m * k);
		double time = multMatrixBlockOpt(A.data(), B.data(), C.data(), m, n, k, deviceNvidiaGPU);
		std::cout << "Matrix mult block opt: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	{
		std::vector<float> C(m * k);
		double time = multMatrixBlockImage(A.data(), B.data(), C.data(), m, n, k, deviceNvidiaGPU);
		std::cout << "Matrix mult block image: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	std::cout << std::endl << "--------------------OpenCL Intel GPU--------------------" << std::endl;

	{
		std::vector<float> C(m * k);
		double time = multMatrix(A.data(), B.data(), C.data(), m, n, k, deviceIntelGPU);
		std::cout << "Matrix mult: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	{
		std::vector<float> C(m * k);
		double time = multMatrixBlock(A.data(), B.data(), C.data(), m, n, k, deviceIntelGPU);
		std::cout << "Matrix mult block: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	{
		std::vector<float> C(m * k);
		double time = multMatrixBlockOpt(A.data(), B.data(), C.data(), m, n, k, deviceIntelGPU);
		std::cout << "Matrix mult block opt: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
	}

	{
		std::vector<float> C(m * k);
		double time = multMatrixBlockImage(A.data(), B.data(), C.data(), m, n, k, deviceIntelGPU);
		std::cout << "Matrix mult block image: " << time << ' ';
		std::cout << (checkMatrix(cResult, C) ? "OK" : "BAD") << std::endl;
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