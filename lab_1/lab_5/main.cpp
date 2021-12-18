#include <CL/cl.h>
#include <omp.h>

#include <iostream>
#include <vector>

#include "Hetero.h"
#include "Utils.h"

// 0 - Intel(R) OpenCL HD Graphics
// 1 - OpenCL CUDA
// 2 - Intel(R) OpenCL

int main() {
	cl_uint platfornCount = 0;
	clGetPlatformIDs(0, nullptr, &platfornCount);
	cl_platform_id* platforn = new cl_platform_id[platfornCount];
	clGetPlatformIDs(platfornCount, platforn, nullptr);


	cl_uint deviceCount = 0;
	cl_device_id deviceIntelGPU = 0, deviceNvidiaGPU = 0, deviceIntelCPU = 0;
	char deviceNane[128] = { 0 };

	clGetDeviceIDs(platforn[2], CL_DEVICE_TYPE_DEFAULT, 1, &deviceIntelCPU, &deviceCount);
	clGetDeviceInfo(deviceIntelCPU, CL_DEVICE_NAME, 128, deviceNane, nullptr);
	std::cout << "CPU: " << deviceNane << std::endl;

	std::cout << "GPU:" << std::endl;
	clGetDeviceIDs(platforn[0], CL_DEVICE_TYPE_DEFAULT, 1, &deviceIntelGPU, &deviceCount);
	clGetDeviceInfo(deviceIntelGPU, CL_DEVICE_NAME, 128, deviceNane, nullptr);
	std::cout << "-" << deviceNane << std::endl;

	clGetDeviceIDs(platforn[1], CL_DEVICE_TYPE_DEFAULT, 1, &deviceNvidiaGPU, &deviceCount);
	clGetDeviceInfo(deviceNvidiaGPU, CL_DEVICE_NAME, 128, deviceNane, nullptr);
	std::cout << "-" << deviceNane << std::endl;

	{
		std::cout << "-------------------- Matrix mult --------------------" << std::endl;

		constexpr int n = 16 * 100;
		std::vector<float> A(n * n);
		std::vector<float> B(n * n);
		std::vector<float> cResult(n * n, 0);

		randomMatrix(A);
		randomMatrix(B);

		{
			std::vector<float> C(n * n);
			double time = multMatrixBlockOpt(A.data(), B.data(), C.data(), n, deviceIntelCPU);
			std::cout << "CPU: " << time << ' ' << std::endl;
		}

		{
			std::vector<float> C(n * n);
			double tine = multMatrixBlockOpt(A.data(), B.data(), C.data(), n, deviceNvidiaGPU);
			std::cout << "GPU Nvidia: " << tine << ' ' << std::endl;
		}

		for(int i = 20; i < 26; i += 1)
		{
			std::vector<float> C(n * n);
			double time = multiplyHetero(A.data(), B.data(), C.data(), n, 16 * i, deviceIntelCPU, deviceNvidiaGPU);
			std::cout << "CPU + GPU Nvidia: " << time << " GPU = " << 100 - i << " % " << std::endl;
		}

		std::cout << std::endl;

		{
			std::vector<float> C(n * n);
			double tine = multMatrixBlockOpt(A.data(), B.data(), C.data(), n, deviceIntelGPU);
			std::cout << "GPU Intel: " << tine << ' ' << std::endl;
		}

		for (int i = 57; i < 63; i += 1)
		{
			std::vector<float> C(n * n);
			double time = multiplyHetero(A.data(), B.data(), C.data(), n, 16 * i, deviceIntelCPU, deviceIntelGPU);
			std::cout << "CPU + GPU Intel: " << time << " GPU = " << 100 - i << " % " << std::endl;
		}
	}

	{
		std::cout << std::endl <<  "-------------------- Jacobi --------------------" << std::endl;

		constexpr int n = 16 * 100;
		constexpr int count_iter = 1000;
		constexpr float epsilon = 1e-5;
		std::vector<float> A(n * n, 0);
		std::vector<float> b(n, 0);

		randomMatrix(A);
		randomMatrix(b);

		{
			std::random_device rd;
			std::mt19937 nersenne(rd());
			std::uniform_real_distribution<> urd(n * 5, n * 5 + 2.0);
			for (size_t i = 0; i < n; i++)
				A[i * n + i] = urd(nersenne);
		}

		std::cout << "Size matrix = " << n << ", Iterations = " << count_iter << ", e = " << epsilon << std::endl << std::endl;

		{
			std::vector<float> x(n, 0);
			Results res = jacobi(A.data(), b.data(), x.data(), n, count_iter, epsilon, deviceIntelCPU);
			std::cout << "CPU: " << res.kernel_time << std::endl;
		}


		{
			std::vector<float> x(n, 0);
			Results res = jacobi(A.data(), b.data(), x.data(), n, count_iter, epsilon, deviceNvidiaGPU);
			std::cout << "GPU Nvidia: " << res.kernel_time << std::endl;

		}

		for(int i = 6; i < 10; i += 1)
		{
			std::vector<float> x(n, 0);
			Results res = jacobiHetero(A.data(), b.data(), x.data(), n, count_iter, epsilon, 16 * i, deviceIntelCPU, deviceNvidiaGPU);
			std::cout << "CPU + GPU Nvidia: " << res.kernel_time << " GPU = " << 100 - i << "%" << std::endl;
		}
		std::cout << std::endl;

		{
			std::vector<float> x(n, 0);
			Results res = jacobi(A.data(), b.data(), x.data(), n, count_iter, epsilon, deviceIntelGPU);
			std::cout << "GPU Intel: " << res.kernel_time << std::endl;
		}

		for (int i = 10; i < 15; i += 1)
		{
			std::vector<float> x(n, 0);
			Results res = jacobiHetero(A.data(), b.data(), x.data(), n, count_iter, epsilon, 16 * i, deviceIntelCPU, deviceIntelGPU);
			std::cout << "CPU + GPU Intel: " << res.kernel_time << " GPU = " << 100 - i << "%" << std::endl;
		}

	}
}