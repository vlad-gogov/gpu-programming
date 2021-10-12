#include <CL/cl.h>

#include <iostream>
#include <vector>

#include "Additional.h"

int main() {
	cl_uint platformCount = 0;
	clGetPlatformIDs(0, nullptr, &platformCount);
	cl_platform_id* platform = new cl_platform_id[platformCount]; 
	clGetPlatformIDs(platformCount, platform, nullptr);
	for (cl_uint i = 0; i < platformCount; ++i) {
		char platformName[128];
		clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 128, platformName, nullptr);
		std::cout << platformName << std::endl;
	}

	cl_uint deviceCount = 0;
	cl_device_id deviceId = 0;
	clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 1, &deviceId, &deviceCount);
	cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
	cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);
	
	IAm(context, deviceId, queue, 512);
	std::vector<cl_uint> vec(512, 0);
	SumGlobalId(vec, context, deviceId, queue);
	
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	delete[] platform;
}