#include <CL/cl.h>

#include <iostream>
#include <vector>

#include "Additional.h"

int main() {
	cl_uint platformCount = 0;
	clGetPlatformIDs(0, nullptr, &platformCount);
	cl_platform_id* platform = new cl_platform_id[platformCount]; 
	clGetPlatformIDs(platformCount, platform, nullptr);
	cl_uint deviceCount = 0;
	cl_device_id deviceId = 0;
	for (cl_uint i = 0; i < platformCount; ++i) {
		char platformName[128];
		clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 128, platformName, nullptr);
		clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 1, &deviceId, &deviceCount);
		std::cout << platformName << " Device Count: " << deviceCount << std::endl;
		
	}
	char cBuffer[1024];
	deviceId = 0;
	cl_int ret = clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &deviceCount);
	clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
	std::cout << "DEVICE: " << cBuffer << std::endl;
	cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
	cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);
	
	IAm(context, deviceId, queue, 512);
	std::vector<cl_uint> vec(512, 0);
	SumGlobalId(vec, context, deviceId, queue);
	
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	delete[] platform;
}