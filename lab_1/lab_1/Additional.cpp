#include <CL/cl.h>

#include <iostream>
#include <string>
#include <vector>

#include "Additional.h"
#include "ReadFile.h"

void IAm(cl_context context, cl_device_id deviceId, cl_command_queue queue, size_t globalWorkSize) {
	std::string source = readFile("IAm.cl");
	const char* strings[] = { source.c_str() };
	cl_program program = nullptr;
	cl_kernel kernel = nullptr;
	cl_int ret = 0;
	try {
		program = clCreateProgramWithSource(context, 1, strings, nullptr, &ret);
		checkRetValue("clCreateProgramWithSource" , ret);
		ret = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
		checkRetValue("clBuildProgram", ret);
		kernel = clCreateKernel(program, "IAm", &ret);
		checkRetValue("clCreateKernel", ret);
		size_t group;
		ret = clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);
		checkRetValue("clGetKernelWorkGroupInfo", ret);
		ret = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &group, 0, nullptr, nullptr);
		checkRetValue("clEnqueueNDRangeKernel", ret);
		std::cout << std::endl;
	} catch (std::string& message) {
		std::cout << message << std::endl;
	}
	clReleaseKernel(kernel);
	clReleaseProgram(program);
}

void SumGlobalId(std::vector<cl_uint>& vec, cl_context context, cl_device_id deviceId, cl_command_queue queue) {
	size_t size = vec.size();
	cl_mem memory = nullptr;
	cl_program program = nullptr;
	cl_kernel kernel = nullptr;
	cl_int ret = 0;

	try {
		std::string source = readFile("SumGlobalId.cl");
		const char* strings[] = { source.c_str() };
		program = clCreateProgramWithSource(context, 1, strings, nullptr, &ret);
		checkRetValue("clCreateProgramWithSource", ret);
		ret = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
		checkRetValue("clBuildProgram", ret);
		kernel = clCreateKernel(program, "SumGlobalId", &ret);
		checkRetValue("clCreateKernel", ret);

		memory = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(cl_int), nullptr, &ret);
		checkRetValue("clCreateBuffer", ret);
		ret = clEnqueueWriteBuffer(queue, memory, CL_TRUE, 0, size * sizeof(cl_int), vec.data(), 0, nullptr, nullptr);
		checkRetValue("clEnqueueWriteBuffer", ret);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memory);
		checkRetValue("clSetKernelArg", ret);

		ret = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &size, nullptr, 0, nullptr, nullptr);
		checkRetValue("clEnqueueNDRangeKernel", ret);
		clFinish(queue);
		ret = clEnqueueReadBuffer(queue, memory, CL_TRUE, 0, size * sizeof(cl_int), vec.data(), 0, nullptr, nullptr);
		checkRetValue("clEnqueueReadBuffer", ret);

		for (auto elem : vec)
			std::cout << elem << " ";
		std::cout << std::endl;
	} catch (std::string& message) {
		std::cout << message << std::endl;
	}

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(memory);
}

void checkRetValue(const std::string& name, size_t ret) {
	if (ret != CL_SUCCESS) {
		throw name + " :" + std::to_string(ret);
	}
}