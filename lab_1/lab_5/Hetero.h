#pragma once

#include <CL/cl.h>

#include <vector>

double multMatrixBlockOpt(float* a, float* b, float* c, int n, cl_device_id deviceId);
double multiplyHetero(float* a, float* b, float* c, int n, int delim, cl_device_id cpuDeviceId, cl_device_id gpuDeviceId);

struct Results {
	double kernel_time;
	double full_time;
	int count_iter;
	float converg;
};

Results jacobi(float* a, float* b, float* x, int n, int iter, float epsilon, cl_device_id deviceId);
float norm(const std::vector<float>& x0, const std::vector<float>& x1);
float accuracy(float* a, float* b, float* x, int n);

Results jacobiHetero(float* a, float* b, float* x, int n, int iter, float convThreshold, int delim, cl_device_id cpuDeviceId, cl_device_id gpuDeviceId);