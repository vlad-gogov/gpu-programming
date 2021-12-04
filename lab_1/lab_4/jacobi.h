#pragma once

#include <CL/cl.h>

struct Results {
	double kernel_time;
	double full_time;
	int count_iter;
	float converg;
};

Results jacobi(float* a, float* b, float* x, int n, int iter, float epsilon, cl_device_id deviceId);
float accuracy(float* a, float* b, float* x, int n);