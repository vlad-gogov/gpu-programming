#pragma once

#include <CL/cl.h>

void multMatrix(float* a, float* b, float* c, int m, int n, int k);

void multMatrixOmp(float* a, float* b, float* c, int m, int n, int k);

double multMatrix(float* a, float* b, float* c, int m, int n, int k, cl_device_id deviceId);
double multMatrixBlock(float* a, float* b, float* c, int m, int n, int k, cl_device_id deviceId);
double multMatrixBlockOpt(float* a, float* b, float* c, int m, int n, int k, cl_device_id deviceId);
double multMatrixBlockImage(float* a, float* b, float* c, int m, int n, int k, cl_device_id deviceId);
