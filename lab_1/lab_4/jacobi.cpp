#include "jacobi.h"

#include <cmath>
#include <vector>

#include <omp.h>

#include "utils.h"

float norm(const std::vector<float>& x0, const std::vector<float>& x1) {
    size_t n = x0.size();
    float sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += (x0[i] - x1[i]) * (x0[i] - x1[i]) / (x1[i] * x1[i]);
    }
    return std::sqrt(sum);
}

float accuracy(float* a, float* b, float* x, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) {
        float sum_num = 0;
        for (int j = 0; j < n; j++)
            sum_num += a[j * n + i] * x[j];
        sum_num -= b[i];
        norm += sum_num * sum_num / (b[i] * b[i]);
    }
    return sqrt(norm);
}

Results jacobi(float* a, float* b, float* x, int n, int iter, float epsilon, cl_device_id deviceId) {
    Results results;
    results.full_time = omp_get_wtime();
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    std::string source = readFile("jacobi.cl");
    const char* strings[] = { source.c_str() };
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "jacobi", nullptr);

    size_t vecSize = static_cast<size_t>(n) * sizeof(float);
    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * vecSize, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, n * vecSize, a, 0, nullptr, nullptr);
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, vecSize, b, 0, nullptr, nullptr);

    cl_mem x0Mem = clCreateBuffer(context, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    cl_mem x1Mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vecSize, nullptr, nullptr);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &x0Mem);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &x1Mem);
    clSetKernelArg(kernel, 4, sizeof(int), &n);

    results.count_iter = 0;
    results.converg = 0;
    std::vector<float> x0(0, n);
    std::vector<float> x1(b, b + n);
    results.kernel_time = 0;
    do {
        x0 = x1;

        clEnqueueWriteBuffer(queue, x0Mem, CL_TRUE, 0, vecSize, x0.data(), 0, nullptr, nullptr);

        size_t globalWorkSize = static_cast<size_t>(n);

        double t1 = omp_get_wtime();
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue);
        double t2 = omp_get_wtime();
        results.kernel_time += t2 - t1;
        clEnqueueReadBuffer(queue, x1Mem, CL_TRUE, 0, vecSize, x1.data(), 0, nullptr, nullptr);
        clFinish(queue);

        results.converg = norm(x0, x1);
    } while (++results.count_iter < iter && results.converg > epsilon);
    for (int i = 0; i < n; i++)
       x[i] = x1[i];

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(x0Mem);
    clReleaseMemObject(x1Mem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    results.full_time = omp_get_wtime() - results.full_time;
    return results;
}