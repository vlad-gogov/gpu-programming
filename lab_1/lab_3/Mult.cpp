#include "Mult.h"

#include <string>

#include <omp.h>

#include "utils.h"

const char* kernelsName = "matrixMultBlock.cl";

void multMatrix(float* a, float* b, float* c, int m, int n, int k) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            for (int i = 0; i < n; i++) {
                c[k * row + col] += a[row * n + i] * b[col + k * i];
            }
        }
    }
}

void multMatrixOmp(float* a, float* b, float* c, int m, int n, int k) {
#pragma omp parallel for
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            for (int i = 0; i < n; i++) {
                c[k * row + col] += a[row * n + i] * b[col + k * i];
            }
        }
    }
}

double multMatrix(float* a, float* b, float* c, int m, int n, int k, cl_device_id deviceId) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    cl_mem aMem = nullptr;
    cl_mem bMem = nullptr;
    cl_mem cMem = nullptr;

    std::string source = readFile("matrixMult.cl");
    const char* strings[] = { source.c_str() };
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrixMult", nullptr);

    cl_int ret;
    aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, m * n * sizeof(float), a, 0, nullptr, nullptr);
    bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, n * k * sizeof(float), b, 0, nullptr, nullptr);
    cMem = clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &n);
    ret = clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    size_t globalWorkSize[] = { static_cast<size_t>(m), static_cast<size_t>(k) };
    size_t localWorkSize[] = { 16u, 16u };
    double t1 = omp_get_wtime();
    ret = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    double t2 = omp_get_wtime();

    ret = clEnqueueReadBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return t2 - t1;
}

double multMatrixBlock(float* a, float* b, float* c, int m, int n, int k, cl_device_id deviceId) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    cl_int ret;

    std::string source = readFile(kernelsName);
    const char* strings[] = { source.c_str() };
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, &ret);
    ret = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrixMultBlock", &ret);

    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, m * n * sizeof(float), a, 0, nullptr, nullptr);
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, n * k * sizeof(float), b, 0, nullptr, nullptr);
    cl_mem cMem = clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &n);
    ret = clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    size_t globalWorkSize[] = { static_cast<size_t>(m), static_cast<size_t>(k) };
    size_t localWorkSize[] = { 16u, 16u };
    double t1 = omp_get_wtime();
    ret = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    double t2 = omp_get_wtime();

    ret = clEnqueueReadBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return t2 - t1;
}

double multMatrixBlockOpt(float* a, float* b, float* c, int m, int n, int k, cl_device_id deviceId) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    std::string source = readFile(kernelsName);
    const char* strings[] = { source.c_str() };
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrixMultBlockOpt", nullptr);

    cl_int ret;
    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, m * n * sizeof(float), a, 0, nullptr, nullptr);
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, n * k * sizeof(float), b, 0, nullptr, nullptr);
    cl_mem cMem = clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &n);
    ret = clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    size_t globalWorkSize[] = { static_cast<size_t>(m), static_cast<size_t>(k) };
    size_t localWorkSize[] = { 16u, 16u };
    double t1 = omp_get_wtime();
    ret = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    double t2 = omp_get_wtime();

    ret = clEnqueueReadBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return t2 - t1;
}

double multMatrixBlockImage(float* a, float* b, float* c, int m, int n, int k, cl_device_id deviceId) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    cl_int ret;

    std::string source = readFile(kernelsName);
    const char* strings[] = { source.c_str() };
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, &ret);
    ret = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrixMultBlockImage", &ret);

    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_FLOAT;
    size_t origin[] = { 0, 0, 0 };

    cl_mem aMem = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, static_cast<size_t>(m), static_cast<size_t>(n), 0, nullptr, &ret);
    cl_mem bMem = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, static_cast<size_t>(n), static_cast<size_t>(k), 0, nullptr, &ret);
    cl_mem cMem = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, static_cast<size_t>(m), static_cast<size_t>(k), 0, nullptr, &ret);

    {
        size_t region[] = { static_cast<size_t>(m), static_cast<size_t>(n), 1 };
        ret = clEnqueueWriteImage(queue, aMem, CL_TRUE, origin, region, 0, 0, a, 0, nullptr, nullptr);
    }

    {
        size_t region[] = { static_cast<size_t>(n), static_cast<size_t>(k), 1 };
        ret = clEnqueueWriteImage(queue, bMem, CL_TRUE, origin, region, 0, 0, b, 0, nullptr, nullptr);
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &n);
    ret = clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t globalWorkSize[] = { static_cast<size_t>(m), static_cast<size_t>(k) };
    size_t localWorkSize[] = { 16u, 16u };
    double t1 = omp_get_wtime();
    ret = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    double t2 = omp_get_wtime();

    {
        size_t region[] = { static_cast<size_t>(m), static_cast<size_t>(k), 1 };
        ret = clEnqueueReadImage(queue, cMem, CL_TRUE, origin, region, 0, 0, c, 0, nullptr, nullptr);
    }

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return t2 - t1;
}