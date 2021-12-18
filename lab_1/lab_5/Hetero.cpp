#include "Hetero.h"

#include <cmath>
#include <vector>

#include <omp.h>

#include "utils.h"

double multMatrixBlockOpt(float* a, float* b, float* c, int n, cl_device_id deviceId) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    std::string source = readFile("MatrixMult.cl");
    const char* strings[] = { source.c_str() };
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrixMultBlockOpt", nullptr);

    cl_int ret;
    size_t byteSize = n * n * sizeof(float);
    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, byteSize, nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, byteSize, a, 0, nullptr, nullptr);
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, byteSize, nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, byteSize, b, 0, nullptr, nullptr);
    cl_mem cMem = clCreateBuffer(context, CL_MEM_READ_WRITE, byteSize, nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, cMem, CL_TRUE, 0, byteSize, c, 0, nullptr, nullptr);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &n);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    size_t globalWorkSize[] = { static_cast<size_t>(n), static_cast<size_t>(n) };
    size_t localWorkSize[] = { 16u, 16u };
    double t1 = omp_get_wtime();
    ret = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    double t2 = omp_get_wtime();

    ret = clEnqueueReadBuffer(queue, cMem, CL_TRUE, 0, byteSize, c, 0, nullptr, nullptr);

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return t2 - t1;
}

double multiplyHetero(float* a, float* b, float* c, int n, int delim, cl_device_id cpuDeviceId, cl_device_id gpuDeviceId) {
    cl_int ret = 0;
    cl_context cpuContext = clCreateContext(nullptr, 1, &cpuDeviceId, nullptr, nullptr, &ret);
    cl_context gpuContext = clCreateContext(nullptr, 1, &gpuDeviceId, nullptr, nullptr, &ret);
    cl_command_queue cpuQueue = clCreateCommandQueue(cpuContext, cpuDeviceId, CL_QUEUE_PROFILING_ENABLE, &ret);
    cl_command_queue gpuQueue = clCreateCommandQueue(gpuContext, gpuDeviceId, CL_QUEUE_PROFILING_ENABLE, &ret);

    std::string source = readFile("MatrixMult.cl");
    const char* strings[] = { source.c_str() };
    cl_program cpuProgram = clCreateProgramWithSource(cpuContext, 1, strings, nullptr, &ret);
    ret = clBuildProgram(cpuProgram, 1, &cpuDeviceId, nullptr, nullptr, nullptr);
    cl_program gpuProgram = clCreateProgramWithSource(gpuContext, 1, strings, nullptr, &ret);
    ret = clBuildProgram(gpuProgram, 1, &gpuDeviceId, nullptr, nullptr, nullptr);
    cl_kernel cpuKernel = clCreateKernel(cpuProgram, "matrixMultBlockOpt", &ret);
    cl_kernel gpuKernel = clCreateKernel(gpuProgram, "matrixMultBlockOpt", &ret);

    size_t byteSize = n * n * sizeof(float);
    cl_mem aMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, byteSize, nullptr, &ret);
    cl_mem aMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, byteSize, nullptr, &ret);
    ret = clEnqueueWriteBuffer(cpuQueue, aMemCpu, CL_FALSE, 0, byteSize, a, 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(gpuQueue, aMemGpu, CL_FALSE, 0, byteSize, a, 0, nullptr, nullptr);
    cl_mem bMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, byteSize, nullptr, &ret);
    cl_mem bMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, byteSize, nullptr, &ret);
    ret = clEnqueueWriteBuffer(cpuQueue, bMemCpu, CL_FALSE, 0, byteSize, b, 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(gpuQueue, bMemGpu, CL_FALSE, 0, byteSize, b, 0, nullptr, nullptr);
    cl_mem cMemCpu = clCreateBuffer(cpuContext, CL_MEM_WRITE_ONLY, byteSize, nullptr, &ret);
    cl_mem cMemGpu = clCreateBuffer(gpuContext, CL_MEM_WRITE_ONLY, byteSize, nullptr, &ret);
    ret = clFinish(cpuQueue);
    ret = clFinish(gpuQueue);

    ret = clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), &aMemCpu);
    ret = clSetKernelArg(cpuKernel, 1, sizeof(cl_mem), &bMemCpu);
    ret = clSetKernelArg(cpuKernel, 2, sizeof(cl_mem), &cMemCpu);
    ret = clSetKernelArg(cpuKernel, 3, sizeof(int), &n);

    ret = clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), &aMemGpu);
    ret = clSetKernelArg(gpuKernel, 1, sizeof(cl_mem), &bMemGpu);
    ret = clSetKernelArg(gpuKernel, 2, sizeof(cl_mem), &cMemGpu);
    ret = clSetKernelArg(gpuKernel, 3, sizeof(int), &n);

    size_t cpuWorkSize[] = { static_cast<size_t>(n), static_cast<size_t>(delim) };
    size_t gpuWorkSize[] = { static_cast<size_t>(n), static_cast<size_t>(n - delim) };
    size_t cpuOffset[] = { static_cast<size_t>(0), static_cast<size_t>(0) };
    size_t gpuOffset[] = { static_cast<size_t>(0), static_cast<size_t>(delim) };
    size_t localWorkSize[] = { 16u, 16u };

    cl_event events[2];

    ret = clEnqueueNDRangeKernel(cpuQueue, cpuKernel, 2, cpuOffset, cpuWorkSize, localWorkSize, 0, nullptr, events + 0);
    ret = clEnqueueNDRangeKernel(gpuQueue, gpuKernel, 2, gpuOffset, gpuWorkSize, localWorkSize, 0, nullptr, events + 1);
    clWaitForEvents(2, events);
    ret = clFinish(cpuQueue);
    ret = clFinish(gpuQueue);

    cl_ulong cpuTime[2], gpuTime[2];
    clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), cpuTime, nullptr);
    clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), cpuTime + 1, nullptr);
    clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), gpuTime, nullptr);
    clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), gpuTime + 1, nullptr);
    double times[2] = { 0 };
    times[0] = (cpuTime[1] - cpuTime[0]) / 1e9;
    times[1] = (gpuTime[1] - gpuTime[0]) / 1e9;
    double time =  times[0] > times[1] ? times[0] : times[1];

    ret = clEnqueueReadBuffer(cpuQueue, cMemCpu, CL_FALSE, 0, delim * n * sizeof(float), c, 0, nullptr, nullptr);
    ret = clEnqueueReadBuffer(gpuQueue, cMemGpu, CL_FALSE, delim * n * sizeof(float),
        byteSize - delim * n * sizeof(float), c + delim * n, 0, nullptr, nullptr);
    ret = clFinish(cpuQueue);
    ret = clFinish(gpuQueue);

    ret = clReleaseMemObject(aMemCpu);
    ret = clReleaseMemObject(bMemCpu);
    ret = clReleaseMemObject(cMemCpu);
    ret = clReleaseMemObject(aMemGpu);
    ret = clReleaseMemObject(bMemGpu);
    ret = clReleaseMemObject(cMemGpu);
    ret = clReleaseKernel(cpuKernel);
    ret = clReleaseKernel(gpuKernel);
    ret = clReleaseProgram(cpuProgram);
    ret = clReleaseProgram(gpuProgram);
    ret = clReleaseCommandQueue(cpuQueue);
    ret = clReleaseCommandQueue(gpuQueue);
    ret = clReleaseContext(cpuContext);
    ret = clReleaseContext(gpuContext);

    return time;
}


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

    std::string source = readFile("Jacobi.cl");
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

Results jacobiHetero(float* a, float* b, float* x, int n, int iter, float convThreshold, int delim, cl_device_id cpuDeviceId, cl_device_id gpuDeviceId) {
    Results results;
    results.full_time = omp_get_wtime();
    cl_int ret = 0;

    cl_context cpuContext = clCreateContext(nullptr, 1, &cpuDeviceId, nullptr, nullptr, &ret);
    cl_context gpuContext = clCreateContext(nullptr, 1, &gpuDeviceId, nullptr, nullptr, &ret);
    cl_command_queue cpuQueue = clCreateCommandQueue(cpuContext, cpuDeviceId, CL_QUEUE_PROFILING_ENABLE, &ret);
    cl_command_queue gpuQueue = clCreateCommandQueue(gpuContext, gpuDeviceId, CL_QUEUE_PROFILING_ENABLE, &ret);

    std::string source = readFile("Jacobi.cl");
    const char* strings[] = { source.c_str() };
    cl_program cpuProgram = clCreateProgramWithSource(cpuContext, 1, strings, nullptr, &ret);
    cl_program gpuProgram = clCreateProgramWithSource(gpuContext, 1, strings, nullptr, &ret);

    ret = clBuildProgram(cpuProgram, 1, &cpuDeviceId, nullptr, nullptr, nullptr);
    ret = clBuildProgram(gpuProgram, 1, &gpuDeviceId, nullptr, nullptr, nullptr);
    cl_kernel cpuKernel = clCreateKernel(cpuProgram, "jacobi", &ret);
    cl_kernel gpuKernel = clCreateKernel(gpuProgram, "jacobi", &ret);

    size_t vecSize = static_cast<size_t>(n) * sizeof(float);
    cl_mem aMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, n * vecSize, nullptr, nullptr);
    cl_mem bMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    cl_mem x0MemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    cl_mem x1MemCpu = clCreateBuffer(cpuContext, CL_MEM_WRITE_ONLY, vecSize, nullptr, nullptr);

    cl_mem aMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, n * vecSize, nullptr, nullptr);
    cl_mem bMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    cl_mem x0MemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    cl_mem x1MemGpu = clCreateBuffer(gpuContext, CL_MEM_WRITE_ONLY, vecSize, nullptr, nullptr);

    clEnqueueWriteBuffer(cpuQueue, aMemCpu, CL_FALSE, 0, n * vecSize, a, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cpuQueue, bMemCpu, CL_FALSE, 0, vecSize, b, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(gpuQueue, aMemGpu, CL_FALSE, 0, n * vecSize, a, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(gpuQueue, bMemGpu, CL_FALSE, 0, vecSize, b, 0, nullptr, nullptr);
    clFinish(cpuQueue);
    clFinish(gpuQueue);

    clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), &aMemCpu);
    clSetKernelArg(cpuKernel, 1, sizeof(cl_mem), &bMemCpu);
    clSetKernelArg(cpuKernel, 2, sizeof(cl_mem), &x0MemCpu);
    clSetKernelArg(cpuKernel, 3, sizeof(cl_mem), &x1MemCpu);
    clSetKernelArg(cpuKernel, 4, sizeof(int), &n);

    clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), &aMemGpu);
    clSetKernelArg(gpuKernel, 1, sizeof(cl_mem), &bMemGpu);
    clSetKernelArg(gpuKernel, 2, sizeof(cl_mem), &x0MemGpu);
    clSetKernelArg(gpuKernel, 3, sizeof(cl_mem), &x1MemGpu);
    clSetKernelArg(gpuKernel, 4, sizeof(int), &n);

    results.count_iter = 0;
    results.converg = 0;
    results.kernel_time = 0;

    std::vector<float> x0(0, n);
    std::vector<float> x1(b, b + n);
    size_t cpuWorkSize = static_cast<size_t>(delim);
    size_t gpuWorkSize = static_cast<size_t>(n - delim);
    size_t cpuOffset = 0;
    size_t gpuOffset = static_cast<size_t>(delim);

    cl_event events[2];
    cl_ulong cpuTime[2], gpuTime[2];
    double times[2] = { 0 };

    do {
        x0 = x1;
        clEnqueueWriteBuffer(cpuQueue, x0MemCpu, CL_FALSE, 0, vecSize, x0.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(gpuQueue, x0MemGpu, CL_FALSE, 0, vecSize, x0.data(), 0, nullptr, nullptr);
        clFinish(cpuQueue);
        clFinish(gpuQueue);

        ret = clEnqueueNDRangeKernel(cpuQueue, cpuKernel, 1, &cpuOffset, &cpuWorkSize, nullptr, 0, nullptr, events + 0);
        ret = clEnqueueNDRangeKernel(gpuQueue, gpuKernel, 1, &gpuOffset, &gpuWorkSize, nullptr, 0, nullptr, events + 1);
        clWaitForEvents(2, events);
        clFinish(cpuQueue);
        clFinish(gpuQueue);

        clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), cpuTime, nullptr);
        clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), cpuTime + 1, nullptr);
        clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), gpuTime, nullptr);
        clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), gpuTime + 1, nullptr);
        times[0] = (cpuTime[1] - cpuTime[0]) / 1e9;
        times[1] = (gpuTime[1] - gpuTime[0]) / 1e9;
        results.kernel_time += times[0] > times[1] ? times[0] : times[1];

        clEnqueueReadBuffer(cpuQueue, x1MemCpu, CL_FALSE, 0, delim * sizeof(float), x1.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(gpuQueue, x1MemGpu, CL_FALSE, delim * sizeof(float), vecSize - delim * sizeof(float),
            x1.data() + delim, 0, nullptr, nullptr);
        clFinish(cpuQueue);
        clFinish(gpuQueue);
        results.converg = norm(x0, x1);
    } while (++results.count_iter < iter && results.converg > convThreshold);

    for (int i = 0; i < n; i++)
        x[i] = x1[i];

    clReleaseMemObject(aMemCpu);
    clReleaseMemObject(bMemCpu);
    clReleaseMemObject(x0MemCpu);
    clReleaseMemObject(x1MemCpu);
    clReleaseKernel(cpuKernel);
    clReleaseProgram(cpuProgram);
    clReleaseCommandQueue(cpuQueue);
    clReleaseContext(cpuContext);

    clReleaseMemObject(aMemGpu);
    clReleaseMemObject(bMemGpu);
    clReleaseMemObject(x0MemGpu);
    clReleaseMemObject(x1MemGpu);
    clReleaseKernel(gpuKernel);
    clReleaseProgram(gpuProgram);
    clReleaseCommandQueue(gpuQueue);
    clReleaseContext(gpuContext);

    results.full_time = omp_get_wtime() - results.full_time;
    return results;
}