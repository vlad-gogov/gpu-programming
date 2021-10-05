#pragma once

void IAm(cl_context context, cl_device_id deviceId, cl_command_queue queue, size_t globalWorkSize);
void SumGlobalId(std::vector<cl_uint>& vec, cl_context context, cl_device_id deviceId, cl_command_queue queue);

void checkRetValue(const std::string& name, size_t ret);