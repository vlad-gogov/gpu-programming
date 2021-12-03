__kernel void SumGlobalId(__global int *arr)
{
	int globalId = get_global_id(0);
	arr[globalId] += globalId;
}