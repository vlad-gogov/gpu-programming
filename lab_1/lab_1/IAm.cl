__kernel void IAm() {
    int groupId = get_group_id(0);
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    printf("I am from %d block, %d thread (global index: %d)\n", groupId, localId, globalId);
}