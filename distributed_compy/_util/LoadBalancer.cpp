#include "LoadBalancer.h"

void LoadBalancer(float* proc_bands_arr, int* proc_loads_arr, float total_bandwidth, int datasize, int n_proc_units)
{
    int res;
    int loads_remaining = datasize;
    float factor = datasize / total_bandwidth;
    for (int i = 0; i < n_proc_units; i++) {
        res = (int)ceil(proc_bands_arr[i]*factor);
        res = std::min(res, loads_remaining);
        loads_remaining -= res;
        proc_loads_arr[i] = res;
    }
}
#include <stdio.h>
void NodeLoadBalancer(float* proc_bands_arr, int* proc_loads_arr, int* offset_index_arr,
                      float total_bandwidth, int datasize, int n_proc_units)
{
    int res;
    int loads_remaining = datasize;
    int offset = 0;
    float factor = datasize / total_bandwidth;
    for (int i = 0; i < n_proc_units; i++) {
        res = (int)ceil(proc_bands_arr[i]*factor);
        res = std::min(res, loads_remaining);
        loads_remaining -= res;
        proc_loads_arr[i] = res;
        offset_index_arr[i] = offset;
        offset += res;
    }
}