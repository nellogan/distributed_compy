#pragma once
#include <math.h>

void LoadBalancer(float* proc_bands_arr, int* proc_loads_arr, float total_bandwidth, int datasize, int n_proc_units);

void NodeLoadBalancer(float* proc_bands_arr, int* proc_loads_arr, int* offset_index_arr,
                      float total_bandwidth, int datasize, int n_proc_units);