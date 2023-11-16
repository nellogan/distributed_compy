#include "GetGPUBands.h"
// #include <stdio.h>

int GetNumGPUs()
{
    int n_devices;
    cudaGetDeviceCount(&n_devices);
    return n_devices;
}

float* GetGPUBands(int n_devices)
{
    float* gpu_bands = (float*)malloc(n_devices*sizeof(float));
    for (int i = 0; i < n_devices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        int clock_rate = prop.memoryClockRate;
        int bus_width = prop.memoryBusWidth/8;
        float device_band = (2.0*clock_rate*(bus_width) / 1000000);
        gpu_bands[i] = (2.0*clock_rate*(bus_width) / 1000000);
    }
    return gpu_bands;
}