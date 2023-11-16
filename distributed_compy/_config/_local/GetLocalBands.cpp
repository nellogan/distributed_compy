#include "GetLocalBands.h"

float* GetLocalBands(float* total_local_band, int* num_procs)
{
    int n_gpus = GetNumGPUs();
    int procs = 1 + n_gpus; // processors = cpu + n_gpus
    float* local_bands = (float*)malloc(procs*sizeof(float));


    /*
        TODO: get cpu bandwidth from something like CPUID such as reading RAM clock speed and data bus width -> calc
        usually cpu bandwidth is bottlenecked by RAM data transfer speed
        assume dual channel, 1600MHz RAM, 64bit width bus
        2 * (1600 * 1e6)(s^-1) * 64(bits) * (1(byte)/8(bits)) / (1e9)
        = 25.6 (GB/s)
        for now assume ~25 GB/s
    */
    local_bands[0] = 25;
    float* gpu_bands = GetGPUBands(n_gpus);
    memcpy(&local_bands[1], &gpu_bands[0], n_gpus*sizeof(float));
    free(gpu_bands);

    float tmp_total_local_band = local_bands[0];
    for (int i = 1; i < procs; i++)
    {
        tmp_total_local_band += local_bands[i];
    }

    *num_procs = procs;
    *total_local_band = tmp_total_local_band;

    return local_bands;
}