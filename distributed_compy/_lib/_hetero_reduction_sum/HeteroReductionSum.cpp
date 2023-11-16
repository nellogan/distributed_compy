#include "HeteroReductionSum.h"

/*
    omp_set_max_active_levels(2);

    default nested #pragma omp parallels are off by default (default=1)
    2 max levels:
    -> allows for all cpu cores to fire from OMPNaiveSum -> level 2
    -> allows MultiGPUReductionSum2 to fire from as many gpu_threads as there is available gpus -> level 2
    similar to an N-ary tree level order.
                        _________________________________________
                        | HeteroNaiveSum omp parallel level = 1 |
                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                               /                        \
                              /                          \
                             /                            \
                            /                              \
                           /                                \
    ______________________________________          ________________________________________________
    | OMPNaiveSum omp parallel level = 2 |          | MultiGPUReductionSum2 omp parallel level = 2 |
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
//no saved bands data, GetLocalBands
void HeteroReductionSum(float* data, float* res, int size)
{
    if (size < 1)
    {
        *res = 0;
    }
    int num_procs;
    float total_local_band;
    float* local_bands = GetLocalBands(&total_local_band, &num_procs);
    int* proc_loads_arr = (int*)calloc(num_procs, sizeof(int));
    LoadBalancer(local_bands, proc_loads_arr, total_local_band, size, num_procs);
    free(local_bands);
    int cpu_load = proc_loads_arr[0];
    float cpu_out = 0;
    float gpu_out = 0;
    omp_set_max_active_levels(2);

    #pragma omp parallel num_threads(2)
    {
        int thread_id = omp_get_thread_num();

        if (thread_id == 0) // CPU thread
        {
            OMPReductionSum(data, &cpu_out, cpu_load);
        }
        else // GPU thread
        {
            MultiGPUReductionSum2(&data[cpu_load], &gpu_out, proc_loads_arr+1, size-cpu_load);
        }
    }
    *res = cpu_out + gpu_out;
    free(proc_loads_arr);
}

// limit the number of processors to compute HeteroReductionSum with num_procs param
void HeteroReductionSumBands(float* data, float* res, int size,
                             int num_procs, float* local_bands, float total_local_band)
{
    if (size < 1)
    {
        *res = 0;
    }
    int* proc_loads_arr = (int*)calloc(num_procs, sizeof(int));
    LoadBalancer(local_bands, proc_loads_arr, total_local_band, size, num_procs);
    free(local_bands);
    int cpu_load = proc_loads_arr[0];
    float cpu_out = 0;
    float gpu_out = 0;
    omp_set_max_active_levels(2);

    #pragma omp parallel num_threads(2)
    {
        int thread_id = omp_get_thread_num();

        if (thread_id == 0) // CPU thread
        {
            OMPReductionSum(data, &cpu_out, cpu_load);
        }
        else // GPU thread
        {
            MultiGPUReductionSum2(&data[cpu_load], &gpu_out, proc_loads_arr+1, size-cpu_load);
        }
    }
    *res = cpu_out + gpu_out;
    free(proc_loads_arr);
}


// load local_bands_file_name and total_local_band_file_name
void HeteroReductionSumConfigured(float* data, float* res, int size,
                     const char *local_bands_file_name, const char *total_local_band_file_name)
{
    if (size < 1)
    {
        *res = 0;
    }
    long num_procs = GetFileLen(local_bands_file_name);
    float* proc_bands_arr = DataExtractor(local_bands_file_name, num_procs);
    float* total_bandwidth_ptr = DataExtractor(total_local_band_file_name, 1);
    float total_bandwidth = *total_bandwidth_ptr;
    free(total_bandwidth_ptr);
    int* proc_loads_arr = (int*)calloc(num_procs, sizeof(int));
    LoadBalancer(proc_bands_arr, proc_loads_arr, total_bandwidth, size, num_procs);
    int cpu_load = proc_loads_arr[0];
    float cpu_out = 0;
    float gpu_out = 0;

    omp_set_max_active_levels(2);

    #pragma omp parallel num_threads(2)
    {
        int thread_id = omp_get_thread_num();

        if (thread_id == 0) // CPU thread
        {
            OMPReductionSum(data, &cpu_out, cpu_load);
        }
        else // GPU thread
        {
            MultiGPUReductionSum2(&data[cpu_load], &gpu_out, proc_loads_arr+1, size-cpu_load);
        }
    }
    *res = cpu_out + gpu_out;
    free(proc_bands_arr);
    free(proc_loads_arr);
}