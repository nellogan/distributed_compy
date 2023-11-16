#include "OMPReductionSum.h"

void OMPReductionSum(float* data, float* out, int size)
{
    int procs = omp_get_num_procs();
    float res = 0.0;
    #pragma omp parallel num_threads(procs)
    {
        #pragma omp for reduction(+:res)
        for (int i=0; i<size; i++)
        {
            res += data[i];
        }
    }
    out[0] = res;
}

/*
    Old fashioned way
*/
//void OMPNaiveSum(float* data, float* out, int size)
//{
//    int procs = omp_get_num_procs();
//    float final_res = 0.0;
//    float* temp_res = (float*)calloc(procs, sizeof(float));
//    #pragma omp parallel num_threads(procs)
//    {
//        int tid = omp_get_thread_num();
//        #pragma omp for
//        for (int i=0; i<size; i++)
//        {
//            temp_res[tid] += data[i];
//        }
//    }
//    // final reduction
//    for (int i=0; i<procs; i++)
//    {
//        final_res += temp_res[i];
//    }
//    free(temp_res);
//    out[0] = final_res;
//}