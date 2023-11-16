#include "GPUReductionSum.h"

template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

template <typename T, unsigned int block_size, bool n_is_pow2>
__global__ void SumKernel(T *g_idata, T *g_odata, unsigned int n) {
    T *sdata = SharedMemory<T>();
  // Perform first level of reduction, reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int grid_size = block_size * gridDim.x;

  T my_sum = 0;

  // We reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger grid_size and therefore fewer elements per thread
  if (n_is_pow2) {
    unsigned int i = blockIdx.x * block_size * 2 + threadIdx.x;
    grid_size = grid_size << 1;

    while (i < n) {
      my_sum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + block_size) < n) {
        my_sum += g_idata[i + block_size];
      }
      i += grid_size;
    }
  } else {
    unsigned int i = blockIdx.x * block_size + threadIdx.x;
    while (i < n) {
      my_sum += g_idata[i];
      i += grid_size;
    }
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = my_sum;
  __syncthreads();

  // do reduction in shared mem
  if ((block_size >= 512) && (tid < 256)) {
    sdata[tid] = my_sum = my_sum + sdata[tid + 256];
    __syncthreads();
  }

  if ((block_size >= 256) && (tid < 128)) {
    sdata[tid] = my_sum = my_sum + sdata[tid + 128];
    __syncthreads();
  }

  if ((block_size >= 128) && (tid < 64)) {
    sdata[tid] = my_sum = my_sum + sdata[tid + 64];
  }

  __syncthreads();

  if (tid < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (block_size >= 64) my_sum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = 1; offset < 32; offset <<= 1) {
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
    }
  }

  // write result for this block to global mem
  if (tid == 0) atomicAdd(&g_odata[0], my_sum);
}


void GPUReductionSum(float* h_in, float* h_out, int size)
{
    if (size < 1)
    {
        *h_out = 0;
        return;
    };
    float *d_in, *d_out;
    int block_size = 128;
    int num_blocks = (size+block_size-1) / block_size;
    unsigned int n_is_pow2 = !(size & (size-1));
    unsigned int smemsize = block_size * sizeof(float);
    cudaMalloc(&d_in, size*sizeof(float));
    cudaMalloc(&d_out, 1*sizeof(float));
    cudaMemcpy(d_in, h_in, size*sizeof(float), cudaMemcpyHostToDevice);
    if (n_is_pow2)
    {
        switch(block_size)
        {
            case 64:
                SumKernel<float,64, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, size); break;
            case 128:
                SumKernel<float,128, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, size); break;
            case 256:
                SumKernel<float,256, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, size); break;
            case 512:
                SumKernel<float,512, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, size); break;
        }
    }
    else
    {
        switch(block_size)
        {
            case 64:
                SumKernel<float,64, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, size); break;
            case 128:
                SumKernel<float,128, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, size); break;
            case 256:
                SumKernel<float,256, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, size); break;
            case 512:
                SumKernel<float,512, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, size); break;
        }
    };
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 1*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

void MultiGPUReductionSum(float* h_in, float* h_out, int size)
{
    if (size < 1)
    {
        *h_out = 0;
        return;
    };
    omp_set_dynamic(0);
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    omp_set_num_threads(num_gpus);
    float* tmp_h_out = (float*)malloc(num_gpus * sizeof(float));
    #pragma omp parallel num_threads(num_gpus)
    {
        int thread_id = omp_get_thread_num();
        int partition_size = size / num_gpus;
        int index = thread_id*partition_size;
        if (thread_id == num_gpus-1)
        {
            partition_size = size - index;
        }
        cudaSetDevice(thread_id);
        float *d_in, *d_out;
        int block_size = 128; //TODO: dynamically choose block_size instead of static
        int num_blocks = (partition_size+block_size-1) / block_size;
        unsigned int n_is_pow2 = !(partition_size & (partition_size-1));
        unsigned int smemsize = block_size * sizeof(float);
        cudaError_t status_in = cudaMalloc((void**)&d_in, partition_size*sizeof(float));
        if (status_in != cudaSuccess)
        {
            printf("Error allocating pinned host memory -- in\n");
        }
        cudaError_t status_out = cudaMalloc((void**)&d_out, 1*sizeof(float));
        if (status_out != cudaSuccess)
        {
            printf("Error allocating pinned host memory -- out\n");
        }
        cudaMemset(d_out, 0, 1*sizeof(float));

        cudaMemcpy(d_in, h_in+index, partition_size*sizeof(float), cudaMemcpyHostToDevice);
        if (n_is_pow2)
        {
            switch(block_size)
            {
                case 64:
                    SumKernel<float, 64, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 128:
                    SumKernel<float, 128, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 256:
                    SumKernel<float, 256, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 512:
                    SumKernel<float, 512, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
            }
        }
        else
        {
            switch(block_size)
            {
                case 64:
                    SumKernel<float, 64, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 128:
                    SumKernel<float, 128, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 256:
                    SumKernel<float, 256, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 512:
                    SumKernel<float, 512, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
            }
        };
        cudaDeviceSynchronize();//cudaStreamSynchronize(); if multiple streams per device
        cudaMemcpy(&tmp_h_out[thread_id], d_out, 1*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_in);
        cudaFree(d_out);
    }
    for (int i=1; i< num_gpus; i++)
    {
        tmp_h_out[0] += tmp_h_out[i];
    }
    *h_out = tmp_h_out[0];
    free(tmp_h_out);
}

// MultiGPUReductionSum2 relies on *loads supplied from LoadBalancer (HeteroNaiveSum.cpp and LoadBalancer.cpp)
void MultiGPUReductionSum2(float* h_in, float* h_out, int* loads, int size)
{
    if (size < 1)
    {
        *h_out = 0;
        return;
    };
    omp_set_dynamic(0);
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    omp_set_num_threads(num_gpus);
    float* tmp_h_out = (float*)calloc(num_gpus,sizeof(float));
    int partition_size;
    int index;
    #pragma omp parallel num_threads(num_gpus) private(partition_size, index)
    {
        int thread_id = omp_get_thread_num();
        partition_size = *(loads+thread_id);
        index = thread_id*partition_size;
        if(thread_id = num_gpus-1)
        {
            partition_size = size - index;
        }
        cudaSetDevice(thread_id);
        float *d_in, *d_out;
        int block_size = 128; //TODO: dynamically choose block_size instead of static
        int num_blocks = (partition_size+block_size-1) / block_size;
        unsigned int n_is_pow2 = !(partition_size & (partition_size-1));
        unsigned int smemsize = block_size * sizeof(float);
        cudaError_t status_in = cudaMalloc((void**)&d_in, partition_size*sizeof(float));
        if (status_in != cudaSuccess)
        {
            printf("Error allocating pinned host memory -- in\n");
        }
        cudaError_t status_out = cudaMalloc((void**)&d_out, 1*sizeof(float));
        if (status_out != cudaSuccess)
        {
            printf("Error allocating pinned host memory -- out\n");
        }
        cudaMemset(d_out, 0, 1*sizeof(float));

        cudaMemcpy(d_in, h_in, partition_size*sizeof(float), cudaMemcpyHostToDevice);
        if (n_is_pow2)
        {
            switch(block_size)
            {
                case 64:
                    SumKernel<float, 64, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 128:
                    SumKernel<float, 128, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 256:
                    SumKernel<float, 256, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 512:
                    SumKernel<float, 512, true><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
            }
        }
        else
        {
            switch(block_size)
            {
                case 64:
                    SumKernel<float, 64, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 128:
                    SumKernel<float, 128, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 256:
                    SumKernel<float, 256, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
                case 512:
                    SumKernel<float, 512, false><<< num_blocks, block_size, smemsize >>>(d_in, d_out, partition_size); break;
            }
        };
        cudaDeviceSynchronize();//cudaStreamSynchronize(); if multiple streams per device
        cudaMemcpy(&tmp_h_out[thread_id], d_out, 1*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_in);
        cudaFree(d_out);
    }
    for (int i=1; i< num_gpus; i++)
    {
        tmp_h_out[0] += tmp_h_out[i];
    }
    *h_out = tmp_h_out[0];
    free(tmp_h_out);
}

/*
    Pinned memory vs pageable memory overview:

    PROS: 1. Overlap copy operations and kernel launches
          2. Faster host2device and device2host transfers (limited by PCIe x(lanes) speed)
          3. Does not need to allocate on device, can directly read/write from/to pinned memory e.g. no need for
             cudaMalloc(d_in...)
          4. Inter-GPU communication can benefit greatly as CPU is bypassed (direct memory access). Make sure inter-gpu
             communications are batched to saturation threshold if message size is small.

    CONS: 1. OS needs to allocate large contiguous memory block on RAM. Can easily run out of host RAM.
          2. Number 1 above is slower than allocating pageable memory.
          3. Unless pinned memory is reused multiple times, the performance gain of faster h2d/d2h transfers is dwarfed.

    OVERALL: For reduction sum: Because the data buffer is not reused multiple times and no inter-GPU communication is
          required, use of pinned memory results in major performance loss. Opt to use pageable memory for reduction
          sum.
*/
void GPUReductionSumPinned(float* h_in, float* h_out, int size)
{
    if (size < 1)
    {
        *h_out = 0;
        return;
    };
    float *dh_pinned_in, *dh_pinned_out;
    int block_size = 128;
    int num_blocks = (size+block_size-1) / block_size;
    unsigned int n_is_pow2 = !(size & (size-1));
    unsigned int smemsize = block_size * sizeof(float);
    cudaError_t status_in = cudaMallocHost((void**)&dh_pinned_in, size*sizeof(float));
    if (status_in != cudaSuccess)
    {
        printf("Error allocating pinned host memory -- in\n");
    }
    cudaError_t status_out = cudaMallocHost((void**)&dh_pinned_out, 1*sizeof(float));
    if (status_out != cudaSuccess)
    {
        printf("Error allocating pinned host memory -- out\n");
    }
    cudaMemsetAsync(dh_pinned_out, 0, 1*sizeof(float));

    cudaMemcpyAsync(dh_pinned_in, h_in, size*sizeof(float), cudaMemcpyHostToDevice);
    if (n_is_pow2)
    {
        switch(block_size)
        {
            case 64:
                SumKernel<float, 64, true><<< num_blocks, block_size, smemsize >>>(dh_pinned_in, dh_pinned_out, size); break;
            case 128:
                SumKernel<float, 128, true><<< num_blocks, block_size, smemsize >>>(dh_pinned_in, dh_pinned_out, size); break;
            case 256:
                SumKernel<float, 256, true><<< num_blocks, block_size, smemsize >>>(dh_pinned_in, dh_pinned_out, size); break;
            case 512:
                SumKernel<float, 512, true><<< num_blocks, block_size, smemsize >>>(dh_pinned_in, dh_pinned_out, size); break;
        }
    }
    else
    {
        switch(block_size)
        {
            case 64:
                SumKernel<float, 64, false><<< num_blocks, block_size, smemsize >>>(dh_pinned_in, dh_pinned_out, size); break;
            case 128:
                SumKernel<float, 128, false><<< num_blocks, block_size, smemsize >>>(dh_pinned_in, dh_pinned_out, size); break;
            case 256:
                SumKernel<float, 256, false><<< num_blocks, block_size, smemsize >>>(dh_pinned_in, dh_pinned_out, size); break;
            case 512:
                SumKernel<float, 512, false><<< num_blocks, block_size, smemsize >>>(dh_pinned_in, dh_pinned_out, size); break;
        }
    };
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, dh_pinned_out, 1*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFreeHost(dh_pinned_in);
    cudaFreeHost(dh_pinned_out);
}

void MultiGPUReductionSumPinned(float* h_in, float* h_out, int size)
{
    if (size < 1)
    {
        *h_out = 0;
        return;
    };
    omp_set_dynamic(0);
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    omp_set_num_threads(num_gpus);
    float *pinned_in, *pinned_out;

    // cudaHostAllocPortable flag will make the memory portable across all contexts
    cudaError_t status_in = cudaHostAlloc((void**)&pinned_in, size*sizeof(float), cudaHostAllocPortable);
    if (status_in != cudaSuccess)
    {
        printf("Error allocating pinned host memory -- in\n");
    }
    cudaError_t status_out = cudaHostAlloc((void**)&pinned_out, num_gpus*sizeof(float), cudaHostAllocPortable);
    if (status_out != cudaSuccess)
    {
        printf("Error allocating pinned host memory -- out\n");
    }
    cudaMemsetAsync(pinned_out, 0, 1*sizeof(float));

    cudaMemcpy(pinned_in, h_in, size*sizeof(float), cudaMemcpyHostToDevice);

    #pragma omp parallel num_threads(num_gpus)
    {
        int thread_id = omp_get_thread_num();
        int partition_size = size / num_gpus;
        int index = thread_id*partition_size;
        if (thread_id == num_gpus-1)
        {
            partition_size = size - index;
        }
        cudaSetDevice(thread_id);
        int block_size = 128; //TODO: dynamically choose block_size instead of static
        int num_blocks = (partition_size+block_size-1) / block_size;
        unsigned int n_is_pow2 = !(partition_size & (partition_size-1));
        unsigned int smemsize = block_size * sizeof(float);
        if (n_is_pow2)
        {
            switch(block_size)
            {
                case 64:
                    SumKernel<float, 64, true><<< num_blocks, block_size, smemsize>>>
                    (pinned_in+index,pinned_out+thread_id, partition_size); break;
                case 128:
                    SumKernel<float, 128, true><<< num_blocks, block_size, smemsize >>>
                    (pinned_in+index, pinned_out+thread_id, partition_size); break;
                case 256:
                    SumKernel<float, 256, true><<< num_blocks, block_size, smemsize >>>
                    (pinned_in+index, pinned_out+thread_id, partition_size); break;
                case 512:
                    SumKernel<float, 512, true><<< num_blocks, block_size, smemsize >>>
                    (pinned_in+index, pinned_out+thread_id, partition_size); break;
            }
        }
        else
        {
            switch(block_size)
            {
                case 64:
                    SumKernel<float, 64, false><<< num_blocks, block_size, smemsize >>>
                    (pinned_in+index, pinned_out+thread_id, partition_size); break;
                case 128:
                    SumKernel<float, 128, false><<< num_blocks, block_size, smemsize >>>
                    (pinned_in+index, pinned_out+thread_id, partition_size); break;
                case 256:
                    SumKernel<float, 256, false><<< num_blocks, block_size, smemsize >>>
                    (pinned_in+index, pinned_out+thread_id, partition_size); break;
                case 512:
                    SumKernel<float, 512, false><<< num_blocks, block_size, smemsize >>>
                    (pinned_in+index, pinned_out+thread_id, partition_size); break;
            }
        };
        cudaDeviceSynchronize();//cudaStreamSynchronize(); if multiple streams per device
    }
    for (int i=1; i< num_gpus; i++)
    {
        pinned_out[0] += pinned_out[i];
    }
    *h_out = pinned_out[0];
    cudaFree(pinned_in);
    cudaFree(pinned_out);
}