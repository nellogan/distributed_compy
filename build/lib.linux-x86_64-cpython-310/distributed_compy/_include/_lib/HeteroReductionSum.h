#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "GetFileLen.h"
#include "DataExtractor.h"
#include "LoadBalancer.h"
#include "GPUReductionSum.h"
#include "OMPReductionSum.h"

//#include "../../_config/_util/_local/SetLocalBands.h"
#include "GetLocalBands.h"
#include "GetLocalBands.h"

// GetLocalBands to determine load size distributed to each processor
void HeteroReductionSum(float* data, float* res, int size);

// For previously calculated GetLocalBands or manually supplied local data
void HeteroReductionSumBands(float* data, float* res, int size,
                             int num_procs, float* local_bands, float total_local_band);

// Local data already configured -> read local bandwidth data from files
void HeteroReductionSumConfigured(float* data, float* res, int size,
                     const char *local_bands_file_name, const char *total_local_band_file_name);
