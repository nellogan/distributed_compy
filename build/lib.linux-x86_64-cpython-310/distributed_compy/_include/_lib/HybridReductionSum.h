#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "GetLocalBands.h"
#include "GetFileLen.h"
#include "DataExtractor.h"
#include "LoadBalancer.h"
#include "HeteroReductionSum.h"
#include "mpi.h"


void HybridReductionSum(const char* data_file_name, const char* res_file_name);

void HybridReductionSumConfigured(const char* data_file_name, const char* res_file_name,
                    const char* local_bands_file, const char* total_local_band_file,
                    const char* node_bands_file, const char* total_node_band_file);