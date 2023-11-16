#pragma once
#include "GetLocalBands.h"
#include "SetLocalBands.h"
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

void ConfigureNetwork(const char *node_bands_file_name, const char *total_network_band_file_name);

void ConfigureCluster(const char *local_bands_file_name, const char *total_local_band_file_name,
                          const char *node_bands_file_name, const char *total_network_band_file_name);