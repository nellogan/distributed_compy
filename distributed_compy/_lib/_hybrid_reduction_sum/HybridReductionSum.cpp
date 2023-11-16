#include "HybridReductionSum.h"

/*
     HybridReductionSum -> No saved local or network bandwidth data
     1. each node calcs local_bands and total_local_band
     2. MPI_Reduce total_local_bands -> total_nodes_band, MPI_Gather total_local_band -> nodes_band
     3. split work load among nodes from master node
     4. scatter results to all nodes
     5. read data from data_file at nodes respective node_load and offset_index
     6. calculate HeteroReductionSum for nodes data
     7. reduce sum individual node results to get final_res
     8. return result by saving to res_file_name
*/
void HybridReductionSum(const char* data_file_name, const char* res_file_name)
{
    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);
    int world_size;
    MPI_Status status;
    MPI_Comm_size(comm, &world_size);
    float local_res = 0;
    float final_res = 0;

    // 1.
    int num_local_procs;
    float total_local_band;
    float* local_bands = GetLocalBands(&total_local_band, &num_local_procs); //dont forget to free later

    // 2.
    float* node_bands;
    float total_nodes_band;
    MPI_Reduce(&total_local_band, &total_nodes_band, 1, MPI_FLOAT, MPI_SUM, 0, comm);
    if (world_rank==0)
    {
        node_bands = (float*)calloc(world_size, sizeof(float)); //free later
    }
    MPI_Gather(&total_local_band, 1, MPI_FLOAT, node_bands, 1, MPI_FLOAT, 0, comm);

    // 3.
    int* node_loads_arr;
    int* offset_index_arr;
    if (world_rank == 0)
    {
        int data_size = GetFileLen(data_file_name);
        node_loads_arr = (int*)calloc(world_size, sizeof(int));
        offset_index_arr = (int*)calloc(world_size, sizeof(int));
        NodeLoadBalancer(node_bands, node_loads_arr, offset_index_arr, total_nodes_band, data_size, world_size);
    }
    // 4.
    int this_nodes_load = 0;
    int this_nodes_offset_index = 0;
    MPI_Scatter(node_loads_arr, 1, MPI_INT, &this_nodes_load, 1, MPI_INT, 0, comm);
    MPI_Scatter(offset_index_arr, 1, MPI_INT, &this_nodes_offset_index, 1, MPI_INT, 0, comm);
    if (world_rank==0)
    {
        free(node_bands);
        free(node_loads_arr);
        free(offset_index_arr);
    }
    // 5.
    MPI_File fh;
    float* data = (float*)malloc(this_nodes_load*sizeof(float));
    if(MPI_File_open(comm, data_file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    {
        printf("[MPI process %d] Failure in opening data file.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    MPI_File_set_view(fh, this_nodes_offset_index*sizeof(float), MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, data, this_nodes_load, MPI_FLOAT, &status);
    MPI_File_close(&fh);

    // 6.
    HeteroReductionSumBands(data, &local_res, this_nodes_load, num_local_procs, local_bands, total_local_band);

    // 7.
    MPI_Reduce(&local_res, &final_res, 1, MPI_FLOAT, MPI_SUM, 0, comm);
    // 8.
    // NOTE: Could just write final_res to res_file_name from master thread only instead of using collective write.
    int access_mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    if(MPI_File_open(comm, res_file_name, access_mode, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    {
        printf("[MPI process %d] Failure in opening result file.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    MPI_File_set_view(fh, 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

    if (world_rank==0)
    {
        MPI_File_write_all(fh, &final_res, 1, MPI_FLOAT, &status);
    }
    else
    {
        MPI_File_write_all(fh, &final_res, 0, MPI_FLOAT, &status);
    };

    MPI_File_close(&fh);

    MPI_Finalize();
}


// Similar to HybridReductionSum but instead reads param data from files
void HybridReductionSumConfigured(const char* data_file_name, const char* res_file_name,
                    const char* local_bands_file, const char* total_local_band_file,
                    const char* node_bands_file, const char* total_node_band_file)
{
    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);
    int world_size;
    MPI_Status status;
    MPI_Comm_size(comm, &world_size);
    float local_res = 0;
    float final_res = 0;
    int this_nodes_load = 0;
    int this_nodes_offset_index = 0;
    int* node_loads_arr;
    int* offset_index_arr;

    if (world_rank == 0)
    {
        int data_size = GetFileLen(data_file_name);
        int num_nodes = GetFileLen(node_bands_file);
        if (world_size < num_nodes)
        {
            num_nodes = world_size;
        }

        float* node_bands = DataExtractor(node_bands_file, num_nodes);
        float* total_nodes_band_arr = DataExtractor(total_node_band_file, 1);
        // arr because DataExtractor returns a pointer array, could write separate modified function
        float total_nodes_band = total_nodes_band_arr[0];
        free(total_nodes_band_arr);

        node_loads_arr = (int*)calloc(world_size, sizeof(int));
        offset_index_arr = (int*)calloc(world_size, sizeof(int));
        /*
            Could adjust NodeLoadBalancer to accommodate more than one process per node by changing
            arg: n_proc_units -> world_size
            and
            adding arg: num_nodes
            then
            changing res = ceil(proc_bands_arr[i]*factor) in loop
            to: res = ceil(proc_bands_arr[i%num_nodes]*factor);
        */
        NodeLoadBalancer(node_bands, node_loads_arr, offset_index_arr, total_nodes_band, data_size, num_nodes);
    }

    MPI_Scatter(node_loads_arr, 1, MPI_INT, &this_nodes_load, 1, MPI_INT, 0, comm);
    MPI_Scatter(offset_index_arr, 1, MPI_INT, &this_nodes_offset_index, 1, MPI_INT, 0, comm);

   if (world_rank==0)
    {
        free(node_loads_arr);
        free(offset_index_arr);
    }

    MPI_File fh;

    float* data = (float*)malloc(this_nodes_load*sizeof(float)); //change to malloc later?
    if(MPI_File_open(comm, data_file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    {
        printf("[MPI process %d] Failure in opening data file.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    MPI_File_set_view(fh, this_nodes_offset_index*sizeof(float), MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

    MPI_File_read_all(fh, data, this_nodes_load, MPI_FLOAT, &status);
    MPI_File_close(&fh);

    HeteroReductionSumConfigured(data, &local_res, this_nodes_load, local_bands_file, total_local_band_file);

    MPI_Reduce(&local_res, &final_res, 1, MPI_FLOAT, MPI_SUM, 0, comm);

    int access_mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    if(MPI_File_open(comm, res_file_name, access_mode, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    {
        printf("[MPI process %d] Failure in opening result file.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    MPI_File_set_view(fh, 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

    if (world_rank==0)
    {
        MPI_File_write_all(fh, &final_res, 1, MPI_FLOAT, &status);
    }
    else
    {
        MPI_File_write_all(fh, &final_res, 0, MPI_FLOAT, &status);
    };

    MPI_File_close(&fh);

    MPI_Finalize();
}