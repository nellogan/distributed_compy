#include "ConfigureNetwork.h"

void ConfigureNetwork(const char *node_bands_file_name, const char *total_network_band_file_name)
{
    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);
    int world_size;
    MPI_Status status;
    MPI_Comm_size(comm, &world_size);

    int num_local_procs;
    float total_local_band;
    float* local_bands = GetLocalBands(&total_local_band, &num_local_procs);
    free(local_bands);

    int access_mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    MPI_File fh;

    if(MPI_File_open(comm, node_bands_file_name, access_mode, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    {
        printf("[MPI process %d] Failure in opening result file.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    MPI_File_set_view(fh, world_rank, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, &total_local_band, 1, MPI_FLOAT, &status);
    MPI_File_close(&fh);

    float total_network_band = 0;
    MPI_Reduce(&total_local_band, &total_network_band, 1, MPI_FLOAT, MPI_SUM, 0, comm);

    if(MPI_File_open(comm, total_network_band_file_name, access_mode, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    {
        printf("[MPI process %d] Failure in opening result file.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (world_rank == 0)
    {
        MPI_File_set_view(fh, 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
        MPI_File_write_all(fh, &total_network_band, 1, MPI_FLOAT, &status);
    }
    else
    {
        MPI_File_set_view(fh, 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
        MPI_File_write_all(fh, &total_network_band, 0, MPI_FLOAT, &status);
    }

    MPI_File_close(&fh);

    MPI_Finalize();
}


void ConfigureCluster(const char *local_bands_file_name, const char *total_local_band_file_name,
                          const char *node_bands_file_name, const char *total_network_band_file_name)
{
    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);
    int world_size;
    MPI_Status status;
    MPI_Comm_size(comm, &world_size);
    float total_local_band = 0;

    { //local
        SetLocalBands(local_bands_file_name, total_local_band_file_name);
        FILE *fp = fopen(total_local_band_file_name, "rb");
        fread(&total_local_band, 1*sizeof(float), 1, fp);
        fclose(fp);
    }

    int access_mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    MPI_File fh;
    if(MPI_File_open(comm, node_bands_file_name, access_mode, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    {
        printf("[MPI process %d] Failure in opening result file.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    MPI_File_set_view(fh, world_rank, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, &total_local_band, 1, MPI_FLOAT, &status);

    float total_nodes_band = 0;
    MPI_Reduce(&total_local_band, &total_nodes_band, 1, MPI_FLOAT, MPI_SUM, 0, comm);

    if(MPI_File_open(comm, total_network_band_file_name, access_mode, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    {
        printf("[MPI process %d] Failure in opening result file.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    MPI_File_set_view(fh, 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

    if (world_rank==0)
    {
        MPI_File_write_all(fh, &total_nodes_band, 1, MPI_FLOAT, &status);
    }
    else
    {
        MPI_File_write_all(fh, &total_nodes_band, 0, MPI_FLOAT, &status);
    };

    MPI_Finalize();
}