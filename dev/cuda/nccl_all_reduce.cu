#include "common.h"
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>

void nccl_check(ncclResult_t status, const char *file, int line) {
  if (status != ncclSuccess) {
    printf("[NCCL ERROR] at file %s:%d:\n%d\n", file, line, status);
    exit(EXIT_FAILURE);
  }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

void mpi_check(int status, const char *file, int line) {
  if (status != MPI_SUCCESS) {
    printf("[MPI ERROR] at file %s:%d:\n%d\n", file, line, status);
    exit(EXIT_FAILURE);
  }
}
#define mpiCheck(err) (mpi_check(err, __FILE__, __LINE__))

// Sets a vector to a predefined value
__global__ void set_vector(float *data, int N, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Check for out-of-bounds access
  if (i < N) {
    data[i] = value;
  }
}

int cdiv(int a, int b) { return (a + b - 1) / b; }

struct NCCL {
  int process_rank;
  int mpi_world_size;
  ncclComm_t nccl_comm;
  cudaStream_t stream;
};

void ncclInit(int *argc, char ***argv, NCCL *nccl) {
  // Initialize MPI.
  mpiCheck(MPI_Init(argc, argv));
  mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &nccl->process_rank));
  mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &nccl->mpi_world_size));

  // This process manages the defined device.
  cudaCheck(cudaSetDevice(0));

  printf("Current process rank is: %d/%d\n", nccl->process_rank,
         nccl->mpi_world_size);

  // Generate and broadcast a unique NCCL ID for initialization.
  ncclUniqueId nccl_id;
  if (nccl->process_rank == 0) {
    ncclGetUniqueId(&nccl_id);
  }
  mpiCheck(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                     MPI_COMM_WORLD));

  ncclComm_t comm;
  ncclCheck(ncclCommInitRank(&comm, nccl->mpi_world_size, nccl_id,
                             nccl->process_rank));

  // Create a stream for cross-device operations.
  cudaCheck(cudaStreamCreate(&nccl->stream));
}

void ncclDestroy(NCCL *nccl) {
  cudaCheck(cudaStreamDestroy(nccl->stream));
  ncclCommDestroy(nccl->nccl_comm);
  mpiCheck(MPI_Finalize());
}

int main(int argc, char **argv) {
  // Some constants
  const int all_reduce_buffer_size = 1024;
  const int threads_per_block = 64;

  NCCL nccl;
  ncclInit(&argc, &argv, &nccl);

  // Allocating buffers on each of the devices.
  float *all_reduce_buffer;
  cudaCheck(
      cudaMalloc(&all_reduce_buffer, all_reduce_buffer_size * sizeof(float)));

  int n_blocks = cdiv(all_reduce_buffer_size, threads_per_block);
  // Set the allocated memory to a defined value.
  set_vector<<<n_blocks, threads_per_block>>>(
      all_reduce_buffer, all_reduce_buffer_size, (float)nccl.process_rank);

  ncclCheck(ncclAllReduce(&all_reduce_buffer, &all_reduce_buffer,
                          all_reduce_buffer_size, ncclFloat, ncclSum,
                          nccl.nccl_comm, nccl.stream));

  float *all_reduce_buffer_host =
      (float *)malloc(all_reduce_buffer_size * sizeof(float));

  cudaCheck(cudaMemcpy(all_reduce_buffer_host, all_reduce_buffer,
                       sizeof(float) * all_reduce_buffer_size,
                       cudaMemcpyDeviceToHost));

  float sum = 0;
  for (int i = 0; i != all_reduce_buffer_size; ++i) {
    sum += all_reduce_buffer_host[i];
  }
  sum /= all_reduce_buffer_size;

  printf("Process rank %d: average value is %.6f", nccl.process_rank, sum);

  free(all_reduce_buffer_host);
  cudaCheck(cudaFree(all_reduce_buffer));
  ncclDestroy(&nccl);
}
