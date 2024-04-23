#include "common.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void nccl_check(ncclResult_t status, const char *file, int line) {
  if (status != ncclSuccess) {
    printf("[NCCL ERROR] at file %s:%d:\n%s\n", file, line,
           ncclGetErrorString(status));
    exit(EXIT_FAILURE);
  }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

void mpi_check(int status, const char *file, int line) {
  if (status != MPI_SUCCESS) {
    char mpi_error[4096];
    int mpi_error_len = 0;
    assert(MPI_Error_string(status, &mpi_error[0], &mpi_error_len) ==
           MPI_SUCCESS);
    printf("[MPI ERROR] at file %s:%d:\n%.*s\n", file, line, mpi_error_len,
           mpi_error);
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

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

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
  cudaCheck(cudaSetDevice(nccl->process_rank));

  printf("Current process rank is: %d/%d\n", nccl->process_rank,
         nccl->mpi_world_size);

  // Generate and broadcast a unique NCCL ID for initialization.
  ncclUniqueId nccl_id;
  if (nccl->process_rank == 0) {
    ncclCheck(ncclGetUniqueId(&nccl_id));
  }
  mpiCheck(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                     MPI_COMM_WORLD));

  ncclCheck(ncclCommInitRank(&nccl->nccl_comm, nccl->mpi_world_size, nccl_id,
                             nccl->process_rank));

  // Create a stream for cross-device operations.
  cudaCheck(cudaStreamCreate(&nccl->stream));
}

void ncclDestroy(NCCL *nccl) {
  cudaCheck(cudaStreamDestroy(nccl->stream));
  ncclCommDestroy(nccl->nccl_comm);
  mpiCheck(MPI_Finalize());
}

float get_mean(float *arr, size_t size, int process_rank) {
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    sum += arr[i];
  }
  return sum / size;
}

int main(int argc, char **argv) {
  // Some constants
  const size_t all_reduce_buffer_size = 32 * 1024 * 1024;
  const size_t threads_per_block = 1024;

  NCCL nccl;
  ncclInit(&argc, &argv, &nccl);

  // Allocating buffers on each of the devices.
  float *all_reduce_buffer;
  cudaCheck(
      cudaMalloc(&all_reduce_buffer, all_reduce_buffer_size * sizeof(float)));

  int n_blocks = cdiv(all_reduce_buffer_size, threads_per_block);
  // Set the allocated memory to a defined value.
  set_vector<<<n_blocks, threads_per_block, 0, nccl.stream>>>(
      all_reduce_buffer, all_reduce_buffer_size,
      (float)(nccl.process_rank + 1));
  cudaCheck(cudaGetLastError());

  float *all_reduce_buffer_host =
      (float *)malloc(all_reduce_buffer_size * sizeof(float));

  cudaCheck(cudaMemcpy(all_reduce_buffer_host, all_reduce_buffer,
                       sizeof(float) * all_reduce_buffer_size,
                       cudaMemcpyDeviceToHost));

  printf("Process rank %d: average value is %.6f\n", nccl.process_rank,
         get_mean(all_reduce_buffer_host, all_reduce_buffer_size,
                  nccl.process_rank));

  float *all_reduce_buffer_recv;
  cudaCheck(cudaMalloc(&all_reduce_buffer_recv,
                       all_reduce_buffer_size * sizeof(float)));

  ncclCheck(ncclAllReduce(
      (const void *)all_reduce_buffer, (void *)all_reduce_buffer_recv,
      all_reduce_buffer_size, ncclFloat, ncclSum, nccl.nccl_comm, nccl.stream));

  cudaStreamSynchronize(nccl.stream);

  cudaCheck(cudaMemcpy(all_reduce_buffer_host, all_reduce_buffer_recv,
                       sizeof(float) * all_reduce_buffer_size,
                       cudaMemcpyDeviceToHost));

  printf("Process rank %d: average value is %.6f\n", nccl.process_rank,
         get_mean(all_reduce_buffer_host, all_reduce_buffer_size,
                  nccl.process_rank));

  free(all_reduce_buffer_host);
  cudaCheck(cudaFree(all_reduce_buffer));
  ncclDestroy(&nccl);
}

/*
#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != MPI_SUCCESS) {                                                    \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static uint64_t getHostHash(const char *string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char *hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

int main(int argc, char *argv[]) {
  int size = 32 * 1024 * 1024;

  int myRank, nRanks, localRank = 0;

  // initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  // calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                         sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < nRanks; p++) {
    if (p == myRank)
      break;
    if (hostHashs[p] == hostHashs[myRank])
      localRank++;
  }

  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0)
    ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamCreate(&s));

  // initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));

  const size_t threads_per_block = 1024;
  int n_blocks = cdiv(size, threads_per_block);
  // Set the allocated memory to a defined value.
  set_vector<<<n_blocks, threads_per_block, 0, s>>>(sendbuff, size,
                                                    (float)(myRank + 1));

  // communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void *)sendbuff, (void *)recvbuff, size,
                          ncclFloat, ncclSum, comm, s));

  // completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  float *all_reduce_buffer_host = (float *)malloc(size * sizeof(float));

  cudaCheck(cudaMemcpy(all_reduce_buffer_host, recvbuff, sizeof(float) * size,
                       cudaMemcpyDeviceToHost));

  printf("Process rank %d: average value is %.6f\n", myRank,
         get_mean(all_reduce_buffer_host, size, myRank));

  // free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  // finalizing NCCL
  ncclCommDestroy(comm);

  // finalizing MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
*/