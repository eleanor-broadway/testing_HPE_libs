#include <stdio.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[]) {
 int numprocs, rank, namelen;
 char processor_name[MPI_MAX_PROCESSOR_NAME];

 int provided;
 
 MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
 MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 MPI_Get_processor_name(processor_name, &namelen);

#pragma omp parallel
 {
  int nthreads = omp_get_num_threads();
  int thread_id = omp_get_thread_num();
  printf("Process %d on %s out of %d. Thread %d of %d.\n", rank, processor_name, numprocs, thread_id, nthreads);
 }
 MPI_Finalize();
}
