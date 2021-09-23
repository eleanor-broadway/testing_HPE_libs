// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <iomanip>
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  long double sum=0.;
  long double localsum=0.;
  
  unsigned long const nterms = 100000000;
  long double const step = (nterms+0.5l) /  size;
  
  // do just one piece on each rank
  unsigned long start = rank * step;
  unsigned long end = (rank+1) * step;
  for (std::size_t t = start; t < end; ++t)
    localsum += (1.0 - 2* (t % 2)) / (2*t + 1);
  
  // now collect all to the master (rank 0)
  MPI_Reduce(&localsum, &sum, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if (rank==0) // only one prints
    std::cout << "pi=" << std::setprecision(18) << 4.*sum << std::endl;
  
  MPI_Finalize();
  return 0;
}
