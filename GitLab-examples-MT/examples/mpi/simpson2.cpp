// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include "simpson.hpp"
#include <cmath>
#include <iostream>
#include <mpi.h>

// The function to integrate
double func(double x)
{
  return x * std::sin(x);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  
  // the ugly solution: pack all into one array of double
  double parms[3]; // lower and upper bound and steps

  // read the parameters on the master rank
  if (rank==0);
    std::cin >> parms[0] >> parms[1] >> parms[2];

  // broadcast the parameters to the other ranks and unpack
  MPI_Bcast(parms, 3, MPI_DOUBLE,0, MPI_COMM_WORLD);

  double a      = parms[0];
  double b      = parms[1];
  double nsteps = parms[2];
  
  
  // integrate just one part on each thread
  double delta = (b-a)/size;
  double result = simpson(func,a+rank*delta,a+(rank+1)*delta,nsteps/size);
  
  //  collect all to the master (rank 0)
  MPI_Reduce(rank==0 ? MPI_IN_PLACE : &result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   // the master prints
  if (rank==0)
    std::cout << result << std::endl;
  
  MPI_Finalize();
  return 0;
}