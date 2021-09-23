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

  double a;            // lower bound of integration
  double b;            // upper bound of integration
  int nsteps; // number of subintervals for integration
  
  // read the parameters on the master rank
  if (rank==0);
    std::cin >> a >> b >> nsteps;
  
  // and then broadcast the parameters to the other ranks
  // inefficient since three broadcasts
  MPI_Bcast(&a, 1, MPI_DOUBLE,0, MPI_COMM_WORLD);
  MPI_Bcast(&b, 1, MPI_DOUBLE,0, MPI_COMM_WORLD);
  MPI_Bcast(&nsteps, 1, MPI_INT,0, MPI_COMM_WORLD);
  
  // integrate just one part on each thread
  double delta = (b-a)/size;
  double result = simpson(func,a+rank*delta,a+(rank+1)*delta,nsteps/size);

  //  collect all to the master (rank 0)
  MPI_Reduce(MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   // the master prints
  if (rank==0)
    std::cout << result << std::endl;
  
  MPI_Finalize();
  return 0;
}