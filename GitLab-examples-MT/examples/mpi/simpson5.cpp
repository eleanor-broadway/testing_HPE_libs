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

  // define a struct for the parameters
  struct parms {
    double a;            // lower bound of integration
    double b;            // upper bound of integration
    int nsteps; // number of subintervals for integration
  };
  
  // describe this struct through sizes, offsets and types
  // and create an MPI data type
  // still dangerous since it assumes that we know any potential padding
  MPI_Datatype parms_t;
  int          blocklens[2]  =  {2,1};
  MPI_Aint     offsets[2]    =  {0,2*sizeof(double)};
  MPI_Datatype types[2]      =  {MPI_DOUBLE, MPI_INT};
  MPI_Type_create_struct(2, blocklens, offsets, types,&parms_t);
  MPI_Type_commit(&parms_t);
  
  parms p;
  
  // read the parameters on the master rank
  if (rank==0);
    std::cin >> p.a >> p.b >> p.nsteps;
  
  // broadcast the parms now using our type
  MPI_Bcast(&p, 1, parms_t, 0, MPI_COMM_WORLD);
  
  // and now free the type
  MPI_Type_free(&parms_t);
  
  // integrate just one part on each thread
  double delta = (p.b-p.a)/size;
  double result = simpson(func,p.a+rank*delta,p.a+(rank+1)*delta,p.nsteps/size);
  
  //  collect all to the master (rank 0)
  MPI_Reduce(rank==0 ? MPI_IN_PLACE : &result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   // the master prints
  if (rank==0)
    std::cout << result << std::endl;
  
  MPI_Finalize();
  return 0;
}