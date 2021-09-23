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
  
  parms p;

  // describe this struct through sizes, offsets and types
  // the safe way getting addresses

  MPI_Aint p_lb, p_a, p_nsteps, p_ub;
  MPI_Get_address(&p,       &p_lb);      // start of the struct is the lower bound
  MPI_Get_address(&p.a,     &p_a);       // address of the first double
  MPI_Get_address(&p.nsteps, &p_nsteps); // address of the integter
  MPI_Get_address(&p+1,     &p_ub);      // start of the next struct is the upper bound
  
  int          blocklens[] =  {0, 2, 1, 0};
  MPI_Datatype types[]     =  {MPI_LB, MPI_DOUBLE, MPI_INT, MPI_UB};
  MPI_Aint     offsets[]   =  {0, p_a-p_lb, p_nsteps-p_lb, p_ub-p_lb};

  MPI_Datatype parms_t;
  MPI_Type_create_struct(4, blocklens, offsets, types,&parms_t);
  MPI_Type_commit(&parms_t);
  
  
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