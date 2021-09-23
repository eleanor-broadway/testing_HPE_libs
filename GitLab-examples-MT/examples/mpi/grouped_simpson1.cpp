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

// define a struct for the parameters
struct parms {
  double a;            // lower bound of integration
  double b;            // upper bound of integration
  int nsteps; // number of subintervals for integration
};


double parallel_simpson(MPI_Comm comm, parms p)
{
  // get the rank and size for the current communicator
  int size;
  int rank;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  
  // integrate just one part on each rank
  double delta = (p.b-p.a)/size;
  double result = simpson(func,p.a+rank*delta,p.a+(rank+1)*delta,p.nsteps/size);
  
  //  collect the results to all ranks
  MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_SUM, comm);
  return result;
}
                        
int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // we want to do three integrals at once
  parms p[3];

  // build the datatype
  MPI_Aint p_lb, p_a, p_nsteps, p_ub;
  MPI_Get_address(p,            &p_lb);      // start of the struct is the lower bound
  MPI_Get_address(&p[0].a,      &p_a);       // address of the first double
  MPI_Get_address(&p[0].nsteps, &p_nsteps);  // address of the integter
  MPI_Get_address(p+1,          &p_ub);      // start of the next struct is the upper bound
  
  int          blocklens[] =  {0, 2, 1, 0};
  MPI_Datatype types[]     =  {MPI_LB, MPI_DOUBLE, MPI_INT, MPI_UB};
  MPI_Aint     offsets[]   =  {0, p_a-p_lb, p_nsteps-p_lb, p_ub-p_lb};

  MPI_Datatype parms_t;
  MPI_Type_create_struct(4, blocklens, offsets, types,&parms_t);
  MPI_Type_commit(&parms_t);
  
  // read the parameters on the master rank
  if (rank==0)
    for (int i=0 ; i<3 ; ++i)
      std::cin >> p[i].a >> p[i].b >> p[i].nsteps;
  // broadcast the parameters
  MPI_Bcast(p, 3, parms_t, 0, MPI_COMM_WORLD);
  
  // split the ranks into three groups
  int which = rank % 3;
  MPI_Comm comm;
  MPI_Comm_split(MPI_COMM_WORLD, which, rank, &comm);
  
  // do the integration in each group
  double result = parallel_simpson(comm,p[which]);
  
  // only the master for each group prints
  int grouprank;
  MPI_Comm_rank(comm, &grouprank);
  if (grouprank==0)
    std::cout << "Integration " << which << " results in " << result << std::endl;
  
  //  free the type and the new communicator
  MPI_Comm_free(&comm);
  MPI_Type_free(&parms_t);
  
  MPI_Finalize();
  return 0;
}
