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
  
  // now create a buffer and pack the values.
  // first get the size for the buffer and allocate a buffer
  int size_double, size_int;
  MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD,&size_double);
  MPI_Pack_size(1, MPI_INT,    MPI_COMM_WORLD,&size_int);
  int buffer_size = 2*size_double+size_int;
  char* buffer = new char[buffer_size];
  
  // now pack the values into the buffer on the master
  if (rank==0) {
    int pos=0;
    MPI_Pack(&a, 1, MPI_DOUBLE, buffer, buffer_size, &pos, MPI_COMM_WORLD);
    MPI_Pack(&b, 1, MPI_DOUBLE, buffer, buffer_size, &pos, MPI_COMM_WORLD);
    MPI_Pack(&nsteps, 1, MPI_INT, buffer, buffer_size, &pos, MPI_COMM_WORLD);
    assert ( pos <= buffer_size );
  }
  
  // then broadcast the buffer
  MPI_Bcast(buffer, buffer_size, MPI_PACKED, 0, MPI_COMM_WORLD);

  // and unpack on the receiving side
  int pos=0;
  MPI_Unpack(buffer, buffer_size, &pos, &a, 1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Unpack(buffer, buffer_size, &pos, &b, 1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Unpack(buffer, buffer_size, &pos, &nsteps, 1, MPI_INT, MPI_COMM_WORLD);
  assert ( pos <= buffer_size );
  
  // and delete the buffer
  delete[] buffer;
  
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