// Example codes for HPC course
// (c) 2012 Andreas Hehn and Matthias Troyer, ETH Zurich

#include "dvector.hpp"
#include <cmath>

// this is not general - it should instead be written as a template
// and use overloaded dispatch to BLAS. It is kept simple though since
// the purpose is to show the distributed operations idea

extern "C" double ddot_(int const& n, double* x, int const& incx, double* y, int const&  incy);

inline double ddot(dvector<double>& x, dvector<double>& y)
{
  assert(x.size() ==  y.size());
  int size = x.size();
  // get the local dot product
  double result = ddot_(size, x.data(), 1, y.data(), 1);
  
  // and perform a reduction
  MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_DOUBLE,MPI_SUM, x.communicator());
  return result;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  
  dvector<double> x(1000000);
  dvector<double> y(1000000);
  
  std::cout << "Rank " << rank << " has " << x.size() << " elements out of " << x.global_size() << std::endl;
  
  // intialize the distributed vector with values 0,1,....
  for (int i=0; i<x.size();++i) {
    x[i] = x.offset() + i;
    y[i] = 1.;
  }
  
  // scale it
  double result = ddot(x,y);
  
  if (rank==0)
    std::cout << "dot product is " << result << "\n";
  
  MPI_Finalize();
  
  
}