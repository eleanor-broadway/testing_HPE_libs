// Example codes for HPC course
// (c) 2012 Andreas Hehn and Matthias Troyer, ETH Zurich

#include "dvector.hpp"
#include <cmath>

// this is not general - it should instead be written as a template
// and use overloaded dispatch to BLAS. It is kept simple though since
// the purpose is to show the distributed operations idea

extern "C" void dscal_(int const& n, double const& alpha, double* x, int const& incx );

inline void dscal(double alpha, dvector<double>& x)
{
  // just scale the local bit
  int size = x.size();
  dscal_(size, alpha, x.data(), 1);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  
  dvector<double> x(1000000);
  
  std::cout << "Rank " << rank << " has " << x.size() << " elements out of " << x.global_size() << std::endl;
  
  // intialize the distributed vector with values 0,1,....
  for (int i=0; i<x.size();++i)
    x[i] = x.offset() + i;
  
  // scale it
  dscal(2.,x);
  
  // calculate error
  double d=0.;
  for (int i=0; i<x.size(); ++i)
    d += std::fabs(x[i]-2.*(x.offset() + i));
  std::cout << "l1-norm of error on rank " << rank << ": " << d << "\n";
  
  MPI_Finalize();
  
  
}