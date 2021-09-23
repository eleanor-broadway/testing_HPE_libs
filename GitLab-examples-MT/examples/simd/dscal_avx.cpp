// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <vector>
#include <cassert>
#include <iostream>
#include <cmath>
#include <x86intrin.h>
#include <immintrin.h>

// a dscal assuming alignment
void dscal(int n, double a, double* x)
{
  // broadcast the scale factor into a register
  __m256d x0 = _mm256_broadcast_sd(&a);
  
  // we assume alignment
  std::size_t xv = *reinterpret_cast<std::size_t*>(&x);
  assert(xv % 16 == 0);
  
  int ndiv4 = n/4;
  
  // loop over chunks of 4 values
  for (int i=0; i<ndiv4; ++i) {
    __m256d x1 = _mm256_load_pd(x+4*i);  // aligned (fast) load
    __m256d x2 = _mm256_mul_pd(x0,x1);   // multiply
    _mm256_store_pd(x+4*i,x2);          // store back aligned
  }
  
  // do the remaining entries
  for (int i=ndiv4*4 ; i< n ; ++i)
    x[i] *= a;
}
  

int main()
{
  // initialize a vector
  std::vector<double> x(1000);
  for (int i=0; i<x.size(); ++i)
    x[i] = i;
  
  // call sscal
  std::cout << "The address is " << &x[0] << "\n";
  dscal(x.size(),4.f,&x[0]);
  
  // calculate error
  double d=0.;
  for (int i=0; i<x.size(); ++i)
    d += std::fabs(x[i]-4.*i);
  std::cout << "l1-norm of error: " << d << "\n";
}

