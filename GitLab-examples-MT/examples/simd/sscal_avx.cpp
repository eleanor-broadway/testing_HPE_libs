// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <vector>
#include <cassert>
#include <iostream>
#include <cmath>
#include <x86intrin.h>
#include <immintrin.h>

// an sscal assuming alignment
void sscal(int n, float a, float* x)
{
  // broadcast the scale factor into a register
  __m256 x0 = _mm256_broadcast_ss(&a);
  
  // we assume alignment to 32 bytes
  assert(((std::size_t) x) % 32 == 0);
  
  int ndiv8 = n/8;

  // loop over chunks of 8 values
  for (int i=0; i<ndiv8; ++i) {
    __m256 x1 = _mm256_load_ps(x+8*i);  // aligned (fast) load
    __m256 x2 = _mm256_mul_ps(x0,x1);   // multiply
    _mm256_store_ps(x+8*i,x2);          // store back aligned
  }
  
  // do the remaining entries
  for (int i=ndiv8*8 ; i< n ; ++i)
    x[i] *= a;
}
  

int main()
{
  // initialize a vector
  std::vector<float> x(1000);
  for (int i=0; i<x.size(); ++i)
    x[i] = i;
  
  // call sscal
  std::cout << "The address is " << &x[0] << "\n";
  sscal(x.size(),4.f,&x[0]);
  
  // calculate error
  float d=0.;
  for (int i=0; i<x.size(); ++i)
    d += std::fabs(x[i]-4.*i);
  std::cout << "l1-norm of error: " << d << "\n";
  
  
  
}

