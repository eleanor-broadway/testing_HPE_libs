// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <x86intrin.h>
#include <immintrin.h>

// a saxpy version requiring alignment
void saxpy(int n, float a, float* x, float* y)
{
  // load the scale factor four times into a regsiter
  __m256 x0 = _mm256_broadcast_ss(&a);
  
  // loop over chunks of 4 values
  int ndiv8 = n/8;
  for (int i=0; i<ndiv8; ++i) {
    __m256 x1 = _mm256_load_ps(x+8*i); // aligned (fast) load
    __m256 x2 = _mm256_load_ps(y+8*i); // aligned (fast) load
    __m256 x3 = _mm256_mul_ps(x0,x1);  // multiply
    __m256 x4 = _mm256_add_ps(x2,x3);  // add
    _mm256_store_ps(y+8*i,x4);         // store back aligned
  }
  
  // do the remaining entries
  for (int i=ndiv8*8 ; i< n ; ++i)
    y[i] += a*x[i];
}
