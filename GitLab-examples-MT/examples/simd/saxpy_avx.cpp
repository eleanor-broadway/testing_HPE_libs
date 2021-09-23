// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <x86intrin.h>
#include <immintrin.h>

#define N 100000
#define REPETITIONS 1000

// a saxpy version requiring alignment
void saxpy(int n, float a, float* x, float* y)
{
  // load the scale factor four times into a regsiter
  __m256 x0 = _mm256_broadcast_ss(&a);
  
  // we assume alignment
  assert(((std::size_t)x) % 32 == 0 && ((std::size_t)y) % 32 == 0);
  
  
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
  

int main()
{
  // initialize two vectors
  std::vector<float> x(N);
  std::vector<float> y(N);
  
  for (int i=0; i<x.size(); ++i)
    x[i] = 1.;
  
  for (int i=0; i<y.size(); ++i)
    y[i] = 2.;
  
  // call saxpy and time it
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::high_resolution_clock::now();
  for (int it=0; it<REPETITIONS; ++it )
    saxpy(x.size(),4.f/REPETITIONS,&x[0],&y[0]);
  end = std::chrono::high_resolution_clock::now();
  int elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  
  // calculate error
  float d=0.;
  for (int i=0; i<x.size(); ++i)
    d += std::fabs(y[i]-6.);
  
  std::cout << "elapsed time: " << elapsed_time/REPETITIONS << "mus\n";
  std::cout << "l1-norm of error: " << d << "\n";
}

