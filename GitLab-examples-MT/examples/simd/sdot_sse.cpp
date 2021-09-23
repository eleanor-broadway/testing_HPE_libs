// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <x86intrin.h>

// alignas workaround
#include "alignas.hpp"

// an sdot version requiring alignment
float sdot(int n, float* x, float* y)
{
  // set the total sum to 0
  __m128 x0 = _mm_set1_ps(0.);
  
  // we assume alignment
  assert(((std::size_t)x) % 16 == 0 && ((std::size_t)y) % 16 == 0);

  int ndiv4 = n/4;
  
  // loop over chunks of 4 values
  for (int i=0; i<ndiv4; ++i) {
    __m128 x1 = _mm_load_ps(x+4*i); // aligned (fast) load
    __m128 x2 = _mm_load_ps(y+4*i); // aligned (fast) load
    __m128 x3 = _mm_mul_ps(x1,x2);  // multiply
    x0 = _mm_add_ps(x0,x3);         // add
  }
  
  // store the 4 partial sums back to aligned memory
  float alignas(16) tmp[4];
  _mm_store_ps(tmp,x0);         
  float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3];
  
  // do the remaining entries
  for (int i=ndiv4*4 ; i< n ; ++i)
    sum += x[i]*y[i];
  
  return sum;
}
  

int main()
{
  // initialize two vectors
  std::vector<float> x(1000);
  std::vector<float> y(1000);

  for (int i=0; i<x.size(); ++i)
    x[i] = 1;

  for (int i=0; i<x.size(); ++i)
    y[i] = 2;

  // call saxpy
  float d = sdot(x.size(),&x[0],&y[0]);
  
  // calculate error
    std::cout << "error: " << d-2000. << "\n";
}

