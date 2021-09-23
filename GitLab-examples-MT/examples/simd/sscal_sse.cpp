// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <vector>
#include <iostream>
#include <cmath>
#include <x86intrin.h>

void sscal(int n, float a, float* x)
{
  // load the scale factor four times into a register
  __m128 x0 = _mm_load1_ps(&a);
  
  // cast the pointer to the appropriate integer type
  // to check the misalignment
  std::size_t xv = (std::size_t) x;
  int misalignedbytes = xv % 16;
  
  // a general unoptimized version if the misalignment is bad
  if (misalignedbytes%4 !=0 ) {
    int ndiv4 = n/4;
    // loop over chunks of 4 values
    for (int i=0; i<ndiv4; ++i) {
      __m128 x1 = _mm_loadu_ps(x+4*i); // unaligned (slow) load
      __m128 x2 = _mm_mul_ps(x0,x1);   // multiply
      _mm_storeu_ps(x+4*i,x2);        // store back unaligned (slow)
    }
    
    // do the remaining entries if the length was not a multiple of 4
    for (int i=ndiv4*4 ; i< n ; ++i)
      x[i] *= a;
  }
  // else do the optimized version for aligned values
  else {
    // do the values up to the alignment boundary
    int words_to_alignment = (4-xv/4)%4;
    for (int i=0; i < words_to_alignment ; ++i)
      x[i] *= a;
    
    float* y = x+words_to_alignment;
    int ndiv4 = (n-words_to_alignment)/4;

    // loop over chunks of 4 values
    for (int i=0; i<ndiv4; ++i) {
      _mm_prefetch((char*) y+4*i+8,_MM_HINT_NTA ); // prefetch data for two iterations later
      __m128 x1 = _mm_load_ps(y+4*i); // aligned (fast) load
      __m128 x2 = _mm_mul_ps(x0,x1);  // multiply
      _mm_store_ps(y+4*i,x2);         // store back aligned
    }
    
    // do the remaining entries, alternative with switch
    int i = ndiv4*4+words_to_alignment;
    switch (n-i) {
      case 3: x[i+2] *= a;
      case 2: x[i+1] *= a;
      case 1: x[i] *= a;
    }
  }
}
  

int main()
{
  // initialize a vector
  std::vector<float> x(999);
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

