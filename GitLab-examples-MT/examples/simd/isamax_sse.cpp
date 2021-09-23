// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich
// adapated from code by Wesley P. Petersen published first in
// Petersen and Arbenz "Intro. to Parallel Computing," Oxford Univ. Press, 2004

#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include "xmmintrin.h"


int isamax0(int n, float *x)
{
  // assume alignment
  assert( ((std::size_t) x % 16) == 0);
  
  __m128 V7      = _mm_set_ps(3.0,2.0,1.0,0.0);
  __m128 V2      = _mm_set_ps(3.0,2.0,1.0,0.0);
  __m128 V6      = _mm_set_ps1(-0.0);
  __m128 offset4 = _mm_set_ps1(4.0);
  __m128 V3;
  
  float *xp=x;
  int nsegs = (n >> 2) - 2;
  int eres  = n - 4*(nsegs+2);
  __m128 V0 = _mm_load_ps(xp); xp += 4;   // first four in 4/time seq.
  __m128 V1 = _mm_load_ps(xp); xp += 4;   // next four in 4/time seq.
  V0 = _mm_andnot_ps(V6,V0);             // take absolute value
  for(int i=0; i<nsegs; i++){
    V1 = _mm_andnot_ps(V6,V1);    // take absolute value
    V3 = _mm_cmpnle_ps(V1,V0);    // compare old max of 4 to new
    int mb = _mm_movemask_ps(V3); // any of 4 bigger?
    V2 = _mm_add_ps(V2,offset4);  // add offset
    if(mb > 0){
      V0 = _mm_max_ps(V0,V1);     // get the new maxima
      V3 = _mm_and_ps(V2,V3);     // the index if the element was bigger or 0 otherwise
      V7 = _mm_max_ps(V7,V3);     // get the maxima of the old and new indices
    }
    V1 = _mm_load_ps(xp); xp += 4;  // bottom load next four
  }
  // finish up the last segment of 4
  V1 = _mm_andnot_ps(V6,V1);    // take absolute value
  V3 = _mm_cmpnle_ps(V1,V0);    // compare old max of 4 to new
  int mb = _mm_movemask_ps(V3); // any of 4 bigger?
  V2 = _mm_add_ps(V2,offset4);  // add offset
  if(mb > 0){
    V0 = _mm_max_ps(V0,V1);
    V3 = _mm_and_ps(V2,V3);
    V7 = _mm_max_ps(V7,V3);
  }
  // Now finish up: segment maxima are in V0, indices in V7
  float xbig[8], indx[8];
  _mm_store_ps(xbig,V0);
  _mm_store_ps(indx,V7);
  // add remaining numbers
  if(eres>0){
    for(int i=0; i<eres; i++){
      xbig[4+i] = fabsf(*(xp++));
      indx[4+i] = (float) (n+i);
    }
  }
  float ebig  = 0.;
  int iebig = 0;
  for(int i=0; i<4+eres; i++){
    if(xbig[i] > ebig){
      ebig = xbig[i];
      iebig = (int) indx[i];
    }
  }
  return iebig;
}



main()
{
  std::vector<float> x(20);
  for(int i=0; i<x.size(); i++)
    x[i] = -2.0 + i;

  x[17] =33500.0;
  int im = isamax0(x.size(),&x[0]);
  std::cout << " maximum index = " << im << "\n";
  std::cout << " maximum value = " << x[im] << "\n";
}
