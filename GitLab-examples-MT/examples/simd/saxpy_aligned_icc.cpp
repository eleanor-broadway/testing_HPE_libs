// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>

#define N 100000
#define REPETITIONS 1000

// a saxpy version without explicit SSE
void saxpy(int n, float a,  float* restrict x, float* restrict y)
{
  #pragma simd
  #pragma vector aligned
  for (int i=0; i<n; ++i)
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

