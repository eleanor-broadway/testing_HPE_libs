// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

// just saxpy to see what the compiler generates
void saxpy(int n, float a,  float* restrict x, float* restrict y)
{
  #pragma simd
  #pragma vector aligned
  for (int i=0; i<n; ++i)
    y[i] += a*x[i];
}

