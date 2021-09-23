// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

// just saxpy to see what the compiler generates
void saxpy(int n, float a,  float* __restrict__ x, float* __restrict__ y)
{
  __builtin_assume_aligned(x,32);
  __builtin_assume_aligned(y,32);
  for (int i=0; i<n; ++i)
    y[i] += a*x[i];
}

