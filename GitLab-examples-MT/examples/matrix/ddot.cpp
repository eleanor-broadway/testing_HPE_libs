// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <vector>
#include <iostream>

extern "C" double ddot_(int& n, double *x, int& incx, double *y, int& incy);

int main()
{
  std::vector<double> x(10, 1.); // intialize a vector with ten 1s
  std::vector<double> y(10, 2.); // intialize a vector with ten 2s
  
  // calculate the inner product
  
  int n=x.size();
  int one = 1;
  double d = ddot_(n,&x[0],one,&y[0],one);
  std:: cout << d << "\n"; // should be 20
}