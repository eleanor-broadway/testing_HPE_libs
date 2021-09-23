#include <iostream>
#include <cassert>
#include <vector>
#include <numeric>
#include <algorithm>
#include "matrix.hpp"

#define N 16384

int main(int argc, char* argv[])
{
  hpcse::matrix<double, hpcse::row_major> A(N,N);
  for(unsigned i = 0; i < N; ++i)
    for(unsigned j = 0; j < N; ++j)
      A(i,j) = double(std::min(i+1,j+1)) / std::max(i+1,j+1);

  std::vector<double> sums_row(N);
  for(unsigned i = 0; i < N; ++i)
    sums_row[i] = std::accumulate(&A(i,0), &A(i,0) + N, 0.0);

  const double L_inf(*std::max_element(sums_row.begin(),sums_row.end()));

  std::cout << "L_inf norm: " << L_inf << std::endl;

  return 0;
}
