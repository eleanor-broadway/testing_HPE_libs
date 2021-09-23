// Example codes for HPC course
// (c) 2012 Andreas Hehn and Matthias Troyer, ETH Zurich

#include "dvector.hpp"
#include "matrix.hpp"
#include <cmath>


extern "C" void dgemv_(char const& trans, int const& m, int const& n, double const& alpha, double* a, int const& lda, double* x, int const& incx, double const& beta, double* y, int const& incy);

typedef hpc12::matrix<double,hpc12::column_major> matrix_type;
typedef std::vector<double,hpc12::aligned_allocator<double,64> > vector_type;
inline void dgemv(matrix_type const& a, vector_type const& x, vector_type& y)
{
  dgemv_('N', a.size1(), a.size2(), 1., const_cast<double*>(a.data()), a.leading_dimension(), const_cast<double*>(&x[0]), 1, 0., &y[0], 1);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // we want a simple size that can be divided evenly by the number
  // of ranks to keep the code simple
  int N=1024;
  int block_size = N/size;
  assert(N % size ==0);
  
  // block distribution of the vectors
  dvector<double> x(N), y(N), y0(N);
  for (int i=0; i<block_size; ++i)
    x[i]=x.offset()+i;
  
  // block row distribution for the matrix: keep only N/size rows
  matrix_type A(block_size,N);
  
  for (int i=0; i<block_size; ++i)
    for (int j=0; j<N; ++j)
      A(i,j) = i+j+rank*block_size;

  //   // calculate the reference result y0
  for (int i=0; i<block_size ; i++)
    for (int j=0; j<N; j++)
      y0[i] += static_cast<double>(i+j+rank*block_size) * static_cast<double>(j);

  // We need to gather all pieces into a big vector and then do a multiplication
  // to get the result
  vector_type fullx(N);
  MPI_Allgather(x.data(), x.size(), MPI_DOUBLE, &fullx[0], x.size(), MPI_DOUBLE, MPI_COMM_WORLD);
  dgemv(A,fullx,y);
  
  // and now test
  double s = 0.;
  for (int j=0; j<block_size; j++)
    s = s + (y[j] - y0[j]) * (y[j] - y0[j]);
  s = std::sqrt(s);
  MPI_Allreduce(MPI_IN_PLACE, &s,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  if (rank==0)
    std::cout << "l2-error:     " << s << std::endl;
  
  MPI_Finalize();
}