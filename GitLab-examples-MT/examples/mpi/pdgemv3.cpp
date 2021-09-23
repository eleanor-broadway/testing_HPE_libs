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

  // block cyclic distribution for the matrix with sqrt(size) * sqrt (size)
  // blocks 

  int N=1024;
  int num_blocks = std::sqrt(size);;
  int block_size = N/std::sqrt(size);
  assert(size = num_blocks * num_blocks);
  assert(N % block_size == 0);
  
  // build a cartesian topology
  int periodic[2] = {false, false};
  int extents[2] = {num_blocks, num_blocks};
  MPI_Comm comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, extents, periodic, true, &comm);
  
  // get my row and column number
  int coords[2];
  MPI_Cart_coords(comm, rank, 2, coords);
  int row = coords[0];
  int col = coords[1];
  
  // build communicators for rows and columns
  MPI_Comm row_comm, col_comm, diag_comm;
  MPI_Comm_split(comm,row,col,&row_comm);
  MPI_Comm_split(comm,col,row,&col_comm);
  
  // block distribution of the vectors on the diagonal
  vector_type x(block_size), y(block_size), y0;
  if (row==col) {
    // intialize the vector x on the diagonals
    for (int i=0; i<block_size; ++i)
      x[i]=row*block_size+i;
    
    // calculate the reference result y0 on the diagonals
    y0.resize(block_size);
    for (int i=0; i<block_size ; i++)
      for (int j=0; j<N; j++)
        y0[i] += static_cast<double>(i+j+row*block_size) * static_cast<double>(j);
  }

  // allocate a block of the matrix everywhere and fill it in
  matrix_type A(block_size,block_size);
  
  for (int i=0; i<block_size; ++i)
    for (int j=0; j<block_size; ++j)
      A(i,j) = i+j+(row+col)*block_size;

  // do the multiplication:
  // 1. broadcast along columns
  MPI_Bcast(&x[0], x.size(), MPI_DOUBLE, col, col_comm);
  // 2. do local multiplication
  dgemv(A,x,y);
  // 3. reduce along row
  MPI_Reduce(row==col ? MPI_IN_PLACE : &y[0], &y[0], y.size(), MPI_DOUBLE, MPI_SUM, row, row_comm);
  
  // and now test
  double s = 0.;
  if (row==col)
    for (int j=0; j<block_size; j++)
      s = s + (y[j] - y0[j]) * (y[j] - y0[j]);
  s = std::sqrt(s);
  MPI_Allreduce(MPI_IN_PLACE, &s,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  if (rank==0)
    std::cout << "l2-error:     " << s << std::endl;
  
  MPI_Finalize();
}