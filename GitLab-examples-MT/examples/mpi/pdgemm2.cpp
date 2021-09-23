// Example codes for HPC course
// (c) 2012 Andreas Hehn and Matthias Troyer, ETH Zurich

#include "dvector.hpp"
#include "matrix.hpp"
#include <cmath>

extern "C" void dgemm_(char const& transa, char const& transb, int const& m, int const& n, int const& k, double const& alpha, double* a, int const& lda, double* b, int const& ldb, double const& beta, double* c, int const& ldc);

typedef hpc12::matrix<double,hpc12::column_major> matrix_type;
typedef std::vector<double,hpc12::aligned_allocator<double,64> > vector_type;

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
  int q = std::sqrt(size);;
  int block_size = N/std::sqrt(size);
  assert(size = q * q);
  assert(N % block_size == 0);
  
  // build a cartesian topology
  int periodic[2] = {false, false};
  int extents[2] = {q, q};
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
  
  // allocate a block of the matrix everywhere and fill it in
  matrix_type A(block_size,block_size);
  matrix_type B(block_size,block_size);
  matrix_type C(block_size,block_size);
  
  for (int i=0; i<block_size; ++i)
    for (int j=0; j<block_size; ++j) {
      A(i,j) = i+j+(row+col)*block_size;
      B(i,j) = i+j+(row+col)*block_size;
      C(i,j) = 0.;
    }

  // allocate working space for the block row of A
  // and the block column of B
  
  matrix_type Atmp(block_size,block_size);
  matrix_type Btmp(block_size,block_size);
  
  // loop over all block
  for (int i=0; i<q; ++i) {
    // 1. broadcast block along row and column
    if (i==col)
      Atmp=A;
    if (i==row)
      Btmp=B;
    MPI_Bcast(Atmp.data() ,block_size*block_size,MPI_DOUBLE,i,row_comm);
    MPI_Bcast(Btmp.data() ,block_size*block_size,MPI_DOUBLE,i,col_comm);
  
    // 2. do all multiplications
    dgemm_('N','N',block_size,block_size,block_size,1.,
            Atmp.data(),block_size,Btmp.data(),block_size,
            1., C.data(),block_size);
  }
  
  MPI_Finalize();
}