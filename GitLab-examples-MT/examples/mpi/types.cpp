// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <mpi.h>
#include "matrix.hpp"

int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  
  hpc12::matrix<double,hpc12::column_major> a(4,4);
  
  MPI_Datatype row, col;
  
  MPI_Type_contiguous(4, MPI_DOUBLE,&col);
  MPI_Type_vector(4,1,4,MPI_DOUBLE,&row);
  
  MPI_Type_commit(&row);
  MPI_Type_commit(&col);
  
  // use them
  // ...
  // and finally free them
  
  MPI_Type_free(&row);
  MPI_Type_free(&col);
 
  
  MPI_Finalize();
  return 0;
}
