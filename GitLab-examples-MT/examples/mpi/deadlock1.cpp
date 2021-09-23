// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <iostream>
#include <string>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Status status;
  int num;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&num);
  
  double ds=3.1415927; // to send
  double dr;           // to receive
  int tag=99;
  
  if(num==0) {
    MPI_Ssend(&ds,1,MPI_DOUBLE,1,tag,MPI_COMM_WORLD);
    MPI_Recv (&dr,1,MPI_DOUBLE,1,tag,MPI_COMM_WORLD,&status);
  }
  else {
    MPI_Ssend(&ds,1,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
    MPI_Recv (&dr,1,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);
  }
  
  MPI_Finalize();
  return 0;
}
