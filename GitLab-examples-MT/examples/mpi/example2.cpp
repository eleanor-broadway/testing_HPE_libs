// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <iostream>
#include <string>
#include <mpi.h>

int main(int argc, char** argv) {
  
  MPI_Init(&argc, &argv);
  int num;
  
  MPI_Comm_rank(MPI_COMM_WORLD,&num);
  
  if(num==0) { // "master"
    MPI_Status status;
    char txt[100];
    MPI_Recv(txt, 100, MPI_CHAR,
            1, 42, MPI_COMM_WORLD, &status);
    std::cout << txt << "\n";
  }
  else { // "worker"
    std::string text="Hello world!";
    MPI_Send(const_cast<char*>(text.c_str()), text.size()+1, MPI_CHAR,
             0, 42, MPI_COMM_WORLD);
  }
  
  MPI_Finalize();
  
  return 0;
}
