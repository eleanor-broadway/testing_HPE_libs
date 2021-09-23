// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <iostream>
#include <string>
#include <stdexcept>
#include <mpi.h>

// throw a std::runtime_error if there was an MPI error
void check_error(int err)
{
  if (err != MPI_SUCCESS) {
    // get the error text for the error code
    int len = MPI_MAX_ERROR_STRING;
    char txt[MPI_MAX_ERROR_STRING];
    MPI_Error_string(err, txt, &len);
    // and throw an exception
    throw std::runtime_error(txt);
  }
}

int main(int argc, char** argv) {
  
  MPI_Init(&argc, &argv);
  int num;
  
  MPI_Comm_rank(MPI_COMM_WORLD,&num);
  
  // tell MPI to return an error code instead of aborting
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  
  try {
    if(num==0) { // "master"
      MPI_Status status;
      char txt[100];
      check_error(MPI_Recv(txt, 100, MPI_CHAR,
              1, 42, MPI_COMM_WORLD, &status));
      std::cout << txt << "\n";
    }
    else { // "worker"
      std::string text="Hello world!";
      check_error(MPI_Send(const_cast<char*>(text.c_str()), text.size()+1, MPI_CHAR,
               0, 42, MPI_COMM_WORLD));
    }
    
    MPI_Finalize();
  }
  catch (std::runtime_error& e)
  {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    // abort the program
    MPI_Abort(MPI_COMM_WORLD,-1);
  }
  return 0;
}
