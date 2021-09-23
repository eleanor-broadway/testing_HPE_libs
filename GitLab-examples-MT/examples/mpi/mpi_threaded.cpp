// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich


#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  switch (provided)
  {
    case MPI_THREAD_SINGLE:
      std::cout << "MPI_THREAD_SINGLE is provided\n";
      break;
    case MPI_THREAD_FUNNELED:
      std::cout << "MPI_THREAD_FUNNELED is provided\n";
    break;
    case MPI_THREAD_SERIALIZED:
      std::cout << "MPI_THREAD_SERIALIZED is provided\n";
      break;
    case MPI_THREAD_MULTIPLE:
      std::cout << "MPI_THREAD_MULTIPLE is provided\n";
      break;
    default:
      std::cout << "Unknown level " << provided << "\n";
  }
  
  MPI_Finalize();
  return 0;
}