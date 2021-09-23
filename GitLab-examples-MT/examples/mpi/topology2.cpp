// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <mpi.h>

int main(int argc, char** argv)
{
  // now initialize MPI and get information about the number of processes
  
  MPI_Init(&argc,&argv);
  
  int size;
  int rank;
  MPI_Status status;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  
  int nums[3] = {0,0,0};
  int periodic[3] = {false, false, false};
  
  // split the nodes automatically
  MPI_Dims_create(size, 3, nums);
  
  if (rank==0)
    std::cout << "We create a " << nums[0] << "x" << nums[1] << "x" << nums[2] << " arrangement.\n";

  // now everyone creates a a cartesian topology
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 3, nums, periodic, true, &cart_comm);
  
  // now get the neighors
  
  int left, right, bottom, top, front, back, newrank;
  
  MPI_Comm_rank(cart_comm,&newrank);

  MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
  MPI_Cart_shift(cart_comm, 1, 1, &bottom, &top);
  MPI_Cart_shift(cart_comm, 2, 1, &front, &back);
  
  std::cout << "Rank " << rank << " has new rank " << newrank << " and neighbors "
            << left << ", " << right << ", " << top << ", " << bottom << ", "
            << front << ", " << back << std::endl;
  
  MPI_Comm_free(&cart_comm);
  MPI_Finalize();
 }

