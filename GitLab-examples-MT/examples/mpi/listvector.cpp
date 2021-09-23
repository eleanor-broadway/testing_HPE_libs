// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <iostream>
#include <list>
#include <vector>
#include <mpi.h>

int main(int argc, char** argv) {
  
  MPI_Init(&argc, &argv);
  int num;
  
  MPI_Comm_rank(MPI_COMM_WORLD,&num);
  
  if(num==0) {
    // receive data into a vector and print it
    std::vector<int> data(10);
    MPI_Status status;
    MPI_Recv(&data[0], 10, MPI_INT,
            1, 42, MPI_COMM_WORLD, &status);
    for (int i=0; i < data.size(); ++i)
      std:: cout << data[i] << "\n";
  }
  else {
    // fill a list with the numbers 0-9
    std::list<int> data;
    for (int i=0; i<10; ++i)
      data.push_back(i);
    
    std::vector<MPI_Datatype> types(10,MPI_INT);
    std::vector<int>          blocklens(10,1);
    std::vector<MPI_Aint>     offsets;
   
    for (int& x : data) {
      MPI_Aint address;
      MPI_Get_address(&x, &address);
      offsets.push_back(address);
    }
    
    MPI_Datatype list_type;
    MPI_Type_create_struct(10, &(blocklens[0]), &offsets[0], &types[0] ,&list_type);
    MPI_Type_commit(&list_type);
    MPI_Send(MPI_BOTTOM, 1, list_type, 0, 42, MPI_COMM_WORLD);
    MPI_Type_free(&list_type);
  }
  
  MPI_Finalize();
  
  return 0;
}
