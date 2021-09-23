// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <iostream>
#include <vector>
#include <array>
#include <mpi.h>

const int dimensions=3;

struct particle {
  std::array<double,dimensions> position;
  std::array<double,dimensions> velocity;
};

int main(int argc, char** argv) {
  
  MPI_Init(&argc, &argv);
  int num;
  
  // assuming that no padding is needed for six doubles
  MPI_Datatype particle_type;
  MPI_Type_contiguous(2*dimensions, MPI_DOUBLE, &particle_type);
  MPI_Type_commit(&particle_type);
  
  MPI_Comm_rank(MPI_COMM_WORLD,&num);
  
  if(num==0) {
    // receive data into a vector and print it
    std::vector<particle> particles(10);
    MPI_Status status;
    // receive up to 10 particles
    MPI_Recv(&particles[0], particles.size(), particle_type,
             1, 42, MPI_COMM_WORLD, &status);
    
    // get the number of actually received particles and resize
    int num;
    MPI_Get_count(&status, particle_type, &num);
    particles.resize(num);
    
    for (auto const& p: particles) {
      std::cout << "A particle is at ";
      for (double coord: p.position)
        std::cout << coord << " ";
      std::cout << "\n";
    }
  }
  else {
    // create some particles at rest in a diagonal line
    std::vector<particle> particles(10);
    for (int i=0; i<particles.size(); ++i)
      for (int d=0 ; d<dimensions; ++d)
        particles[i].position[d] = i;
    
    // create an MPI datatype for particle 2,4 and 7
    MPI_Datatype indexed_type;
    int indices[] = {2,4,7};
    MPI_Type_create_indexed_block(3, 1, indices, particle_type, &indexed_type);
    MPI_Type_commit(&indexed_type);
    
    // and send those three
    MPI_Send(&particles[0], 1, indexed_type, 0, 42, MPI_COMM_WORLD);

    MPI_Type_free(&indexed_type);

  }

  MPI_Type_free(&particle_type);

  MPI_Finalize();
  
  return 0;
}
