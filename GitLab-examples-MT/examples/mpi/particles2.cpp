// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <iostream>
#include <vector>
#include <array>
#include <mpi.h>

const int dimensions=3;

typedef std::array<double,dimensions> coordinate;

struct particle {
  coordinate position;
  coordinate velocity;
};

// every rank creates particles and shares only the positions
// with all other ranks

int main(int argc, char** argv) {
  
  MPI_Init(&argc, &argv);
  int num;
  
  // assuming no padding is needed for three doubles  
  MPI_Datatype coordinate_type;
  MPI_Type_contiguous(dimensions, MPI_DOUBLE, &coordinate_type);
  MPI_Type_commit(&coordinate_type);

  // now create a type that picks out just the positions from the particle
  particle p;
  MPI_Aint p_lb, p_position, p_ub;
  MPI_Get_address(&p, &p_lb);
  MPI_Get_address(&p.position, &p_position);      
  MPI_Get_address(&p+1, &p_ub);      
  
  int          blocklens[] =  {0, 1, 0};
  MPI_Datatype types[]     =  {MPI_LB, coordinate_type, MPI_UB};
  MPI_Aint     offsets[]   =  {0, p_position-p_lb, p_ub-p_lb};
  
  MPI_Datatype position_in_particle_type;
  MPI_Type_create_struct(3, blocklens, offsets, types,&position_in_particle_type);
  MPI_Type_commit(&position_in_particle_type);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  
  
  // create some particles at some "random" locations
  const int particles_per_rank=10;
  std::vector<particle> particles(particles_per_rank);
  for (int i=0; i<particles.size(); ++i)
    for (int d=0 ; d<dimensions; ++d) {
      particles[i].position[d] = i+rank*10;
      particles[i].velocity[d] = 100+rank;
    }

  // now gather the locations of all particles
  
  std::vector<coordinate> allpositions(size*particles_per_rank);
  
  MPI_Allgather(&particles[0], particles_per_rank, position_in_particle_type,
                &allpositions[0], particles_per_rank, coordinate_type, MPI_COMM_WORLD);

  if(rank == 0) {
    // print all particles from one rank
    for (auto const& p: allpositions) {
      std::cout << "A particle is at ";
      for (double coord: p)
        std::cout << coord << " ";
      std::cout << "\n";
    }
  }

  MPI_Type_free(&position_in_particle_type);
  MPI_Type_free(&coordinate_type);
  MPI_Finalize();
  
  return 0;
}
