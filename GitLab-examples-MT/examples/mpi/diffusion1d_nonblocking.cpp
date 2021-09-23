// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich


#include <vector>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <cassert>
#include <mpi.h>

int main(int argc, char** argv)
{
  double L  = 10000.;  // extent of the domain
  double dx = 0.1;  // spatial resolution
  double dt = 0.005;  // time step
  double c  = 1.;    // diffusion constant

  double coefficient = c*dt/dx/dx;
  
  unsigned iterations = 10000; // number of iterations
  unsigned N          = L/dx;  // number of mesh points
  
  // now initialize MPI and get information about the number of processes
  
  MPI_Init(&argc,&argv);
  
  int size;
  int rank;
  MPI_Status status[4];
  MPI_Request reqs[4];
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 
  // get neighbors and assume periodic boundary conditions;
  int left = (rank > 0 ? rank-1 : size-1);
  int right = (rank < size-1 ? rank+1 : 0);

  // get number of points to be used on this process
  int points_per_rank =  N/size;
  int local_N = points_per_rank;
  // take care of roundoff issues for the last segment
  // if N is not a multiple of size
  if (rank==size-1)
    local_N = N-(size-1)*points_per_rank;
  
  // add space for the two ghost cells on the left and right end
  local_N+=2;
  std::vector<double> density(local_N);
  std::vector<double> newdensity(local_N);
  
  std::cout << "Rank " << rank << " has " << local_N << " points\n";
  
  // set the density to 1 in the middle
  // we need to shift and limit to our range
  int mystart = rank*points_per_rank;
  int start1 = N/4-mystart+1; // take care of ghost cell offset
  int end1 = N/4-mystart+1;   // take care of ghost cell offset
  
  if (start1 < local_N && end1 >= 0)
    std::fill(density.begin()+std::max(0,start1),
              density.begin()+std::min(end1,local_N),1.);

  for (int t=0; t<iterations; ++t) {
    // first start the communication to get  the ghost cells and send
    // our boundary values to the neighbor for their ghost cells
    
    // avoid deadlocks by a clear ordering who sends and receives first
    // even ones send left and odd ones receive right
    // even ones receive left and odd ones send right
    // even ones send right and odd ones receive left
    // even ones receive right and odd ones send left
    
    // make sure we have an even number of ranks for this to work
    assert(size %2 == 0);
    if (rank % 2 == 0) {
      MPI_Isend(&density[1],1,MPI_DOUBLE,left,0,MPI_COMM_WORLD,&reqs[0]);
      MPI_Irecv(&density[0],1,MPI_DOUBLE,left,0,MPI_COMM_WORLD,&reqs[1]);
      MPI_Isend(&density[local_N-2],1,MPI_DOUBLE,right,0,MPI_COMM_WORLD,&reqs[2]);
      MPI_Irecv(&density[local_N-1],1,MPI_DOUBLE,right,0,MPI_COMM_WORLD,&reqs[3]);
    }
    else {
      MPI_Irecv(&density[local_N-1],1,MPI_DOUBLE,right,0,MPI_COMM_WORLD,&reqs[0]);
      MPI_Isend(&density[local_N-2],1,MPI_DOUBLE,right,0,MPI_COMM_WORLD,&reqs[1]);
      MPI_Irecv(&density[0],1,MPI_DOUBLE,left,0,MPI_COMM_WORLD,&reqs[2]);
      MPI_Isend(&density[1],1,MPI_DOUBLE,left,0,MPI_COMM_WORLD,&reqs[3]);
    }

    // do calculation  of the interior
    for (int i=2; i<local_N-2;++i)
      newdensity[i] = density[i] + coefficient * (density[i+1]+density[i-1]-2.*density[i]);
    
    // wait for the ghost cells to arrive
    MPI_Waitall(4,reqs,status);
    
    // do the boundaries
    newdensity[1] = density[1] + coefficient * (density[2]+density[0]-2.*density[1]);
    newdensity[local_N-2] = density[local_N-2] + coefficient * (
                               density[local_N-1]+density[local_N-3]-2.*density[local_N]);
    
    // and swap
    density.swap(newdensity);
  }
  
  MPI_Finalize();
 }

