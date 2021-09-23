#include <iostream>
#include <cassert>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include "matrix.hpp"
#include <mpi.h>

#define N 16384

struct world_info {int size, rank, dims[2];} world;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    /// timings
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

    
    //part a
    
    MPI_Comm_size(MPI_COMM_WORLD, &world.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world.rank);
    MPI_Dims_create(world.size, 2, world.dims);
    
    assert(N % world.dims[0] == 0 && N % world.dims[1] == 0);
    
    const std::size_t Nx_loc(N / world.dims[0]);
    const std::size_t Ny_loc(N / world.dims[1]);
    
    if (world.rank == 0) {
    	std::cout << "Proc grid : " << world.dims[0] << " x " <<  world.dims[1] << std::endl;
        std::cout << "Local size : " << Nx_loc << " x " << Ny_loc << std::endl;
    }
    
    hpcse::matrix<double, hpcse::row_major> A_loc(Nx_loc,Ny_loc);
    
    //part b
    
    int coords[2];
    int periods[2] = {false, false};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, world.dims, periods, false, &cart_comm);
    
    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    for(unsigned i = 0; i < Nx_loc; ++i)
        for(unsigned j = 0; j < Ny_loc; ++j){
            const unsigned i0(Nx_loc*coords[0]+i);
            const unsigned j0(Ny_loc*coords[1]+j);
            A_loc(i,j) = double(std::min(i0+1,j0+1)) / std::max(i0+1,j0+1);
        }
    
    //part c
    
    MPI_Comm row_comm, column_comm;
    MPI_Comm_split(cart_comm, coords[0], coords[1], &row_comm);
    MPI_Comm_split(cart_comm, coords[1], coords[0], &column_comm);
    
    std::vector<double> sums_loc(Nx_loc);
    for(unsigned i = 0; i < Nx_loc; ++i)
        sums_loc[i] = std::accumulate(&A_loc(i,0), &A_loc(i,0)+Ny_loc, 0.0);
    
    std::vector<double> sums_row(Nx_loc,0.0);
    MPI_Reduce(sums_loc.data(),sums_row.data(),Nx_loc,MPI_DOUBLE,MPI_SUM,0,row_comm);
    
    if(coords[1] == 0){
        int root;
        int root_coords[2] = {0,0};
        MPI_Cart_rank(cart_comm,root_coords,&root);
        
        double max_strip(*std::max_element(sums_row.begin(),sums_row.end()));
        
        double L_inf(0.0);
        MPI_Reduce(&max_strip,&L_inf,1,MPI_DOUBLE,MPI_MAX,root,column_comm);
        
        if(rank == root)
            std::cout << "L_inf norm: " << L_inf << std::endl;
    }
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&column_comm);
    MPI_Comm_free(&cart_comm);
    
    end = std::chrono::high_resolution_clock::now();
    int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    if (rank == 0)
        std::cout << "Timing " << N << " " << world.size << " " << elapsed_time << "ms" << std::endl;
    
    MPI_Finalize();
    
    return 0;
}
