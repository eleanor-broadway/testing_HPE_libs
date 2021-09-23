#include <iostream>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include "timer.hpp"

typedef float value_type;
typedef std::size_t size_type;

static const int diffusion_block_x = 16;
static const int diffusion_block_y = 16;

__global__ void diffusion_kernel(value_type * rho_out, value_type const * rho, value_type fac, int N)
{
    __shared__ value_type rho_loc[(diffusion_block_x+2)*(diffusion_block_y+2)];
    int const gj = blockIdx.x*blockDim.x + threadIdx.x;
    int const gi = blockIdx.y*blockDim.y + threadIdx.y;

    int const lN = diffusion_block_y+2;
    int const lj = threadIdx.x + 1;
    int const li = threadIdx.y + 1;

    if(gi < N && gj < N)
    {
        // Load the bulk
        rho_loc[li*lN + lj] = rho[gi*N + gj];

        // Load the ghost cells
        if(threadIdx.y == 0)
        {
            rho_loc[(li-1)*lN + lj] = (gi == 0 ? 0 : rho[(gi-1)*N + gj]);
        }
        if(threadIdx.y == blockDim.y-1)
        {
            rho_loc[(lN-1)*lN + lj] = (gi == N-1 ? 0 : rho[(gi+1)*N + gj]);
        }
        if(threadIdx.x == 0)
        {
            rho_loc[li*lN + lj-1] = (gj == 0 ? 0 : rho[gi*N + gj-1]);
        }
        if(threadIdx.x == blockDim.x-1)
        {
            rho_loc[li*lN + lN-1] = (gj == N-1 ? 0 : rho[gi*N + gj+1]);
        }
    }
    __syncthreads();

    if(gi < N && gj < N)
    {
        rho_out[gi*N + gj] = rho_loc[li*lN + lj] + fac
            *
            (  rho_loc[li*lN + (lj+1)]
             + rho_loc[li*lN + (lj-1)]
             + rho_loc[(li+1)*lN + lj]
             + rho_loc[(li-1)*lN + lj]
             - 4*rho_loc[li*lN + lj]
             );
    }
}

class Diffusion2D
{
    
public:
    
    Diffusion2D(
                const value_type D,
                const value_type rmax,
                const value_type rmin,
                const size_type N
                )
    : D_(D)
    , rmax_(rmax)
    , rmin_(rmin)
    , N_(N)
    , N_tot(N*N)
    , d_rho_(0)
    , d_rho_tmp_(0)
    , rho_(N_tot)
    {
        /// real space grid spacing
        dr_ = (rmax_ - rmin_) / (N_ - 1);
        
        /// dt < dx*dx / (4*D) for stability
        dt_ = dr_ * dr_ / (6 * D_);
        
        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);

        // Allocate memory on Device
        cudaMalloc(&d_rho_, N_tot*sizeof(value_type));
        cudaMalloc(&d_rho_tmp_, N_tot*sizeof(value_type));
        
        cudaMemset(d_rho_,0,N_tot);
        cudaMemset(d_rho_tmp_,0,N_tot);
        
        InitializeSystem();
    }
    
    ~Diffusion2D()
    {
        cudaFree(d_rho_tmp_);
        cudaFree(d_rho_);
    }
    
    void PropagateDensity(int steps);
    
    value_type GetMoment() {
        cudaMemcpy(&rho_[0], d_rho_, rho_.size() * sizeof(value_type), cudaMemcpyDeviceToHost);
        value_type sum = 0;
        
        for(size_type i = 0; i < N_; ++i)
            for(size_type j = 0; j < N_; ++j) {
                value_type x = j*dr_ + rmin_;
                value_type y = i*dr_ + rmin_;
                sum += rho_[i*N_ + j] * (x*x + y*y);
            }
        
        return dr_*dr_*sum;
    }
    
    value_type GetTime() const {return time_;}
    
    void WriteDensity(const std::string file_name) const;
    
private:
    
    void InitializeSystem();
    
    const value_type D_, rmax_, rmin_;
    const size_type N_;
    size_type N_tot;
    
    value_type dr_, dt_, fac_;
    
    value_type time_;
    
    value_type *d_rho_, *d_rho_tmp_;
    mutable std::vector<value_type> rho_;
};

void Diffusion2D::WriteDensity(const std::string file_name) const
{
    // Get data from device
    cudaMemcpy(&rho_[0], d_rho_, rho_.size() * sizeof(value_type), cudaMemcpyDeviceToHost);
    std::ofstream out_file;
    out_file.open(file_name.c_str(), std::ios::out);
    if(out_file.good())
    {
        for(size_type i = 0; i < N_; ++i){
            for(size_type j = 0; j < N_; ++j)
                out_file << (i*dr_+rmin_) << '\t' << (j*dr_+rmin_) << '\t' << rho_[i*N_ + j] << "\n";
            
            out_file << "\n";
        }
    }
    
    out_file.close();
}

void Diffusion2D::PropagateDensity(int steps)
{
    using std::swap;
    /// Dirichlet boundaries; central differences in space, forward Euler
    /// in time

    dim3 block_size(diffusion_block_x,diffusion_block_y,1);
    dim3 grid_size((N_+diffusion_block_x-1)/diffusion_block_x,(N_+diffusion_block_y-1)/diffusion_block_y,1); // Round-up needed number of blocks (N/block_size)
    for(int s = 0; s < steps; ++s)
    {
        diffusion_kernel<<<grid_size, block_size>>>(d_rho_tmp_, d_rho_, fac_, N_);
        swap(d_rho_, d_rho_tmp_);
        time_ += dt_;
    }
}

void Diffusion2D::InitializeSystem()
{
    time_ = 0;
    
    /// initialize rho(x,y,t=0)
    value_type bound = 1./2;
    
    for(size_type i = 0; i < N_; ++i){
        for(size_type j = 0; j < N_; ++j){
            if(std::fabs(i*dr_+rmin_) < bound && std::fabs(j*dr_+rmin_) < bound){
                rho_[i*N_ + j] = 1;
            }
            else{
                rho_[i*N_ + j] = 0;
            }
            
        }
    }
    cudaMemcpy(d_rho_, &rho_[0], rho_.size() * sizeof(value_type), cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " <log2(size)>" << std::endl;
        return 1;
    }

    const value_type D = 1;
    const value_type tmax = 0.01;
    const value_type rmax = 1;
    const value_type rmin = -1;
    
    const size_type N_ = 1 << std::atoi(argv[1]);
    const int steps_between_measurements = 100;
    
    Diffusion2D System(D, rmax, rmin, N_);
    
    value_type time = 0;
    
    timer runtime;
    runtime.start();
    
    while(time < tmax){
        System.PropagateDensity(steps_between_measurements);
        time = System.GetTime();
        value_type moment = System.GetMoment();
        std::cout << time << '\t' << moment << std::endl;
    }
    
    runtime.stop();
    
    double elapsed = runtime.get_timing();
    
    std::cerr << argv[0] << "\t N=" <<N_ << "\t time=" << elapsed << "s" << std::endl;
    
    std::string density_file = "Density.dat";
    System.WriteDensity(density_file);
    
    return 0;
}
