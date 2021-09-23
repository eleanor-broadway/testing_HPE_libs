#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <omp.h>
#include "timer.hpp"

typedef float value_type;
typedef std::size_t size_type;

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
    {
        N_tot = N_*N_;
        
        /// real space grid spacing
        dr_ = (rmax_ - rmin_) / (N_ - 1);
        
        /// dt < dx*dx / (4*D) for stability
        dt_ = dr_ * dr_ / (6 * D_);
        
        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);
        
        rho_ = new value_type[N_tot];
        rho_tmp = new value_type[N_tot];
        
        InitializeSystem();
    }
    
    ~Diffusion2D()
    {
        delete[] rho_;
        delete[] rho_tmp;
    }
    
    void PropagateDensity(int steps);
    
    value_type GetMoment() {
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
    
    value_type *rho_, *rho_tmp;
};

void Diffusion2D::WriteDensity(const std::string file_name) const
{
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

#pragma omp parallel
    for(int s = 0; s < steps; ++s)
    {
#pragma omp for collapse(2)
        for(size_type i = 0; i < N_; ++i)
            for(size_type j = 0; j < N_; ++j)
                rho_tmp[i*N_ + j] =
                rho_[i*N_ + j]
                +
                fac_
                *
                (
                 (j == N_-1 ? 0 : rho_[i*N_ + (j+1)])
                 +
                 (j == 0 ? 0 : rho_[i*N_ + (j-1)])
                 +
                 (i == N_-1 ? 0 : rho_[(i+1)*N_ + j])
                 +
                 (i == 0 ? 0 : rho_[(i-1)*N_ + j])
                 -
                 4*rho_[i*N_ + j]
                 );
        
#pragma omp single
        {
            swap(rho_tmp,rho_);
            
            time_ += dt_;
        }
    }
}

void Diffusion2D::InitializeSystem()
{
    time_ = 0;
    
    /// initialize rho(x,y,t=0)
    value_type bound = 1./2;
    
#pragma omp parallel for collapse(2)
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
}

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " <log2(size)>" << std::endl;
        return 1;
    }

#pragma omp parallel
    {
#pragma omp master
        std::cout << "Running with " << omp_get_num_threads() << " threads" << std::endl;
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
