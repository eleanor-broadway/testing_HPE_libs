// Example codes for HPC course
// (c) 2012 Andreas Hehn, ETH Zurich
#include <vector>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <omp.h>
#include "timer.hpp"

int main(int argc, char* argv[]) {

    hpcse::timer<> exec_timing;
    if( argc != 3 ){
        std::cerr << "Usage: " << argv[0] << " <size in Mfloats> <n_threads>" << std::endl;
        return -1;
    }
    std::size_t size             = std::atoi(argv[1]);
    unsigned int const n_threads = std::atoi(argv[2]);
    size *= 1000000;

    omp_set_num_threads(n_threads);

    std::size_t const size_per_thread = size/n_threads;


    //
    // Allocation and init
    //
    float * const x1 = new float[size_per_thread * n_threads];
    float * const x2 = new float[size_per_thread * n_threads];
    std::cout<< "init = numa";
    std::size_t const real_size = size_per_thread * n_threads;
#pragma omp parallel
#pragma omp for schedule(static)
    for(std::size_t i=0; i < real_size; ++i) {
        x1[i] = 1.0;
        x2[i] = 1.1;
    }


    //
    // Main execution
    //
    exec_timing.start();
#pragma omp parallel
    for(int l = 0; l < 100; ++l) {
#pragma omp for schedule(static) nowait
        for(std::size_t i=0; i < real_size; ++i) {
            x1[i] *= x2[i];
        }
    }

    exec_timing.stop();

    //
    // Print results
    // 
    double result = std::accumulate(x1, x1+size_per_thread*n_threads, 0.0);
    std::cout << "  result = " <<result;
    std::cout << "  size = "<<size_per_thread*n_threads << "  omp_threads = " << omp_get_max_threads() <<"  time = "<< exec_timing.get_timing() << "s" << std::endl;
    delete[] x1;
    delete[] x2;
    return 0;
}
