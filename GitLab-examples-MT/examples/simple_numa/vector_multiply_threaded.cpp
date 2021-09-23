// Example codes for HPC course
// (c) 2012 Andreas Hehn, ETH Zurich
#include <vector>
#include <thread>
#include <iostream>
#include <numeric>
#include <cstdlib>
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

    std::size_t const size_per_thread = size/n_threads;

    std::vector<std::thread> threads;


    //
    // Allocation and init
    //
    float * const x1 = new float[size_per_thread * n_threads];
    float * const x2 = new float[size_per_thread * n_threads];
    std::cout<< "init = plain";
    for(std::size_t i=0; i < size_per_thread*n_threads; ++i) {
        x1[i] = 1.0;
        x2[i] = 1.1;
    }


    //
    // Main execution
    //
    exec_timing.start();
    for(unsigned int i=0; i < n_threads; ++i)
        threads.push_back(
            std::thread(
                [=]() {
                    for(int l = 0; l < 100; ++l) {
                        float* x1t = x1+i*size_per_thread;
                        float* x2t = x2+i*size_per_thread;
                        for( ; x1t < x1+(i+1)*size_per_thread; ++x1t, ++x2t)
                            *x1t *= *x2t;
                    }
                }
            )
        );

    for(std::thread& t : threads)
        t.join();

    exec_timing.stop();

    //
    // Print results
    // 
    double result = std::accumulate(x1, x1+size_per_thread*n_threads, 0.0);
    std::cout << "  result = " <<result;
    std::cout << "  size = "<<size_per_thread*n_threads << "  threads = " << n_threads <<"  time = "<< exec_timing.get_timing() << "s" << std::endl;
    delete[] x1;
    delete[] x2;
    return 0;
}
