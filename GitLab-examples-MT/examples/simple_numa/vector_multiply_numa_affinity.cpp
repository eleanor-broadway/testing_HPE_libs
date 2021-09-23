// Example codes for HPC course
// (c) 2012 Andreas Hehn, ETH Zurich
#include <vector>
#include <thread>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <pthread.h>
#include <unistd.h>
#include "timer.hpp"

int main(int argc, char* argv[]) {

    hpcse::timer<> exec_timing;
    if( argc != 3 ){
        std::cerr << "Usage: " << argv[0] << " <size in Mfloats> <n_threads>" << std::endl;
        return -1;
    }
    //
    // hardware configuration
    //
    int pagesize = getpagesize()/sizeof(float);
    std::cout<<"pagesize: " << pagesize;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    std::size_t size             = std::atoi(argv[1]);
    unsigned int const n_threads = std::atoi(argv[2]);
    size *= 1000000;

    std::size_t const size_per_thread = size/n_threads;

    std::vector<std::thread> threads;



    //
    // Allocation and init
    //
    std::size_t const padding = size_per_thread % pagesize == 0 ? 0 : pagesize - (size_per_thread % pagesize);
    float* x1;
    float* x2;
    int err0 = posix_memalign((void**)&x1,pagesize,(size_per_thread+padding) * n_threads * sizeof(float));
    int err1 = posix_memalign((void**)&x2,pagesize,(size_per_thread+padding) * n_threads * sizeof(float));
    if( err0 != 0 || err1 != 0) {
        std::cerr<<"Error in posix_memalign error code:"<<err0<<" and "<<err1<<std::endl;
        return -1;
    }

    std::cout<< " init = numa";
    for(unsigned int i=0; i < n_threads; ++i) {
        CPU_ZERO(&cpuset);
        threads.push_back( std::thread( [=]() {
                    for(std::size_t j=i*(size_per_thread+padding); j < (i+1)*(size_per_thread+padding); ++j) {
                        x1[j] = j < (i*(size_per_thread+padding)+size_per_thread) ? 1.0 : 0.0;
                        x2[j] = j < (i*(size_per_thread+padding)+size_per_thread) ? 1.1 : 0.0;
                    } } )
        );
        CPU_SET(i,&cpuset);
        pthread_t nh = threads.back().native_handle();
        int s = pthread_setaffinity_np(nh,sizeof(cpu_set_t), &cpuset);
        if(s != 0)
            std::cerr<<"Error in setaffinity"<<std::endl;
    }

    for(std::thread& t : threads)
        t.join();

    threads.clear();


    //
    // Main execution
    //
    exec_timing.start();
    for(unsigned int i=0; i < n_threads; ++i) {
        CPU_ZERO(&cpuset);
        threads.push_back(
            std::thread(
                [=]() {
                    for(int l = 0; l < 100; ++l) {
                        float* x1t = x1+i*(size_per_thread+padding);
                        float* x2t = x2+i*(size_per_thread+padding);
                        for( ; x1t < x1+i*(size_per_thread+padding)+size_per_thread; ++x1t, ++x2t)
                            *x1t *= *x2t;
                    }
                }
            )
        );
        CPU_SET(i,&cpuset);
        pthread_t nh = threads.back().native_handle();
        int s = pthread_setaffinity_np(nh,sizeof(cpu_set_t), &cpuset);
        if(s != 0)
            std::cerr<<"Error in setaffinity"<<std::endl;
    }

    for(std::thread& t : threads)
        t.join();

    exec_timing.stop();

    //
    // Print results
    // 
    double result = 0;
    for(unsigned int i=0; i < n_threads; ++i) {
        result = std::accumulate(x1+i*(size_per_thread+padding), x1+(i+1)*(size_per_thread+padding), result);
    }
    std::cout << "  result = " <<result;
    std::cout << "  size = "<<size_per_thread*n_threads << "  threads = " << n_threads <<"  time = "<< exec_timing.get_timing() << "s" << std::endl;
    delete[] x1;
    delete[] x2;
    return 0;
}
