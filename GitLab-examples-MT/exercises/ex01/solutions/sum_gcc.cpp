#include <vector>
#include <numeric>
#include <iostream>
#include "timer.hpp"
#include "aligned_allocator.hpp"


int main( int argc, char** argv )
{
    // repetitions
    const int nrep = 10000;
    // vector size
    const int n = 1<<16;
    
    // initialize 16 byte aligned vectors
    std::vector< float, hpcse::aligned_allocator<float,16> > x(n,-1.2), y(n,3.4), z(n);
    
    /// TODO: fix. remove warning about aliasing.
    hpcse::timer<> tim;
    tim.start();
    for (int k = 0; k < nrep; ++k)
    {
        for( int i = 0; i < n; ++i )
        {
            z[i] = x[i]*x[i] + y[i]*y[i] + 2.*x[i]*y[i];
        }
    }
    tim.stop();
    
    // print result checksum
    std::cout << std::accumulate(z.begin(), z.end(), 0.) << std::endl;
    std::cout << "Task took " << tim.get_timing() << " seconds." << std::endl;
}

