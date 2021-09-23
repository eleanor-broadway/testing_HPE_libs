#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <numeric>
#include <iterator>
#include <cassert>

#include "aligned_allocator.hpp"
using hpcse::aligned_allocator;

typedef std::size_t size_type;
typedef float scalar_type;
// aligned to the cacheline size
typedef std::vector<scalar_type, aligned_allocator<scalar_type,64> > positions_type;

/// parameters
const size_type   N   = 1<<16; // system size
const scalar_type eps = 5.;    // Lenard-Jones, eps
const scalar_type rm  = 0.1;   // Lenard-Jones, r_m


/// compute the Lennard-Jones force particle at position x0
scalar_type compute_force(positions_type const& positions, scalar_type x0, scalar_type rc)
{
    scalar_type rm2 = rm * rm;
    scalar_type force = 0.;
    for (size_type i=0; i<N; ++i) {
        scalar_type r  = x0 - positions[i];
        scalar_type rinv = (r < rc) / r;
        scalar_type rinv2 = rinv * rinv;
        scalar_type s2 = rm2 * rinv2;  // (rm/r)^2
        scalar_type s6 = s2*s2*s2;     // (rm/r)^6
        force += 12*eps * (s6*s6 - s6) * rinv;
    }
    return force;
}

int main(int argc, const char** argv)
{
    // As icc 13 does not provide a fully working implementation of the C++11
    // <random> features we replace the initialization with a naive one so we
    // can benchmarks a greater range of compilers.
    positions_type positions(N);
    for( size_type i = 0; i < N; ++i ) positions[i] = 10*drand48()+5;
    
    /// timings
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    scalar_type x0[] = { 0., -1., -2. };
    scalar_type f0[] = { 0, 0, 0 };
    
    start = std::chrono::high_resolution_clock::now();
    const size_type repetitions = 1000;
    for( std::size_t i = 0; i < repetitions; ++i )
    {
        for( std::size_t j = 0; j < 3; ++j )
            f0[j] += compute_force(positions, x0[j], 10.);
    }
    end = std::chrono::high_resolution_clock::now();
    
    for( std::size_t j = 0; j < 3; ++j )
        std::cout << "Force acting at x_0=" << x0[j] << ": " << f0[j]/repetitions << std::endl;

    int elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "elapsed time: " << elapsed_time << "mus\n";
}

