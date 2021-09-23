// Example codes for HPC course
// (c) 2012 Michele Dolfi, ETH Zurich
// (c) 2012 Jan Gukelberger, ETH Zurich

#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <numeric>
#include <iterator>
#include <functional>
#include <cassert>

typedef std::size_t size_type;
typedef float scalar_type;
typedef std::vector<scalar_type> positions_type;

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
        if (std::abs(r) < rc) {
            scalar_type r2 = r * r;
            scalar_type s2 = rm2 / r2;  // (rm/r)^2
            scalar_type s6 = s2*s2*s2;  // (rm/r)^6
            force += 12*eps * (s6*s6 - s6) / r;
        }
    }
    return force;
}

int main(int argc, const char** argv)
{
    /// init random number generator
    std::mt19937 gen(42);
    std::normal_distribution<scalar_type> dist(10,1);
    
    /// placing particles with gaussian distribution
    positions_type positions(N);
    std::generate(positions.begin(), positions.end(),
                  std::bind(dist,std::ref(gen)));
    
    /// timings
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    scalar_type x0[] = { 0., -1., -2. };
    scalar_type f0[] = { 0, 0, 0 };
    
    const size_type repetitions = 1000;
    start = std::chrono::high_resolution_clock::now();
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

