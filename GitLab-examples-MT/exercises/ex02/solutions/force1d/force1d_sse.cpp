#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <numeric>
#include <iterator>
#include <cassert>
#include <x86intrin.h>

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
    scalar_type rm2   = rm * rm;
    scalar_type eps12 = 12*eps;
    size_type   ndiv4 = N/4;
    
    __m128 mmx0    = _mm_load1_ps(&x0);
    __m128 mmrc    = _mm_load1_ps(&rc);
    __m128 mmrm2   = _mm_load1_ps(&rm2);
    __m128 mmeps12 = _mm_load1_ps(&eps12);
    __m128 mmforce = _mm_setzero_ps();
    // bit mask with 1 at the positions of the sign bits within an __m128 vector
    __m128 signmask = _mm_castsi128_ps(_mm_set1_epi32(1u << 31));
    
    for (size_type i=0; i<ndiv4; ++i) {
        __m128 mmxi    = _mm_load_ps(&positions[i*4]);
        __m128 mmr     = _mm_sub_ps(mmx0,mmxi);
        // compute abs(r) by unsetting the sign bits
        __m128 mmabsr  = _mm_andnot_ps(signmask, mmr);
        // rinv = (abs(r) < rc) ? 1/r : 0
        __m128 mmrinv  = _mm_and_ps( _mm_cmplt_ps(mmabsr,mmrc), _mm_rcp_ps(mmr) );
        __m128 mmrinv2 = _mm_mul_ps(mmrinv,mmrinv);
        __m128 mms2    = _mm_mul_ps(mmrm2,mmrinv2);
        __m128 mms6    = _mm_mul_ps(mms2,_mm_mul_ps(mms2,mms2));
        
        __m128 mmpart  = _mm_mul_ps(_mm_mul_ps(mmeps12,
                                               _mm_sub_ps(_mm_mul_ps(mms6,mms6),
                                                          mms6)),
                                    mmrinv);
        
        mmforce = _mm_add_ps(mmforce, mmpart);
    }
    alignas(16) scalar_type forces[4];
    _mm_store_ps(forces,mmforce);

    for (size_type i=ndiv4*4 ; i<N ; ++i) {
        scalar_type r  = x0 - positions[i];
        if( r >= rc ) continue;
        scalar_type rinv = 1./r;
        scalar_type rinv2 = rinv * rinv;
        scalar_type s2 = rm2 * rinv2;  // (rm/r)^2
        scalar_type s6 = s2*s2*s2;     // (rm/r)^6
        forces[0] += 12*eps * (s6*s6 - s6) * rinv;
    }
    
    
    return forces[0]+forces[1]+forces[2]+forces[3];
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
            f0[j] += compute_force(positions, x0[j],10.);
    }
    end = std::chrono::high_resolution_clock::now();
    
    for( std::size_t j = 0; j < 3; ++j )
        std::cout << "Force acting at x_0=" << x0[j] << ": " << f0[j]/repetitions << std::endl;

    int elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "elapsed time: " << elapsed_time << "mus\n";
}

