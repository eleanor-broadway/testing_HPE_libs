#include <vector>
#include <numeric>
#include <iostream>
#include <x86intrin.h>
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
    
    hpcse::timer<> tim;
    tim.start();
    for (int k = 0; k < nrep; ++k)
    {
        __m128 f2 = _mm_set_ps1(2.f);
        for( int i = 0; i < n; i += 4 )
        {
            // z[i] = x[i]*x[i] + y[i]*y[i] + 2.*x[i]*y[i];
            __m128 xx = _mm_load_ps( &x[i] );
            __m128 yy = _mm_load_ps( &y[i] );
            
            __m128 z1 = _mm_mul_ps( xx, xx );
            __m128 z2 = _mm_mul_ps( yy, yy );
            __m128 z3 = _mm_mul_ps( xx, yy );
            z3 = _mm_mul_ps( f2, z3 );

            __m128 zz = _mm_add_ps( z1, z2 );
            zz = _mm_add_ps( zz, z3 );
            
            // Store back the result.
            // Since the result won't be used any time soon it is useless to
            // store it in the caches. It would only pollute the caches. Hence
            // we store it back to the RAM bypassing the cache-hierarchy by
            // using _mm_stream_ps instead of the regular _mm_store_ps.
            _mm_stream_ps( &z[i], zz );
        }
    }
    tim.stop();
    
    // print result checksum
    std::cout << std::accumulate(z.begin(), z.end(), 0.) << std::endl;
    std::cout << "Task took " << tim.get_timing() << " seconds." << std::endl;
}

