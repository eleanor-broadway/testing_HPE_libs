// Example codes for HPC course
// (c) 2012-2015 Jan Gukelberger, Andreas Hehn, Michele Dolfi, ETH Zurich

#include <array>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <iostream>
#include <iterator>
#include <cassert>
#include <tuple>
#include "timer.hpp"

#include <x86intrin.h>
#include "alignas.hpp"
#include "aligned_allocator.hpp"

const unsigned DIMENSIONS = 2;

typedef std::size_t size_type;
typedef float scalar_type;
typedef std::array<scalar_type,DIMENSIONS> position;

typedef std::vector< scalar_type, hpcse::aligned_allocator<scalar_type,64> > scalar_list;
typedef std::array<scalar_list,DIMENSIONS> position_list;


std::ostream& operator<<(std::ostream& os, const position& x)
{
    std::copy(x.begin(),x.end(),std::ostream_iterator<scalar_type>(os,"\t"));
    return os;
}

struct potential
{
    potential(const position& extent, scalar_type rm, scalar_type epsilon, scalar_type rc=0)
    :   extent_(extent)
    ,   rm2_(rm*rm)
    ,   eps_(epsilon)
    ,   rc2_(1e10*rm)
    ,   shift_(0)
    {
        // default cut-off radius
        if(rc <= 0 )    rc = 2.5*rm/std::pow(2,1/6.);
        
        position x = {{}};
        position y = {{rc}};
        assert( x[0] == 0  && x[1] == 0 );
        assert( y[1] == 0 );
        shift_ = -(*this)(x,y);
        rc2_ = rc*rc;
        std::cout << "# Potential shift -V(rc=" << rc << ")=" << shift_ << std::endl;
        
        // SSE vectors which can be reused
        extent0__ = _mm_set1_ps(extent_[0]);
        extent1__ = _mm_set1_ps(extent_[1]);
        minr0__ = _mm_set1_ps(-extent_[0]/2);
        maxr0__ = _mm_set1_ps( extent_[0]/2);
        minr1__ = _mm_set1_ps(-extent_[1]/2);
        maxr1__ = _mm_set1_ps( extent_[1]/2);
        rc2__ = _mm_set1_ps(rc2_);
        rm2__ = _mm_set1_ps(rm2_);
        eps12__ = _mm_set1_ps(12*eps_);
    }
    
    /// potential V(r^2)
    scalar_type operator()(scalar_type r2) const
    {
        if( r2 >= rc2_ )    return 0;
        
        scalar_type s2 = rm2_ / r2;
        scalar_type s6 = s2*s2*s2;
        return eps_*(s6*s6 - 2*s6) + shift_;
    }
    
    /// potential V(x,y) considering periodic boundaries
    scalar_type operator()(const position& x, const position& y) const
    {
        scalar_type r2 = 0.;
        for( size_type d = 0; d < DIMENSIONS; ++d )
        {
            scalar_type r = dist(x[d],y[d],extent_[d]);
            r2 += r*r;
        }
        
        return (*this)(r2);
    }
    
    /// compute the Lennard-Jones force particle y exerts on x and add it to f
    void add_force(scalar_type& f0, scalar_type& f1,
                   scalar_type x0, scalar_type x1,
                   scalar_type y0, scalar_type y1) const
    {
        scalar_type r0 = dist(x0,y0,extent_[0]);
        scalar_type r1 = dist(x1,y1,extent_[1]);
        scalar_type r2 = r0*r0 + r1*r1;
        
        if( r2 >= rc2_ )    return;
        r2 = 1/r2;
        
        // s = r_m/r
        // V(s) = eps * (s^12 - s^6)
        scalar_type s2 = rm2_ * r2; // (rm/r)^2
        scalar_type s6 = s2*s2*s2;  // (rm/r)^6
        scalar_type fr = 12*eps_ * (s6*s6 - s6) * r2; // common factor
        f0 += fr * r0;
        f1 += fr * r1;
    }
    
    void add_force(__m128& f0, __m128& f1, __m128 x0, __m128 x1, __m128 y0, __m128 y1) const
    {
        // r[d] = dist(x[d],y[d],l[d])
        __m128 r0 = dist(x0,y0,extent0__,minr0__,maxr0__);
        __m128 r1 = dist(x1,y1,extent1__,minr0__,maxr0__);
        
        // r2 = r0*r0 + r1*r1;
        __m128 r2 = _mm_add_ps(_mm_mul_ps(r0,r0), _mm_mul_ps(r1,r1));
        
        // r2 = (r2 < rc2_) / r2;
        r2 = _mm_and_ps( _mm_cmplt_ps(r2,rc2__), _mm_rcp_ps(r2) );
        
        // fr = 12*eps_ * (s6*s6 - s6) * r2
        __m128 s2 = _mm_mul_ps(rm2__,r2);                // s2 = rm2_ * r2
        __m128 s6 = _mm_mul_ps(s2, _mm_mul_ps(s2,s2));   // s6 = s2*s2*s2
        __m128 fr = _mm_sub_ps( _mm_mul_ps(s6,s6), s6 ); // fr = (s6*s6 - s6)
        fr = _mm_mul_ps(_mm_mul_ps(eps12__,fr), r2);     //     * 12*eps_ * r2
        
        // f[d] += fr * r[d]
        f0 = _mm_add_ps(f0, _mm_mul_ps(fr,r0));
        f1 = _mm_add_ps(f1, _mm_mul_ps(fr,r1));
    }
    
    /// compute the Lennard-Jones force particle y exerts on x and add it to f
    void calculate_force(scalar_type& f0_, scalar_type& f1_,
                         const scalar_type x0_, const scalar_type x1_,
                         const scalar_type* y0_, const scalar_type* y1_, const size_type n) const
    {
        __m128 x0 = _mm_set1_ps(x0_);
        __m128 x1 = _mm_set1_ps(x1_);
        __m128 f0 = _mm_setzero_ps();
        __m128 f1 = _mm_setzero_ps();
        
        size_type nvect = 4*(n/4); // # of particles we can vectorize
        for( size_type j = 0; j < nvect; j += 4 )
        {
            // y[d] = y_[d][j]
            __m128 y0 = _mm_load_ps(y0_+j);
            __m128 y1 = _mm_load_ps(y1_+j);
            
            // f[d] += f(x-y)[d]
            add_force(f0,f1,x0,x1,y0,y1);
        }
        
        // reduce resulting force vectors
        alignas(64) scalar_type ff[4];
        _mm_store_ps(ff,f0);    f0_ = ff[0] + ff[1] + ff[2] + ff[3];
        _mm_store_ps(ff,f1);    f1_ = ff[0] + ff[1] + ff[2] + ff[3];
        
        // remaining particles
        for( size_type j = nvect; j < n; ++j )
            add_force(f0_,f1_,x0_,x1_,y0_[j],y1_[j]);
    }
    
    scalar_type cutoff_radius() const { return std::sqrt(rc2_); }
    
private:
    scalar_type dist(scalar_type x, scalar_type y, scalar_type extent) const
    {
        scalar_type r = x-y;
        if     ( r < -extent/2 ) r += extent;
        else if( r >  extent/2 ) r -= extent;
        return r;
    }
    __m128 dist(__m128 x, __m128 y, __m128 l, __m128 minr, __m128 maxr) const
    {
        // scalar_type r = x-y;
        __m128 r = _mm_sub_ps(x,y);
        
        // if     ( r < -extent/2 ) r += extent;
        __m128 mask = _mm_cmplt_ps(r,minr);
        __m128 shift = _mm_and_ps(mask,l);
        r = _mm_add_ps(r,shift);
        
        // else if( r >  extent/2 ) r -= extent;
        mask = _mm_cmpgt_ps(r,maxr);
        shift = _mm_and_ps(mask,l);
        r = _mm_sub_ps(r,shift);
        
        return r;
    }
    
    position extent_;
    scalar_type rm2_;   // r_m^2
    scalar_type eps_;   // \epsilon
    scalar_type rc2_;   // cut-off radius r_c^2
    scalar_type shift_; // potential shift -V(r_c)
    
    // SSE vectors that can be reused
    __m128 extent0__, extent1__;
    __m128 minr0__, maxr0__, minr1__, maxr1__;
    __m128 rc2__, rm2__, eps12__;
};


class simulation
{
public:
    simulation(const position& extent, const potential& pot,
               const position_list& x, const position_list& v )
    :   extent_(extent)
    ,   potential_(pot)
    ,   x_(x)
    ,   v_(v)
    ,   a_(x)
    {
        calculate_forces(a_,x_);
    }
    
    void evolve(scalar_type dt, size_type steps)
    {
        configuration aold(a_);
        
        for( size_type s = 0; s < steps; ++s )
        {
            update_positions(x_,v_,a_,dt);
            std::swap(a_,aold);
            calculate_forces(a_,x_);
            update_velocities(v_,aold,a_,dt);
        }
    }
    
    void print_config() const
    {
        for( size_type i = 0; i < x_[0].size(); ++i )
            std::cout << x_[0][i] << "\t" << x_[1][i] << "\t"
            << v_[0][i] << "\t" << v_[1][i] << "\t"
            << a_[0][i] << "\t" << a_[1][i] << std::endl;
        std::cout << std::endl;
    }
    
    std::pair<scalar_type,scalar_type> measure_energies() const
    {
        scalar_type ekin = 0;
        for( size_type d = 0; d < DIMENSIONS; ++d )
            ekin += std::inner_product(v_[d].begin(),v_[d].end(),v_[d].begin(),scalar_type(0));
        
        const size_type n = x_[0].size();
        const scalar_type* x0 = &x_[0].front();
        const scalar_type* x1 = &x_[1].front();
        scalar_type epot = 0;
        for( size_type i = 0; i < n; ++i )
        {
            const position xx{{x0[i],x1[i]}};
            for( size_type j = 0; j < i; ++j )
            {
                position yy{{x0[j],x1[j]}};
                epot += potential_(xx,yy);
            }
        }
        
        return std::make_pair(0.5*ekin,epot);
    }
    
    
private:
    typedef position_list configuration;
    
    void update_positions(configuration& x, const configuration& v, const configuration& a, scalar_type dt)
    {
        for( size_type d = 0; d < DIMENSIONS; ++d )
        {
            for( size_type i = 0; i < x[d].size(); ++i )
            {
                // Verlet step
                x[d][i] += v[d][i]*dt + 0.5*dt*dt*a[d][i];
                
                // enforce periodic boundaries
                x[d][i] = fmod(x[d][i],extent_[d]);
                if( x[d][i] <  0 )   x[d][i] += extent_[d];
                assert( x[d][i] >= 0 && x[d][i] < extent_[d] );
            }
        }
    }
    
    void update_velocities(configuration& v, const configuration& aold, const configuration& a, scalar_type dt)
    {
        for( size_type d = 0; d < DIMENSIONS; ++d )
        {
            for( size_type i = 0; i < v[d].size(); ++i )
                v[d][i] += 0.5*dt*(aold[d][i] + a[d][i]);
        }
    }
    
    void calculate_forces(configuration& a, configuration& x)
    {
        scalar_type* x0 = &x[0].front();
        scalar_type* x1 = &x[1].front();
        scalar_type* a0 = &a[0].front();
        scalar_type* a1 = &a[1].front();
        
        const size_type n = x[0].size();
        for( size_type i = 0; i < n; ++i )
        {
            std::swap(x0[i],x0[n-1]);
            std::swap(x1[i],x1[n-1]);
            const scalar_type xi0 = x0[n-1];
            const scalar_type xi1 = x1[n-1];
            
            potential_.calculate_force(a0[i],a1[i],xi0,xi1,x0,x1,n-1);
            
            std::swap(x0[i],x0[n-1]);
            std::swap(x1[i],x1[n-1]);
        }
    }
    
    
    position extent_; /// system extent along each dimension
    potential potential_;
    
    configuration x_;
    configuration v_;
    configuration a_;
};

position_list init_circle(const position& extent, size_type n)
{
    using std::sin;
    using std::cos;
    position_list config;
    position midpoint{{extent[0]/2, extent[1]/2}};
    for(size_type i=0; i < n; ++i) {
        config[0].push_back( static_cast<scalar_type>(midpoint[0]+0.9*midpoint[0]*sin(2*i*M_PI/n)) );
        config[1].push_back( static_cast<scalar_type>(midpoint[1]+midpoint[1]*0.9*cos(2*i*M_PI/n)) );
    }
    return config;
}


/// init hexagonal lattice putting n particles in box with given extent,
/// considering optimal particle spacing rm
position_list init_hexagonal(const position& extent, size_type n, scalar_type rm)
{
    assert( DIMENSIONS == 2 );
    
    // determine # of rows and columns such that inter-particle distances are roughly equal
    using std::sqrt;
    size_type ny = sqrt( 2*extent[1]*n / (sqrt(3.)*extent[0]) ) + 0.5;
    const size_type nx = std::ceil(n/double(ny));
    ny = std::min( double(ny), std::ceil(n/double(nx)));
    
    // optimal spacing would be rm, but less if we have to pack in more particles
    const scalar_type dx = std::min(                        rm, extent[0]/nx);
    const scalar_type dy = std::min(scalar_type(sqrt(0.75))*rm, extent[1]/ny);
    
    position_list result;
    for( size_type y = 0; y < ny; ++y )
    {
        for( size_type x = 0; x < nx && result[0].size() < n; ++x )
        {
            if( y % 2 == 0 )    result[0].push_back(dx*x      );
            else                result[0].push_back(dx*(x+0.5));
            result[1].push_back(y*dy);
        }
    }
    
    assert( result[0].size() == result[1].size() );
    return result;
}

position_list init_square_lattice(const position& extent, size_type n)
{
    assert( DIMENSIONS == 2);
    position_list p;
    size_type perrows = static_cast<size_type>(std::ceil(std::sqrt(n)));
    
    scalar_type deltax = extent[0] / perrows;
    scalar_type deltay = extent[1] / perrows;
    scalar_type offsetx = deltax*0.1;
    scalar_type offsety = deltay*0.1;
    for(size_type i=0; i < perrows; ++i)
    {
        for(size_type j=0; j < perrows; ++j)
        {
            if(p[0].size() >= n)
                break;
            p[0].push_back(i*deltax+offsetx);
            p[1].push_back(j*deltay+offsety);
        }
    }
    return p;
}

position_list init_velocities(size_type n, scalar_type ekin)
{
    if( ekin < 0 )
        throw std::runtime_error("init_velocities: cannot set negative kinetic energy "); //+std::to_string(ekin));
    
    // <random> from gcc 4.4 with icc is broken
    // Gaussian velocity distribution
    // std::mt19937 gen;
    // for( size_type i = 0; i < 1000000; ++i )    gen();
    // std::normal_distribution<scalar_type> dist(0,1);
    position_list v{{scalar_list(n),scalar_list(n)}};
    for( scalar_list& vv : v )
    {
        for( scalar_type& vvv : vv ) vvv = 2*drand48() - 1;
    }
    
    // T = 1/2 \sum_i v_i^2
    scalar_type t = 0;
    for( const scalar_list& vv : v )
        t += 0.5 * std::inner_product(vv.begin(),vv.end(),vv.begin(),scalar_type(0));
    
    // rescale v distribution
    scalar_type lambda = std::sqrt(ekin/t);
    for( scalar_list& vv : v )
        std::transform(vv.begin(),vv.end(),vv.begin(),[lambda](scalar_type s) { return lambda*s; });
    
    assert( v[0].size() == v[1].size() );
    return v;
}

int main(int argc, const char** argv)
{
    try
    {
        if( argc < 9 || argc > 10 ){
            std::cerr << "Usage: " << argv[0] << " [box_length] [# particles] [r_m] [epsilon] [time step] [# time steps] [print steps] [init_structure] ([ekin])" << std::endl
            << "    e.g.  " << argv[0] << " 1.0 100 0.05 5.0 1e-7 1000 200 circle 0" << std::endl;
            return -1;
        }
        scalar_type box_length = std::atof(argv[1]);
        size_type   particles  = std::atoi(argv[2]);
        scalar_type rm         = std::atof(argv[3]);
        scalar_type eps        = std::atof(argv[4]);
        scalar_type dt         = std::atof(argv[5]);
        size_type   steps      = std::atoi(argv[6]);
        size_type   printsteps = std::atoi(argv[7]);
        std::string init_cond(argv[8]);
        scalar_type ekinpp     = 0.0;
        if(argc == 10)
            ekinpp = std::atof(argv[9]);
        
        position extent;
        std::fill(extent.begin(),extent.end(),box_length);
        potential pot(extent,rm,eps);
        
        // init particle positions
        position_list x;
        if(init_cond == "hexagonal")
            x = init_hexagonal(extent,particles,rm);
        else if(init_cond == "circle")
            x = init_circle(extent,particles);
        else if(init_cond == "square_lattice")
            x = init_square_lattice(extent,particles);
        else
        {
            std::cerr << "ERROR: Unknown initial condition structure" << std::endl;
            return -1;
        }
        
        particles = x[0].size();
        std::cout << "# nparticles = " << particles << std::endl;
        
        position_list v = init_velocities(particles,ekinpp);
        
        // init and run simulation for [steps] steps
        simulation sim(extent,pot,x,v);
        for( size_type i = 0; i < steps/printsteps; ++i )
        {
            // print energies and configuration every [printsteps] steps
            scalar_type ekin,epot;
            std::tie(ekin,epot) = sim.measure_energies();
            std::cout << "# E/N=" << (ekin+epot)/particles << ", Ekin/N=" << ekin/particles
            << ", Epot/N=" << epot/particles << std::endl;
#ifdef PRINT_CONFIGS
            sim.print_config();
#endif //PRINT_CONFIGS
            // run for [printsteps] steps
            timer t;
            t.start();
            sim.evolve(dt,printsteps);
            t.stop();
            std::cerr << "Timing: time=" << t.get_timing() << " nparticles=" << particles << " steps=" << printsteps << std::endl;
        }
    }
    catch( std::exception& e )
    {
        std::cerr << e.what() << std::endl;
        throw;
    }
}

