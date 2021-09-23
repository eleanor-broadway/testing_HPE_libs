// HPCSE II FS 2015, Exercise on Barnes-Hut Algorithm
// Implementation of Barnes-Hut Algorithm for Gravity in 2D
// 19. March 2015

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cassert>

#include "BarnesHutTree.hpp"


void randomize_configuration(vector_array & pos, std::vector<double> & mass, vector_array & v, const double L)
{
    assert( pos.x.size() == pos.y.size() );
    assert( pos.x.size() == mass.size() );
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> pos_dist(-L,L);
    std::exponential_distribution<double>  mass_dist(1.);
    
    const std::size_t n = pos.x.size();
    for(std::size_t i=0; i < n; ++i)
    {
        pos.x[i] = pos_dist(gen);
        pos.y[i] = pos_dist(gen);
        v.x[i] = 5e-1 * pos_dist(gen);
        v.y[i] = 5e-1 * pos_dist(gen);
        mass[i] = 4.*mass_dist(gen);
    }
}

void tangential_velocities(vector_array & v, vector_array const& r, std::vector<double> const& mass)
{
    const unsigned int n = r.x.size();
    double x = 0;
    double y = 0;
    double tm = 0;
    for(unsigned int i=0; i < n; ++i)
    {
        x += r.x[i] * mass[i];
        y += r.y[i] * mass[i];
        tm += mass[i];
    }
    x/=tm;
    y/=tm;

    for(unsigned int i=0; i < n; ++i)
    {
        double ra = std::sqrt((r.x[i]-x)*(r.x[i]-x) + (r.y[i]-y)*(r.y[i]-y));
        double va = v.x[i]/ra;
        v.x[i] = va*r.y[i];
        v.y[i] = -va*r.x[i];
    }
}

double get_systemsize(vector_array const& r)
{
    // We assume the center of mass is conserved and is approximately at 0,0
    const unsigned int n = r.x.size();
    double r2 = 0;
    for(unsigned int i=0; i < n; ++i)
        r2 = std::max(r2,r.x[i]*r.x[i] + r.y[i]*r.y[i]);
    return std::sqrt(r2);
}


void calculate_accelerations(vector_array & a, vector_array const& r, std::vector<double> const& m, double length)
{
    unsigned int n = r.x.size();
    // Build tree
    double system_size = 2*get_systemsize(r);

    //
    // TODO: calculate the forces/accelerations using Barnes-Hut
    //
}

void print_config(vector_array const& r, std::vector<double> const& m, vector_array const& v, vector_array const& a)
{
    for(unsigned int i=0; i < r.x.size(); ++i)
        std::cout << r.x[i] <<" "<< r.y[i] <<" "<< m[i] <<" "<< v.x[i] <<" "<< v.y[i] <<" "<< a.x[i] <<" "<< a.y[i] <<" "<< std::endl;
    std::cout << std::endl;
}

void evolve(vector_array & r, std::vector<double> const& m, vector_array & v, vector_array & a, double dt, unsigned int steps, double length)
{
    //
    // TODO: implement Verlet algorithm
    //
}

int main() {
    
    const unsigned int N = 10;
    const double initial_L = 50.;
    const double dt = 0.5e-1;
    const unsigned int steps = 10000000;
    const unsigned int printsteps = 100000;
    
    vector_array       positions(N);
    std::vector<double> masses(N);
    
    vector_array       velocities(N);
    vector_array       accelerations(N);
    
    randomize_configuration(positions, masses, velocities, initial_L);
    tangential_velocities(velocities, positions, masses);

    std::cout << "\n";
    std::cout << "# nparticles = " << N << std::endl;
    print_config(positions, masses, velocities, accelerations);
    for( unsigned int i = 0; i < steps/printsteps; ++i )
    {
        evolve(positions, masses, velocities, accelerations, dt, printsteps, initial_L);
        print_config(positions, masses, velocities, accelerations);
    }

    return 0;
}
