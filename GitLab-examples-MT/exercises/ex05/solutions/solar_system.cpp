// HPCSE II FS 2015, Exercise on Barnes-Hut Algorithm
// Implementation of Barnes-Hut Algorithm for Gravity in 2D
// 19. March 2015

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include <cassert>
#include <fstream>

#include "BarnesHutTree.hpp"


double get_systemsize(vector_array const& r)
{
    // We assume the center of mass is conserved and is approximately at 0,0
    const unsigned int n = r.x.size();
    double r2 = 0;
    for(unsigned int i=0; i < n; ++i)
        r2 = std::max(r2,r.x[i]*r.x[i] + r.y[i]*r.y[i]);
    return std::sqrt(r2);
}

void calculate_accelerations(vector_array & a, vector_array const& r, std::vector<double> const& m, double system_size)
{
    const unsigned int n = r.x.size();
    // Build tree
    BarnesHutTree tree({-system_size, system_size, -system_size, system_size});
    for(unsigned int i=0; i < n; ++i)
        tree.InsertPlanet({r.x[i],r.y[i],m[i]});

    for(unsigned int i=0; i < n; ++i)
    {
        const double minv = 1./m[i];
        Force f = tree.GetTotalForce({r.x[i], r.y[i], m[i]});
        a.x[i] = minv * f.aX;
        a.y[i] = minv * f.aY;
    }
}

void update_positions(vector_array& r, vector_array const& v, vector_array const& a, double dt)
{
    for(unsigned int i = 0; i < r.x.size(); ++i )
    {
        // Verlet step
        r.x[i] += v.x[i]*dt + 0.5*dt*dt*a.x[i];
        r.y[i] += v.y[i]*dt + 0.5*dt*dt*a.y[i];
    }
}

void update_velocities(vector_array& v, vector_array const& aold, vector_array const& a, double dt)
{
    for(unsigned int i = 0; i < v.x.size(); ++i )
    {
        v.x[i] += 0.5*dt*(aold.x[i] + a.x[i]);
        v.y[i] += 0.5*dt*(aold.y[i] + a.y[i]);
    }
}

void print_config(vector_array const& r, std::vector<double> const& m, vector_array const& v, vector_array const& a)
{
    for(unsigned int i=0; i < r.x.size(); ++i)
        std::cout << r.x[i] <<" "<< r.y[i] <<" "<< m[i] <<" "<< v.x[i] <<" "<< v.y[i] <<" "<< a.x[i] <<" "<< a.y[i] <<" "<< std::endl;
    std::cout << std::endl;
}

void evolve(vector_array & r, std::vector<double> const& m, vector_array & v, vector_array & a, double dt, unsigned int steps)
{
    using std::swap;
    vector_array aold(a);

    for(unsigned int s = 0; s < steps; ++s )
    {
        update_positions(r, v, a, dt);
        swap(a, aold);
        const double system_size = 2*get_systemsize(r);
        calculate_accelerations(a, r, m, system_size);
        update_velocities(v, aold, a, dt);
    }
}


void readin_configuration(std::vector<double> & m, vector_array & r, vector_array & v)
{
    std::ifstream f("solar_system.dat");
    std::size_t i=0;
    while(f.good())
    {
        f >> m[i] >> r.x[i] >> r.y[i] >> v.x[i] >> v.y[i];
        ++i;
    }
    f.close();
}

int main() {
    
    const unsigned int N = 27;
    const double dt = 1e-6;
    const unsigned int steps = 10000000;
    const unsigned int printsteps = 10000;
    
    vector_array       positions(N);
    std::vector<double> masses(N);
    
    vector_array       velocities(N);
    vector_array       accelerations(N);
    
    readin_configuration(masses, positions, velocities);

    std::cout << "\n";
    std::cout << "# nparticles = " << N << std::endl;
    std::cout << std::setprecision(10);
    print_config(positions, masses, velocities, accelerations);
    for( unsigned int i = 0; i < steps/printsteps; ++i )
    {
        evolve(positions, masses, velocities, accelerations, dt, printsteps);
        print_config(positions, masses, velocities, accelerations);
    }

    return 0;
}
