// HPCSE II FS 2015, Exercise on Barnes-Hut Algorithm
// Implementation of Barnes-Hut Algorithm for Gravity in 2D
// 19. March 2015

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>

#include "BarnesHutTree.hpp"


void randomize_configuration(vector_array & pos, std::vector<double> & mass, vector_array & v, const double L)
{
    assert( pos.x.size() == pos.y.size() );
    assert( pos.x.size() == mass.size() );
    std::mt19937 gen(3000);
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

void calculate_accelerations_n2loop(vector_array & a, vector_array const& r, std::vector<double> const& m)
{
    const std::size_t n = r.x.size();
    const double G = 2.95736e-4;
    for(std::size_t i=0; i < n; ++i)
    {
        double ax = 0;
        double ay = 0;
        for(std::size_t j=0; j < n; ++j)
        {
            if(i==j)
                continue;
    
            const double rx = r.x[i] - r.x[j];
            const double ry = r.x[i] - r.y[i];
            const double r2 = rx*rx + ry*ry;
            const double r  = std::sqrt(r2);
        
            const double ai = -G*m[j] / (r2*r);
    
            ax = ai * rx;
            ay = ai * ry;
        }
        a.x[i] = ax;
        a.y[i] = ay;
    }
}

void print_config(vector_array const& r, std::vector<double> const& m, vector_array const& v, vector_array const& a)
{
    for(unsigned int i=0; i < r.x.size(); ++i)
        std::cout << r.x[i] <<" "<< r.y[i] <<" "<< m[i] <<" "<< v.x[i] <<" "<< v.y[i] <<" "<< a.x[i] <<" "<< a.y[i] <<" "<< std::endl;
    std::cout << std::endl;
}


void benchmark(const unsigned int N, const unsigned int steps)
{
    const double initial_L = 50.;
    vector_array       positions(N);
    std::vector<double> masses(N);
    
    vector_array       velocities(N);
    vector_array       accelerations(N);
    
    randomize_configuration(positions, masses, velocities, initial_L);
    tangential_velocities(velocities, positions, masses);

    // Benchmark Barnes-Hut
    {
        double const system_size = 2*get_systemsize(positions);

        std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
        start = std::chrono::high_resolution_clock::now();
        for( unsigned int i = 0; i < steps; ++i )
            calculate_accelerations(accelerations, positions, masses, system_size);
        stop = std::chrono::high_resolution_clock::now();

        int elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
        std::cout << "Barnes-Hut iterations:" << steps << " N:" << N << "\t time: " << elapsed_time << "mus\n";
    }

    // Benchmark N^2 loop
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
        start = std::chrono::high_resolution_clock::now();
        for( unsigned int i = 0; i < steps; ++i )
            calculate_accelerations_n2loop(accelerations, positions, masses);
        stop = std::chrono::high_resolution_clock::now();

        int elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
        std::cout << "N^2 loop   iterations:" << steps << " N:" << N << "\t time: " << elapsed_time << "mus\n";
    }

}
int main() {
    benchmark(/*number of particles*/, /*iterations*/);
    return 0;
}
