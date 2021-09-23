// HPCSE II FS 2015, Exercise on Barnes-Hut Algorithm
// Implementation of Barnes-Hut Algorithm for Gravity in 2D
// 19. March 2015

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cassert>

#include "BarnesHutTree.hpp"


Force total_force(unsigned int j, vector_array const& pos, std::vector<double> const& mass) {
    // compute the force acting on particle j

    //
    // TODO: implement me
    //
}


void randomize_configuration(vector_array & pos, std::vector<double> & mass, const double L)
{
    assert( pos.x.size() == pos.y.size() );
    assert( pos.x.size() == mass.size() );
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> pos_dist(-L,L);
    std::exponential_distribution<double>  mass_dist(1.);
    
    std::size_t n = pos.x.size();
    for(std::size_t i=0; i < n; ++i)
    {
        pos.x[i] = pos_dist(gen);
        pos.y[i] = pos_dist(gen);
        mass[i] = 4.*mass_dist(gen);
    }
}


int main() {
    
    const unsigned int N = 10;
    const double initial_L = 50.;
    
    vector_array         planet_positions(N);
    std::vector<double>  planet_masses(N);
    
    randomize_configuration(planet_positions, planet_masses, initial_L);

    BarnesHutTree tree({-2*initial_L,2*initial_L,-2*initial_L,2*initial_L});
    
    // Fill the tree
    for (unsigned int i=0; i < N; ++i)
        tree.InsertPlanet({planet_positions.x[i], planet_positions.y[i], planet_masses[i]});
    
    
    // Compute center of mass
    double cx=0., cy=0.;
    double total_mass=0.;
    for (unsigned int i=0; i < N; ++i)
    {
        cx += planet_positions.x[i]*planet_masses[i];
        cy += planet_positions.y[i]*planet_masses[i];
        total_mass += planet_masses[i];
    }
    cx /= total_mass;
    cy /= total_mass;
    std::cout << "# Manual center of mass : " << cx << "  " << cy << std::endl;
    std::cout << "# Tree center of mass   : " << tree.TotalCenterofMass().first << "  " << tree.TotalCenterofMass().second << std::endl;
    
    // Compute the force and print it out
    for (unsigned int i=0; i < N; ++i) { 
        
        Force force_trivial  = total_force( i, planet_positions, planet_masses);
        Force force_tree     = tree.GetTotalForce({planet_positions.x[i], planet_positions.y[i], planet_masses[i]});
        
        // check for correctness
        double x_error = std::abs(force_trivial.aX -  force_tree.aX) / std::sqrt(force_trivial.aX*force_trivial.aX + force_trivial.aY*force_trivial.aY);
        double y_error = std::abs(force_trivial.aY -  force_tree.aY) / std::sqrt(force_trivial.aX*force_trivial.aX + force_trivial.aY*force_trivial.aY);
        if (x_error > 0.01 || y_error > 0.01)
            std::cerr << "Force mismatch!  x_error=" << x_error << ",  y_error=" << y_error << std::endl;
        
        // print results
        std::cout << planet_masses[i]      << "    "
                  << planet_positions.x[i] << "    "
                  << planet_positions.y[i] << "    "
                  << force_tree.aX << "    "
                  << force_tree.aY << "    "
                  << std::endl;
    }
    
    return 0;
}
