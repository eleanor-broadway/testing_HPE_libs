// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

// workaround for missing alignas in g++-4

#include <alignas.hpp>


int alignas(32) x[16];

