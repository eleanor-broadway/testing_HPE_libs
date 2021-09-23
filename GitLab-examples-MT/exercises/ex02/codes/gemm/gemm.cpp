#include <iostream>
#include <cassert>
#include <algorithm>
#include <chrono>
#include "matrix.hpp"


typedef hpcse::matrix<double,hpcse::column_major> matrix_type;


// This function does a naive matrix multiplication c = a*b
// We assume column_major ordering
void gemm(
      matrix_type const& a
    , matrix_type const& b
    , matrix_type& c
){
    // Assumtions on the sizes of the matrices
    assert(num_rows(a) == num_cols(b));
    assert(num_rows(c) == num_rows(a));
    assert(num_cols(c) == num_cols(b));

    for(unsigned int i=0; i < num_rows(a); ++i) {
        for(unsigned int j=0; j < num_cols(b); ++j) {
            c(i,j) = 0.0;
            for(unsigned int k=0; k < num_cols(a); ++k) {
                c(i,j) += a(i,k) * b(k,j);
            }
        }
    }
}


int main() {

    matrix_type c(1536,1536);
    matrix_type b(1536,1536);
    matrix_type a(1536,1536);

    // Fill matrices a and b with some values
    double x = 0.0;
    std::generate_n( a.data(), num_rows(a)*num_cols(a), [&x]() -> double { x+=0.1; return x; });
    std::generate_n( b.data(), num_rows(b)*num_cols(b), [&x]() -> double { x-=0.15; return x; });

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    // Do gemm
    gemm(a,b,c);
    end = std::chrono::high_resolution_clock::now();

    double elapsed_seconds = std::chrono::duration<double>(end-start).count();
    std::cout <<"GEMM ran " << elapsed_seconds << "s" <<std::endl;
    return 0;
}
