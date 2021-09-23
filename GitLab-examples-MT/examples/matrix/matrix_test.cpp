#include <iostream>
#include <algorithm>
#include "matrix.hpp"



void test_column_major() {
    using hpc12::matrix;

    std::cout<<"Testing matrix (column major)"<<std::endl;
    matrix<double> m(5,8);

    // Test num_rows(), num_cols()
    std::cout << "Num rows:" << num_rows(m) << ", size1: " << m.size1() << std::endl;
    std::cout << "Num cols:" << num_cols(m) << ", size2: " << m.size2() << std::endl;
    std::cout << "Leading dimension: " << leading_dimension(m) <<std::endl;

    // Test element access
    m(1,0) = 1.0;
    m(2,0) = 2.0;
    m(0,1) = 0.1;
    m(1,1) = 1.1;
    m(0,2) = 0.2;

    //  Test ostream
    std::cout << m << std::endl;

    //  Test access to data pointer
    std::transform( m.data(), m.data() + (num_rows(m)*num_cols(m)), m.data(), [](double v){ return v + 5;} );

    //  Test copy constructor
    matrix<double> a(m);
    std::cout << a << std::endl;

    // Test swap
    matrix<double> b(3,2);
    swap(b,m);
    std::cout <<"m = " << m << std::endl;
    std::cout <<"b = " << b << std::endl;

    // Test assignment
    m = b;

    std::cout <<"m = " << m << std::endl;
    std::cout <<"b = " << b << std::endl;
}

void test_row_major() {
    using hpc12::matrix;
    using hpc12::row_major;

    std::cout<<"Testing matrix (row major)"<<std::endl;
    matrix<double,row_major> m(5,8);

    // Test num_rows(), num_cols()
    std::cout << "Num rows:" << num_rows(m) << ", size1: " << m.size1() << std::endl;
    std::cout << "Num cols:" << num_cols(m) << ", size2: " << m.size2() << std::endl;
    std::cout << "Leading dimension: " << leading_dimension(m) <<std::endl;

    // Test element access
    m(1,0) = 1.0;
    m(2,0) = 2.0;
    m(0,1) = 0.1;
    m(1,1) = 1.1;
    m(0,2) = 0.2;

    //  Test ostream
    std::cout << m << std::endl;

    //  Test access to data pointer
    std::transform( m.data(), m.data() + (num_rows(m)*num_cols(m)), m.data(), [](double v){ return v + 5;} );

    //  Test copy constructor
    matrix<double,row_major> a(m);
    std::cout << a << std::endl;

    // Test swap
    matrix<double,row_major> b(3,2);
    swap(b,m);
    std::cout <<"m = " << m << std::endl;
    std::cout <<"b = " << b << std::endl;

    // Test assignment
    m = b;

    std::cout <<"m = " << m << std::endl;
    std::cout <<"b = " << b << std::endl;
}

int main() {
    test_column_major();
    test_row_major();
    return 0;
}
