// Example codes for HPC course
// (c) 2012 Andreas Hehn and Matthias Troyer, ETH Zurich

// a stripped-down matrix class for the examples of the HPC course
#ifndef HPC12_MATRIX_HPP
#define HPC12_MATRIX_HPP
#include <cassert>
#include <iostream>
#include <vector>
#include "aligned_allocator.hpp"


namespace hpc12
{

struct column_major {
    static unsigned int size1(unsigned int n_rows, unsigned int n_cols) {
        return n_rows;
    }
    static unsigned int size2(unsigned int n_rows, unsigned int n_cols) {
        return n_cols;
    }
    static std::size_t index(unsigned int n_rows, unsigned int n_cols, unsigned int i, unsigned int j)
    {
        return n_rows*j+i;
    }
};

struct row_major{
    static unsigned int size1(unsigned int n_rows, unsigned int n_cols) {
        return n_cols;
    }
    static unsigned int size2(unsigned int n_rows, unsigned int n_cols) {
        return n_rows;
    }
    static std::size_t index(unsigned int n_rows, unsigned int n_cols, unsigned int i, unsigned int j)
    {
        return n_cols*i+j;
    }
};

template <typename T, typename Ordering = column_major, typename Allocator = hpc12::aligned_allocator<T,64> >
class matrix {
  public:
    typedef T value_type;
    explicit matrix(unsigned int rows = 0, unsigned int cols = 0, T init = T())
    : data_(rows*cols,init)
    , n_rows_(rows)
    , n_cols_(cols)
    {
    }

    unsigned int num_rows() const {
        return n_rows_;
    }

    unsigned int num_cols() const {
        return n_cols_;
    }

    unsigned int size1() const {
        return Ordering::size1(n_rows_,n_cols_);
    }

    unsigned int size2() const {
        return Ordering::size2(n_rows_,n_cols_);
    }

    unsigned int leading_dimension() const {
        return size1();
    }

    // Element access
    value_type& operator()(unsigned int i, unsigned int j) {
        assert( i < n_rows_);
        assert( j < n_cols_);
        return data_[Ordering::index(n_rows_,n_cols_,i,j)];
    }

    value_type const& operator()(unsigned int i, unsigned int j) const {
        assert( i < n_rows_);
        assert( j < n_cols_);
        return data_[Ordering::index(n_rows_,n_cols_,i,j)];
    }

    value_type const* data() const {
        return data_.empty() ? 0 : &data_.front();
    }

    value_type* data() {
        return data_.empty() ? 0 : &data_.front();
    }

    friend void swap(matrix& a, matrix& b) {
        using std::swap;
        swap(a.data_,   b.data_);
        swap(a.n_rows_, b.n_rows_);
        swap(a.n_cols_, b.n_cols_);
    }
  private:
    std::vector<value_type,Allocator> data_;
    unsigned int n_rows_;
    unsigned int n_cols_;
};


// Get number of rows/colums
template <typename T, typename Ordering, typename Allocator>
unsigned int num_rows(matrix<T,Ordering,Allocator> const& m) {
    return m.num_rows();
}

template <typename T, typename Ordering, typename Allocator>
unsigned int num_cols(matrix<T,Ordering,Allocator> const& m) {
    return m.num_cols();
}

template <typename T, typename Ordering, typename Allocator>
unsigned int leading_dimension(matrix<T,Ordering,Allocator> const& m) {
    return m.leading_dimension();
}


// A nice operator for the output (to write std::cout << matrix )
  template <typename T, typename Ordering, typename Allocator>
std::ostream& operator << (std::ostream& os, matrix<T,Ordering,Allocator> const& m) {
    std::cout << "[";
    for(unsigned int i=0; i < num_rows(m); ++i) {
        if(i > 0)
            std::cout << "," << std::endl;
        std::cout << "[";
        for(unsigned int j=0; j < num_cols(m); ++j) {
            if(j > 0)
                std::cout << ", ";
            std::cout << m(i,j);
        }
        std::cout << "]";
    }
    std::cout << "]" << std::endl;
    return os;
}

} // end namespace hpc12

#endif //HPC12_MATRIX_HPP
