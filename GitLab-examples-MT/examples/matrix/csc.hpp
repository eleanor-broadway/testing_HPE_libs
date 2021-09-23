// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

// this is a code fragment used to show the multiplication
// it can be used as the basis for a CSR matrix class to be used
// in the exercises

#ifndef CSR_HPP
#define CSR_HPP

#include <vector>

template <class ValueType, class SizeType=std::size_t>
class csc_matrix
{
  typedef ValueType value_type;
  typedef SizeType size_type;
  
  csc_matrix(size_type s = 0)
  : n_(s)
  , row_starts(s+1)
  {}
  
  // we are missing functions to actually fill the matrix
  
  size_type dimension() const { return n_;}
  
  std::vector<value_type> multiply(std::vector<value_type> const& x) const;
  
private:
  size_type n_;
  std::vector<size_type> row_indices;
  std::vector<size_type> col_starts;
  std::vector<value_type> data;
};


template <class ValueType, class SizeType>
std::vector<ValueType> csc_matrix<ValueType,SizeType>::multiply(std::vector<value_type> const& x) const
{
  assert( x.size()== dimension());
  std::vector<value_type> y(dimension());
  
  
  // loop over all columns
  #pragma omp parallel for
  for (size_type col = 0 ; col < dimension() ; ++ col) {
    // loop over all non-zero elements of the column
    for (size_type i = col_starts[col] ; i != col_starts[col+1] ; ++i)
      #pragma omp atomic
      y[row_indices[i]] += data[i] * x[col];
  }
  return y;
}

#endif
