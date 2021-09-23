// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

// this is a code fragment used to show the multiplication
// it can be used as the basis for a CSR matrix class to be used
// in the exercises

#ifndef CSR_HPP
#define CSR_HPP

#include <vector>

template <class ValueType, class SizeType=std::size_t>
class csr_matrix
{
  typedef ValueType value_type;
  typedef SizeType size_type;
  
  csr_matrix(size_type s = 0)
  : n_(s)
  , row_starts(s+1)
  {}
  
  // we are missing functions to actually fill the matrix
  
  size_type dimension() const { return n_;}
  
  std::vector<value_type> multiply(std::vector<value_type> const& x) const;
  
private:
  size_type n_;
  std::vector<size_type> col_indices;
  std::vector<size_type> row_starts;
  std::vector<value_type> data;
};


template <class ValueType, class SizeType>
std::vector<ValueType> csr_matrix<ValueType,SizeType>::multiply(std::vector<value_type> const& x) const
{
  assert( x.size()== dimension());
  std::vector<value_type> y(dimension());
  
  
  // loop over all rows
  #pragma omp parallel for
  for (size_type row = 0 ; row < dimension() ; ++ row) {
    // loop over all non-zero elements of the row
    for (size_type i = row_starts[row] ; i != row_starts[row+1] ; ++i)
      y[row] += data[i] * x[col_indices[i]];
  }
  return y;
}

#endif
