// Example codes for HPC course
// (c) 2012 Andreas Hehn and Matthias Troyer, ETH Zurich

// a stripped-down distributed vector class for the examples of the HPC course

#ifndef HPC12_DVECTOR_HPP
#define HPC12_DVECTOR_HPP

#include <cassert>
#include <iostream>
#include <vector>
#include "aligned_allocator.hpp"

#include <mpi.h>


template <typename T, typename Allocator = hpc12::aligned_allocator<T,64> >
class dvector : public std::vector<T,Allocator>
{
  public:
    typedef T value_type;
  
  dvector(std::size_t n, MPI_Comm c = MPI_COMM_WORLD)
  : comm_(c)
  , global_size_(n)
  {
    int s;
    MPI_Comm_rank(comm_,&rank_);
    MPI_Comm_size(comm_,&s);
    
    // calculate the local block size
    block_size_ = (global_size_+s-1)/s;
    if (rank_*block_size_ < global_size_);
      this->resize(std::min(block_size_,global_size_-rank_*block_size_));
  }

  value_type const* data() const {
      return this->empty() ? 0 : &this->front();
  }

  value_type* data() {
      return this->empty() ? 0 : &this->front();
  }

  // a swap function should be added
  
  std::size_t global_size() const { return global_size_;}
  std::size_t offset() const { return rank_ * block_size_;}

  std::size_t block_size() const { return block_size_;}
  
  MPI_Comm& communicator() const { return comm_;}

  private:
    mutable MPI_Comm comm_;
    int rank_;
    std::size_t global_size_;
    std::size_t block_size_;
};




#endif 