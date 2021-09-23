CXX?= c++
CXXFLAGS?= -std=c++11 -Wall -O3
LIBS=

all: gemm

gemm: gemm.cpp matrix.hpp aligned_allocator.hpp
	$(CXX) $(CXXFLAGS) gemm.cpp $(LIBS) -o gemm
