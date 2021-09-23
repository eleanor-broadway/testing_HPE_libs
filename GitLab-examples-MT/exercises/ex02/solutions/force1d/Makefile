CXX?=g++
CXXFLAGS= -Wall -std=c++11
LDFLAGS= 
SSEFLAG=-msse4
# -msse4 may cause problems on some AMD CPUS
# use -msse4a instead


all: force1d force1d_O3 force1d_vec_sse force1d_vec_avx force1d_sse

force1d: force1d.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -O0 force1d.cpp -o force1d

force1d_O3: force1d.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -O3 -ffast-math -fno-tree-vectorize force1d_vec.cpp -o force1d_O3

force1d_vec_sse: force1d_vec.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -O3 $(SSEFLAG) -funsafe-math-optimizations -ftree-vectorize -ftree-vectorizer-verbose=3 force1d_vec.cpp -o force1d_vec_sse

force1d_vec_avx: force1d_vec.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -O3 -mavx -funsafe-math-optimizations -ftree-vectorize -ftree-vectorizer-verbose=3 force1d_vec.cpp -o force1d_vec_avx

force1d_sse: force1d_sse.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -O3 $(SSEFLAG) -funsafe-math-optimizations -ftree-vectorize force1d_sse.cpp -o force1d_sse

clean:
	rm force1d force1d_O3 force1d_vec_sse force1d_vec_avx force1d_sse
