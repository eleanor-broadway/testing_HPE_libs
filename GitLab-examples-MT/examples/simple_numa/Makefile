LIBS = -lrt -lpthread
OPENMPFLAG = -fopenmp
LIBTBB = -ltbb
CXX?= g++
CXXFLAGS?= -O3 -DNDEBUG -std=c++11

FILES=vector_multiply vector_multiply_threaded vector_multiply_numa vector_multiply_numa_affinity vector_multiply_openmp vector_multiply_openmp_numa
#
all: $(FILES)

#vector_multiply_threaded: vector_multiply_threaded.cpp
#	$(CXX) $(CXXFLAGS) -pthread $^ $(LIBS) -o $@

vector_multiply: vector_multiply.cpp
	$(CXX) $(CXXFLAGS) $^ $(LIBS) -o $@

vector_multiply_numa: vector_multiply_numa.cpp
	$(CXX) $(CXXFLAGS) -pthread $^ $(LIBS) -o $@

vector_multiply_numa_affinity: vector_multiply_numa_affinity.cpp
	$(CXX) $(CXXFLAGS) -pthread $^ $(LIBS) -o $@

vector_multiply_openmp: vector_multiply_openmp.cpp
	$(CXX) $(CXXFLAGS) -pthread $(OPENMPFLAG) $^ $(LIBS) -o $@

vector_multiply_openmp_numa: vector_multiply_openmp_numa.cpp
	$(CXX) $(CXXFLAGS) -pthread $(OPENMPFLAG) $^ $(LIBS) -o $@

vector_multiply_tbb_numa: vector_multiply_tbb_numa.cpp
	$(CXX) $(CXXFLAGS) $^ $(LIBS) $(LIBTBB) -o $@

vector_multiply_tbb_numa_affinity: vector_multiply_tbb_numa_affinity.cpp
	$(CXX) $(CXXFLAGS) $^ $(LIBS) $(LIBTBB) -o $@

.PHONY: clean
clean:
	-rm $(FILES)
