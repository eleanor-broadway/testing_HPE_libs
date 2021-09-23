# Makefile
CXX?=g++
CXXFLAGS?=-Wall -O3 -std=c++11
OPENMPFLAGS?=-fopenmp

.PHONY: all
all: pagerank.serial pagerank.openmp pagerank.openmp_numa

pagerank.serial: PageRank_Serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

pagerank.openmp: PageRank_OpenMP.cpp
	$(CXX) $(CXXFLAGS) $(OPENMPFLAGS) -o $@ $<

pagerank.openmp_numa: PageRank_OpenMP.cpp
	$(CXX) $(CXXFLAGS) $(OPENMPFLAGS) -DHPCSE_USE_NUMA -o $@ $<

.PHONY: clean
clean:
	-rm pagerank.serial pagerank.openmp pagerank.openmp_numa
