CXX=g++
CXXFLAGS=-Wall -std=c++11 -O2

all: fleas_uncorrelated fleas_binning

clean:
	rm -r fleas_uncorrelated
	rm -r fleas_binning

fleas_uncorrelated: fleas_uncorrelated.cpp fleas.hpp
	$(CXX) $(CXXFLAGS) fleas_uncorrelated.cpp -o $@

fleas_binnning: fleas_binning.cpp fleas.hpp accumulator.hpp
	$(CXX) $(CXXFLAGS) fleas_binning.cpp -o $@
