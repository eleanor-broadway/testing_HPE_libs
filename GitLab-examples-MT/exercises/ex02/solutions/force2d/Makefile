CXX?=c++
CXXFLAGS = -std=c++11 -O2


all: nbody nbody_sse

nbody: nbody.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

nbody_sse: nbody_sse.cpp
	$(CXX) $(CXXFLAGS) -msse -o $@ $<
