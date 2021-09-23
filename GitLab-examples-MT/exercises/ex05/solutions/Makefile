# HPCSE 2015, Banres-Hut Algorithm
# Copyright Damian Steiger
# 19. March 2015

CXX?=clang++
CXXFLAGS?=-O3 -std=c++11 -Wall -DNDEBUG -march=native

all: test benchmark barnes_hut solar_system

BarnesHutTree.o: BarnesHutTree.cpp BarnesHutTree.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test.o: test.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

solar_system.o: solar_system.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: BarnesHutTree.o test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

barnes_hut: BarnesHutTree.o main.o
	$(CXX) $(CXXFLAGS) $^ -o $@

solar_system: BarnesHutTree.o solar_system.o
	$(CXX) $(CXXFLAGS) $^ -o $@

benchmark: BarnesHutTree.o benchmark.o
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f BarnesHutTree.o
	rm -f main.o
	rm -f solar_system.o
	rm -f test.o
	rm -f benchmark.o
	rm -f test
	rm -f barnes_hut
	rm -f solar_system
	rm -f benchmark
