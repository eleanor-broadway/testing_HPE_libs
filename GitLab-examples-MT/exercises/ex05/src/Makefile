# HPCSE 2015, Banres-Hut Algorithm
# Copyright Damian Steiger
# 19. March 2015

CXX?=clang++
CXXFLAGS?=-O2 -std=c++11 -Wall 

all: test barnes_hut

BarnesHutTree.o: BarnesHutTree.cpp BarnesHutTree.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test.o: test.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: BarnesHutTree.o test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

barnes_hut: BarnesHutTree.o main.o
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f BarnesHutTree.o
	rm -f main.o
	rm -f test.o
	rm -f test
	rm -f barnes_hut
