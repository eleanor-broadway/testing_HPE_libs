# Makefile

SHELL := /bin/bash
CXX?=g++
CXXFLAGS?= -Wall -O3 -std=c++11 -lstdc++
MPICXX?=mpicxx

all: diffusion2d_mpi diffusion2d_hybrid

diffusion2d_mpi: diffusion2d_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o diffusion2d_mpi diffusion2d_mpi.cpp

diffusion2d_hybrid: diffusion2d_hybrid.cpp
	TODO

.PHONY: clean
clean:
	-rm diffusion2d_mpi
	-rm diffusion2d_hybrid
