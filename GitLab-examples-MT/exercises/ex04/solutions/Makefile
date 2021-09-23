# Makefile

SHELL := /bin/bash
CXX?=g++
CXXFLAGS?= -Wall -O3 -std=c++11 -lstdc++
MPICXX?=mpicxx

all: diffusion_mpi

diffusion_mpi: diffusion2d_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o diffusion_mpi diffusion2d_mpi.cpp

.PHONY: clean
clean:
	-rm diffusion_mpi
