# Makefile

SHELL := /bin/bash
CXX?=g++
CXXFLAGS?= -Wall -O3 -lstdc++
NVCC?=nvcc
NVCCFLAGS?=-O3 -arch=sm_20 --ptxas-options=-v
OPENMPFLAG?=-fopenmp

.DEFAULT: all

.PHONY: all
all: diffusion_serial diffusion_openmp diffusion_cuda_shared get_compute_capability

diffusion_serial: diffusion2d_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

diffusion_openmp: diffusion2d_openmp.cpp
	$(CXX) $(CXXFLAGS) $(OPENMPFLAG) -o $@ $^

diffusion_cuda_shared: diffusion2d_cuda_shared.cu 
	$(NVCC) $(NVCCFLAGS) -o $@ $^

get_compute_capability: get_compute_capability.cu 
	$(NVCC) $(NVCCFLAGS) -o $@ $^

.PHONY: clean
clean:
	-rm diffusion_serial
	-rm diffusion_openmp
	-rm diffusion_cuda_shared
	-rm get_compute_capability
