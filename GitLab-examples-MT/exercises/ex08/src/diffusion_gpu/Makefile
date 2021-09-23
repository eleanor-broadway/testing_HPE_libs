# Makefile

SHELL := /bin/bash
CXX?=g++
CXXFLAGS?= -Wall -O3 -std=c++11 -lstdc++
OPENMPFLAG?=-fopenmp

.DEFAULT: all

.PHONY: all
all: diffusion_serial diffusion_openmp diffusion_cuda

diffusion_serial: diffusion2d_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

diffusion_openmp: diffusion2d_openmp.cpp
	$(CXX) $(CXXFLAGS) $(OPENMPFLAG) -o $@ $^

diffusion_cuda: diffusion2d_cuda.cu 
	TODO

.PHONY: clean
clean:
	-rm diffusion_serial
	-rm diffusion_openmp
	-rm diffusion_cuda
