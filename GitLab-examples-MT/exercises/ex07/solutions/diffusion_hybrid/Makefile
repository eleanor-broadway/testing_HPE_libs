# Makefile

SHELL := /bin/bash
CXX?=g++
CXXFLAGS?= -Wall -O3 -std=c++11
MPICXX?=mpicxx

.DEFAULT: all

.PHONY: all
all: diffusion2d_hybrid

diffusion2d_hybrid: diffusion2d_hybrid.cpp
	$(MPICXX) $(CXXFLAGS) -o diffusion2d_hybrid diffusion2d_hybrid.cpp -fopenmp

.PHONY: clean
clean:
	-rm diffusion2d_hybrid
