# module load open_mpi/1.6.5 

all: mpi_communicators

mpi_communicators: mpi_communicators.cpp
	mpicxx -O3 -Wall -std=c++11 -o $@ $<

