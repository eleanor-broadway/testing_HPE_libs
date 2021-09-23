NVCC?=nvcc
NVCCFLAGS?=-O3

all: vector_addition

vector_addition: vector_addition.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^
