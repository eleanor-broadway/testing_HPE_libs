CC=g++
CXXFLAGS = -Wall -std=c++11
LDFLAGS = 


all: force1d force2d

force1d: force1d.cpp
	$(CC) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

force2d: force2d.cpp
	$(CC) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

clean:
	rm force1d force1d_O3 force1d_vec_sse force1d_vec_avx force1d_sse
