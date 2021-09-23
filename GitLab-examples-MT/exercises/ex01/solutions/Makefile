CXX?=g++-mp-4.8
CXXFLAGS?=-Wall -std=c++11 -O2

all: sum_serial sum_sse sum_avx sum_gcc

clean:
	rm -f sum_serial sum_sse sum_avx sum_gcc

sum_serial: sum_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

sum_sse: sum_sse.cpp
	$(CXX) $(CXXFLAGS) -msse -o $@ $<

sum_avx: sum_avx.cpp
	$(CXX) $(CXXFLAGS) -mavx -Wa,-q -o $@ $< # for mac
	#$(CXX) $(CXXFLAGS) -mavx -o $@ $< # for linux

## Note: without -Ox there is no optimization.
sum_gcc: sum_gcc.cpp
	$(CXX) $(CXXFLAGS) -O2 -funsafe-math-optimizations -ftree-vectorize -ftree-vectorizer-verbose=1 -o $@ $<

