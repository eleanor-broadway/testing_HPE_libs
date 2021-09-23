GCC=g++-mp-4.8
ICC=iCC
CLANG=clang++
$GCC -std=c++11 -O2                              saxpy.cpp              -o saxpy_gcc_novec
$ICC -no-vec -O3 -std=c++11                      saxpy.cpp              -o saxpy_icc_novec
$CLANG -fno-vectorize -stdlib=libc++ -std=c++11 -O3           saxpy.cpp  -o saxpy_clang_novec

$CLANG -stdlib=libc++ -std=c++11 -O3           saxpy.cpp              -o saxpy_clang

$GCC -ftree-vectorize -std=c++11 -O2             saxpy.cpp              -o saxpy_gcc1
$GCC -std=c++11 -O3                              saxpy.cpp              -o saxpy_gcc2
$GCC -ftree-vectorize -std=c++11 -O2             saxpy_restrict.cpp     -o saxpy_gcc3
$GCC -ftree-vectorize -std=c++11 -O2             saxpy_aligned_gcc.cpp  -o saxpy_gcc4

$ICC -mavx -qopt-report=2 -qopt-report-phase=vec saxpy_restrict.cpp     -o saxpy_icc1
$ICC -mavx -qopt-report=2 -qopt-report-phase=vec -O3 -restrict -std=c++11 saxpy_aligned_icc.cpp  -o saxpy_icc2

$ICC -mavx -fast -O3 -std=c++11                  saxpy_sse.cpp          -o saxpy_sse
$ICC -mavx -fast -O3 -std=c++11                  saxpy_avx.cpp          -o saxpy_avx
$ICC -mavx -fast -no-vec -O0 -std=c++11          saxpy_avx.cpp          -o saxpy_avx_noopt

$GCC -c -save-temps -ftree-vectorize -std=c++11 -O3             saxpy_simple_gcc.cpp
$ICC -save-temps -c -restrict -mavx -qopt-report=3 -qopt-report-phase=vec -O3 -std=c++11 saxpy_simple_icc.cpp
$ICC -save-temps -c -restrict -mavx -no-vec -O0 -std=c++11      saxpy_simple_avx.cpp
