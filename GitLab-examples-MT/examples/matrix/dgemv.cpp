// Example codes for HPC course

// Some variants of doing matrix-vector multiply using OpenMP.
// Adapted from Section 4.8.1 of Arbenz and Petersen,
// "Intro. to Parallel Computing", Oxford Uni. Press, 2004.

#include <vector>
#include <chrono>
#include <functional>
#include <cmath>
#include <omp.h>
#include "matrix.hpp"

typedef hpc12::matrix<double,hpc12::column_major> matrix_type;

// the dgemv call

extern "C" void dgemv_(char& trans, int& m, int& n, double& alpha, double const* a, int& lda,
                       double const* x, int& incx, double& beta, double * y, int& incy);
#define N 2000
#define M 2000
#define REPETITIONS 20


template <class Matrix, class Vector>
void gemv1(Matrix const& A, Vector const& x, Vector& y)
{
  #pragma omp parallel for
  for (int i=0; i<M; i++) {
    y[i] = 0.;
    for (int j=0; j<N; j++)
      y[i] += A(i,j) * x[j];
  }
}

template <class Matrix, class Vector>
void gemv2(Matrix const& A, Vector const& x, Vector& y)
{
  #pragma omp parallel 
  for (int i=0; i<M; i++) {
    #pragma omp single
    y[i] = 0.;
    double tmp = 0.;
    //#pragma omp parallel for reduction(+ : tmp)
    #pragma omp for nowait
    for (int j=0; j<N; j++)
      tmp += A(i,j) * x[j];
    #pragma omp atomic
    y[i] += tmp;
  }
}


template <class Matrix, class Vector>
void gemv3(Matrix const& A, Vector const& x, Vector& y)
{
  std::fill(y.begin(),y.end(),0.);

  double z[M];

  #pragma omp parallel private(z)
  {
    std::fill(z,z+M,0.);
    
    #pragma omp for
    for (int j=0; j<N; j++)
      for (int i=0; i<M; i++)
        z[i] += A(i,j) * x[j];

    #pragma omp critical
    for (int i=0; i<M; i++)
      y[i] += z[i];
  }
}


template <class Matrix, class Vector>
void gemv4(Matrix const& A, Vector const& x, Vector& y)
{
  for (int i=0; i<M; i++)
    y[i] = 0.;
    
  for (int j=0; j<N; j++)
    #pragma omp parallel for
    for (int i=0; i<M; i++)
      y[i] += A (i,j) * x[j] ;
}

template <class Matrix, class Vector>
void gemv5(Matrix const& A, Vector const& x, Vector& y)
{
  double DONE  = 1.;
  double DZERO = 0.;
  int    ONE   = 1;
  int    lda   = N;
  int    n     = N;
  char   trans = 'N';

  #pragma omp parallel
  {
    int p = omp_get_num_threads();
    int n0 = (M + p - 1)/p;
      
    #pragma omp for
    for (int i=0; i<p; i++) {
      int n1 = std::min(n0, M-i*n0);
      if (n1 > 0)
        dgemv_(trans, n1, n, DONE, A.data()+i*n0, lda, &x[0], ONE, DZERO, &y[i*n0], ONE);
    }
  }
}

template <class Matrix, class Vector>
void gemv6(Matrix const& A, Vector const& x, Vector& y)
{
  double DONE  = 1.;
  double DZERO = 0.;
  int    ONE   = 1;
  int    lda   = N;
  int    n     = N;
  int    m     = M;
  char   trans = 'N';
  
  dgemv_(trans, m, n, DONE, A.data(), lda, &x[0], ONE, DZERO, &y[0], ONE);
}


void timeit(std::function<void(void)> gemv, std::vector<double>& y, std::vector<double> const& y0)
{
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  
  start = std::chrono::high_resolution_clock::now();
  for (int it=0; it<REPETITIONS; ++it )
    gemv();
  end = std::chrono::high_resolution_clock::now();
  int elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  
  double s = 0.;
  for (int j=0; j<M; j++)
    s = s + (y[j] - y0[j]) * (y[j] - y0[j]);
  s = std::sqrt(s);
  
  std::cout << "elapsed time: " << elapsed_time/REPETITIONS << "musec\n";
  std::cout << "l2-error:     " << s << "\n" << std::endl;
}

int main()
{
  
  matrix_type A(N,M);
  std::vector<double> x(N), y(M), y0(M);
  
  // setup the matrix A, input vector x
  for (int j=0; j<N; j++) {
    x[j] = j;
    for (int i=0; i<M; i++)
      A(i,j) = i+j;
  }
  
  // calculate the reference result y0
  for (int i=0; i<N; i++)
    for (int j=0; j<M; j++)
      y0[i] += A(i,j) * x[j];

  
  std::cout << "Case 1: i,j / outer\n";
  timeit([&]() { gemv1(A,x,y); }, y, y0);

  std::cout << "Case 2: i,j / / inner with reduction\n";
  timeit([&]() { gemv2(A,x,y); }, y, y0);

  std::cout << "Case 3: j,i / outer with 'vector reduction'/critical section\n";
  timeit([&]() { gemv3(A,x,y); }, y, y0);

  std::cout << "Case 4: j,i / inner\n";
  timeit([&]() { gemv4(A,x,y); }, y, y0);
  
  std::cout << "Case 5: dgemv / outer\n";
  timeit([&]() { gemv5(A,x,y); }, y, y0);

  std::cout << "Case 6: dgemv\n";
  timeit([&]() { gemv6(A,x,y); }, y, y0);
  
}

