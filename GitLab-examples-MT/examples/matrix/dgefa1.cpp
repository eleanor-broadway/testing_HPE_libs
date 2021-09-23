// dgefa adapeted from a code by Wesley Petersen
// (c) Matthias Troyer, ETH Zurich

#include "matrix.hpp"

extern "C" int idamax(int& n, double* x, int& incx);
extern "C" void dswap(int& n, double* x, int& incx, double* y,int incy);
extern "C" void dscal(int& n, double& a, double* x, int& incx);
extern "C" void daxpy(int& n, double& a, double* x, int& incx, double* y,int incy);

void dgefa(hpc12::matrix<double,hpc12::column_major>& a,
          std::vector<int> pivot)
{
  assert(a.num_rows() == a.num_cols());
  
  pivot.clear();
  
  int one=1;
  int n=a.num_rows();
  int lda=a.leading_dimension();
  
  for(int k=0; k < a.num_rows()-1; k++){
    // 1. find the index of the largest element in columnk k starting at row k
    int nk = n-k;
    int l = idamax(nk,&a(k,k),one) + k;
    pivot.push_back(l); // and save it
    assert( a(l,k) =!0.0); // error if the largest element is zero
    
    // 2. swap rows l and k, starting at column k
    dswap(nk,&a(l,k),lda,&a(k,k),lda);
    
    // 3. scale the column k below row k by the inverse negative pivot element
    double t = -1./a(k,k);
    int nkm1 = n-k-1;
    dscal(nkm1,t,&a(k+1,k),one);
    
    // 4. add the k-th row times the scale factor store in column k to all
    //    rows from k+1
    for(int j=k+1; j<n; j++) {   // multiple daxpy
      double t   = a(j,k);
      daxpy(nkm1,t,&a(k,k+1),one,&a(j,k+1),one);
    }
  }
}
