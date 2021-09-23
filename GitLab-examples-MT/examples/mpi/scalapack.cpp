// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include "matrix.hpp"
#include <cassert>
#include <limits>
#include <cstdlib>

// Cblacs and ScaLAPACK function prototypes
extern "C" {
  void Cblacs_pinfo(int*, int*);
  void Cblacs_get(int, int, int*);
  void Cblacs_gridinit(int*, char const*, int, int);
  void Cblacs_gridinfo(int, int*, int*, int*, int*);
  void Cblacs_barrier(int , char*);
  void Cblacs_gridexit(int);
  void Cblacs_exit(int);
  
  int numroc_(int const& n, int const& nb, int const& iproc, int const& isproc, int const& nprocs);
  int indxg2p_(int const& glob, int const& nb, int const& iproc, int const& isproc, int const& nprocs);
  int indxl2g_(int const& loc, int const& nb, int const& iproc, int const& isproc, int const& nprocs);
  void descinit_( int *desc, int const& m, int const& n, int const& mb, int const& nb, int const& irsrc, int const& icsrc, int const& ictxt, int const& lld, int *info);
  void pdgesv_( int const& n, int const& nrhs, double *A, int const& ia, int const& ja, int *desca, int* ipiv, double *B, int const& ib, int const& jb, int *descb, int *info);
  void pdgemm_( char const *transa, char const *transb, int const& M, int const& N, int const& K, double const& ALPHA,  double * A, int const& IA, int const& JA, int * DESCA, double * B, int const& IB, int const& JB, int * DESCB, double const& BETA, double * C, int const& IC, int const& JC, int * DESCC );
  double pdlange_( char const *norm, int const& m, int const& n, double *A, int const& ia, int const& ja, int *desca, double *work);
}


int main(int argc, char** argv)
{
  int rank;
  int nprocs;
  int myrow;
  int mycol;
  int ctxt;

  // we want to use 2x3 processes
  int nprow=2;
  int npcol=3;
  
  // we want a 100x 100 matrix
  int n=100;
  
  // and we want to solve with 1 right hand side
  int nrhs = 1;
  
  // we will use 32x32 size blocks in the block-cyclic layout
  int nb=32;

  // initialize MPI using BLACS. Using MPI this is simimilar to the three statements below
  // MPI_Init(&argc, &argv);
  // MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Cblacs_pinfo(&rank,&nprocs);
  assert(nprocs>=nprow*npcol);
  
  
  // get the system context 
  Cblacs_get(0,0,&ctxt);

  // initialize a 2x3 process grid
  Cblacs_gridinit(&ctxt,"Row-major",nprow,npcol);
  
  // get my coordinates in the process grid
  Cblacs_gridinfo(ctxt,&nprow,&npcol,&myrow,&mycol);
  
  // continue only if this rank is actually part of the grid
  if (myrow>=0) {
    
    std::cout << "Rank " << rank << " has coordinates " << myrow << " " << mycol << ", ";
    
    // now intialize the matrix
    // first calculate how many rows (np) and columns (nq) we store locally
    int np = numroc_(n,nb,myrow,0,nprow);
    int nq = numroc_(n,nb,mycol,0,npcol);
    int nqrhs = numroc_(nrhs,nb,mycol,0,npcol);
    
    std::cout << "stores " << np << "x" << nq << " of the matrix and "
              << np << "x" << nqrhs << " of the vector" << std::endl;

    // allocate local storage
    hpc12::matrix<double> A(np,nq);
    hpc12::matrix<double> B(np,nqrhs);
    
    // fill the local vector: we take a simple test vector with entry B[i]=i+1 if it were a local vector
    if (nqrhs)
      for (int i=0;i<np;++i)
        B(i,0) = indxl2g_(i+1,nb,myrow,0,nprow);
    
    // fill the local matrix as a random matrix
    srand48(rank);
    for (int i=0;i<np;++i)
      for (int j=0;j<nq;++j) 
        A(i,j) = drand48();

    // copy the local matrix to keep it safe for checking
    // also copy the right hand side
    hpc12::matrix<double> Acopy = A;
    hpc12::matrix<double> X     = B;
    
    // create descriptors for the matrix and right hand side
    int descA[9], descB[9], info;
    descinit_( descA, n, n   , nb, nb, 0, 0, ctxt, A.leading_dimension(), &info );
    descinit_( descB, n, nrhs, nb, nb, 0, 0, ctxt, B.leading_dimension(), &info );

    // call the parallel solver
    std::vector<int> pivot(np+nb);
    pdgesv_(n, nrhs, A.data(), 1, 1, descA, &pivot[0], X.data(), 1, 1, descB, &info );
    assert(info==0);
    
    // now call pdgemm to calculate the 1-norm of the residual ||A * X  - B||_1
    pdgemm_( "N", "N", n, nrhs, n, 1., Acopy.data(), 1, 1, descA, X.data(), 1, 1, descB, -1., B.data(), 1, 1, descB);
    int workspace   = numroc_( n, nb, mycol, indxg2p_(1, nb, mycol, 0, npcol ), npcol );
    std::vector<double> work(workspace);
    double rnorm = pdlange_( "1", n, nrhs, B.data(),     1, 1, descB, &work[0]);
    
    if ( rank==0 )
        std::cout <<  "1 norm of residual: " << rnorm << "\n";
    
    // clean up
    Cblacs_gridexit(ctxt);
  }
  else 
    std::cout << "Rank " << rank << " is not used \n";

  // we are done: call MPI_Finalize()
  Cblacs_exit(0);
  
  return 0;
}

