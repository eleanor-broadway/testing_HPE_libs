#include <stdio.h>
#include <cblas.h>
#include <lapacke.h>

int
main ()
{
  int i;
  double blas_array[3] = { 1, 2, 3 };
  double lapack_A[4] = {
    3, -1,
    1, 6
  };
  double lapack_B[2] = { 4, 6 };
  int pivot[2];

  int ok;

  /////////////////////////////////////
  // X = a*X
  /////////////////////////////////////
  printf ("\n");
  printf ("Testing the CBLAS library");
  printf ("\n");

  for (i = 0; i < 3; ++i)
  printf ("%f\n", blas_array[i]);

  cblas_dscal(3, 5, blas_array, 1);

  printf ("\n");
  printf ("Scaling the array by 5");
  printf ("\n");

  for (i = 0; i < 3; ++i)
    printf ("%f\n", blas_array[i]);

  /////////////////////////////////////
  // A*X = B
  // Where A, X and B are matrices
  // A = 3.0 -1.0  B = 4.0
  //     1.0  6.0      6.0
  /////////////////////////////////////
  printf ("\n");
  printf ("Testing the CLAPACKE library");
  printf ("\n");
  printf ("Solving a set of linear equations");
  printf ("\n");

  LAPACKE_dgesv(LAPACK_ROW_MAJOR, 2, 1, lapack_A, 2, pivot, lapack_B, 1);

  printf ("Correct answer: 1.57894742,  0.736842036");
  printf ("\n");

  for (i = 0; i < 2; ++i)
    printf ("%f\n", lapack_B[i]);




return 0;
} 
