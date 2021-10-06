Basic tests for the HPE provided libraries on ARCHER2
=====================================================

LibSci:
-------
* Fortran BLAS
* Fortran LAPACK
* C CBLAS
* C LAPACKE
* BLACS Parallel
* ScaLAPACK Parallel

There are two codes, written in fortran and c, to test the various BLAS and LAPACK library interfaces. These do a simple calculation to scale matrix A by factor X, using the BLAS sscal routine, and solve two linear equations using the LAPACK sgevs routine.

Both parallel libraries are tested using an example found on GitLab (https://gitlab.phys.ethz.ch/lwossnig/lecture/-/blob/33dfc1bbd8a47d09a620a058f7d6ddc01781857a/examples/mpi/scalapack.cpp).

FFTW:
----
Extremely simple example of using FFTW, written in c, from GitHub (https://github.com/undees/fftw-example.git).

The code initialises two sine waves with different frequencies and amplitudes. Performs a simple DFT calculation.

HDF5 & NetCDF:
---------------
* **Serial**
* Parallel - Using the BENCHIO (https://github.com/EPCCed/benchio.git)

Usage
------
Change the PRFX to your filesystem location, run once for each programming environment. Requires the user to manually check that the outputs work with no errors.

The "script" is not a script, you will need to manually enter the commands. Treat it like a set of instructions. (Yes I know most of the codes are trash)
