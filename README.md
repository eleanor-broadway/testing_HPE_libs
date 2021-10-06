Basic tests for the HPE provided libraries on ARCHER2
=====================================================

Includes:
- NetCDF parallel - Benchio
- HDF5 parallel - Benchio
- FFTW - Example from github
- LibSci: BLAS (ftn) - My own code
- LibSci: LAPACK (ftn) - My own code
- LibSci: ScaLAPACK (parallel) - Testing these with c++, GitLab examples/scalapack.cpp
- LibSci: BLACS (parallel) - Testing these with c++, GitLab examples/scalapack.cpp
- HDF5 serial


- LibSci: CBLAS (C)
- LibSci: LAPACKE (C)
- NetCDF serial

Usage
------
Change the PRFX to your filesystem location, run once for each programming environment. Requires the user to manually check that the outputs work with no errors.
