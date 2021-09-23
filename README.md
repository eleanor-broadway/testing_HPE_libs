Basic tests for the HPE provided libraries on ARCHER2
=====================================================

Includes:
- NetCDF parallel
- HDF5 parallel
- FFTW
- LibSci: BLAS
- LibSci: LAPACK

- LibSci: CBLAS
- LibSci: LAPACKE
- LibSci: BLACS
- LibSci: ScaLAPACK
- NetCDF serial
- HDF5 serial

Usage
------
Change the PRFX to your filesystem location, run once for each programming environment. Requires the user to manually check that the outputs work with no errors.


I think that file-per-process is serial.. and shared_source is parallel
