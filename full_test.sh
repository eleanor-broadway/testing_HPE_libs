##############################################
module restore PrgEnv-gnu
##############################################

##############################################
# Set up
##############################################
PRFX=/work/z19/z19/eleanorb/testing_HPE
cd $PRFX

git clone https://github.com/EPCCed/benchio.git
git clone https://github.com/undees/fftw-example.git

mv archer2_netcdf_hdf5_parallel.slurm $PRFX/benchio/shared-file/source
mv archer2_fftw.slurm $PRFX/fftw-example

##############################################
# Serial NetCDF and HDF5
##############################################
module load cray-hdf5

cd $PRFX
CC simple_hdf5.cpp -o hdf5.x

sbatch archer2_hdf5.slurm

module unload cray-hdf5

##############################################
# Parallel NetCDF and HDF5
##############################################
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel

cd $PRFX/benchio/shared-file/source
make clean
make

mkdir -p benchio_files
lfs setstripe -c -1 benchio_files

sbatch archer2_netcdf_hdf5_parallel.slurm

module unload cray-hdf5-parallel
module unload cray-netcdf-hdf5parallel


##############################################
# FFTW
##############################################
module load cray-fftw

cd $PRFX/fftw-example
CC fftw_example.c -o fftw.x
sbatch archer2_fftw.slurm

module unload cray-fftw

##############################################
# Cray LibSci
# Provides: BLAS, LAPACK, CBLAS, LAPACKE, BLACS, ScaLAPACK
##############################################
# cray-libsci is loaded by default

cd $PRFX
ftn blas_lapack_test.f90 -o blas_lapack.x
sbatch archer2_libsci.slurm
