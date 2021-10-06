##############################################
module restore PrgEnv-gnu
##############################################

##############################################
# Set up
##############################################
PRFX=/work/z19/z19/eleanorb/test_again_lol/testing_HPE_libs
cd $PRFX

git clone https://github.com/EPCCed/benchio.git
git clone https://github.com/undees/fftw-example.git

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

mv $PRFX/archer2_netcdf_hdf5_parallel.slurm $PRFX/benchio/shared-file/source

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

mv $PRFX/archer2_fftw.slurm $PRFX/fftw-example

module load cray-fftw

cd $PRFX/fftw-example
cc fftw_example.c -o fftw.x
sbatch archer2_fftw.slurm

module unload cray-fftw

##############################################
# Cray LibSci
# Provides: BLAS, LAPACK, CBLAS, LAPACKE, BLACS, ScaLAPACK
##############################################
# cray-libsci is loaded by default

cd $PRFX
ftn blas_lapack_test.f90 -o libsci.x
sbatch archer2_libsci.slurm

cp $PRFX/archer2_libsci.slurm $PRFX/GitLab-examples-MT/examples/mpi
cd $PRFX/GitLab-examples-MT/examples/mpi
CC scalapack.cpp -o libsci.x
sbatch archer2_libsci.slurm



##############################################
# Checking the slurm scripts
##############################################

cd $PRFX
# ftn blas_lapack_test.f90 -o libsci.x (archer2_libsci.slurm)
# CC simple_hdf5.cpp -o hdf5.x (archer2_hdf5.slurm)

cd $PRFX/benchio/shared-file/source
# benchio (archer2_netcdf_hdf5_parallel.slurm)

cd $PRFX/fftw-example
# cc fftw_example.c -o fftw.x (archer2_fftw.slurm)

cd $PRFX/GitLab-examples-MT/examples/mpi
# CC scalapack.cpp -o libsci.x (archer2_libsci.slurm) 
