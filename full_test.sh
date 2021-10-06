##############################################

# For ARCHER2 :

module load cpe/21.09
module swap PrgEnv-cray PrgEnv-gnu
# module load cray-fftw


##############################################

##############################################
# Set up
##############################################
PRFX=/work/z19/z19/eleanorb/testing_HPE_libs
cd $PRFX

git clone https://github.com/EPCCed/benchio.git
git clone https://github.com/undees/fftw-example.git

##############################################
# Serial NetCDF and HDF5
##############################################
# module load cray-hdf5
#
# cd $PRFX
# CC simple_hdf5.cpp -o hdf5.x
#
# sbatch archer2_hdf5.slurm
#
# module unload cray-hdf5

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

# CRAY
# eleanorb@ln01:/work/z19/z19/eleanorb/testing_HPE_libs/benchio/shared-file/source> module list
#
# Currently Loaded Modules:
#   1) cce/11.0.4              6) perftools-base/21.02.0                     11) bolt/0.7
#   2) craype/2.7.6            7) xpmem/2.2.40-7.0.1.0_2.7__g1d7a24d.shasta  12) epcc-setup-env
#   3) craype-x86-rome         8) cray-mpich/8.1.4                           13) load-epcc-module
#   4) libfabric/1.11.0.4.71   9) cray-libsci/21.04.1.1                      14) cray-hdf5-parallel/1.12.0.3
#   5) craype-network-ofi     10) PrgEnv-cray/8.0.0                          15) cray-netcdf-hdf5parallel/4.7.4.3

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

# eleanorb@ln01:/work/z19/z19/eleanorb/testing_HPE_libs/fftw-example> module list
#
# Currently Loaded Modules:
#   1) cce/11.0.4              5) craype-network-ofi                          9) cray-libsci/21.04.1.1  13) load-epcc-module
#   2) craype/2.7.6            6) perftools-base/21.02.0                     10) PrgEnv-cray/8.0.0      14) cray-fftw/3.3.8.9
#   3) craype-x86-rome         7) xpmem/2.2.40-7.0.1.0_2.7__g1d7a24d.shasta  11) bolt/0.7
#   4) libfabric/1.11.0.4.71   8) cray-mpich/8.1.4                           12) epcc-setup-env

module unload cray-fftw

##############################################
# Cray LibSci
# Provides: BLAS, LAPACK, CBLAS, LAPACKE, BLACS, ScaLAPACK
##############################################
# cray-libsci is loaded by default


# eleanorb@ln01:/work/z19/z19/eleanorb/testing_HPE_libs> module list
#
# Currently Loaded Modules:
#   1) cce/11.0.4              5) craype-network-ofi                          9) cray-libsci/21.04.1.1  13) load-epcc-module
#   2) craype/2.7.6            6) perftools-base/21.02.0                     10) PrgEnv-cray/8.0.0
#   3) craype-x86-rome         7) xpmem/2.2.40-7.0.1.0_2.7__g1d7a24d.shasta  11) bolt/0.7
#   4) libfabric/1.11.0.4.71   8) cray-mpich/8.1.4                           12) epcc-setup-env


cd $PRFX
ftn blas_lapack_test.f90 -o libsci_f90.x
sbatch archer2_libsci_f90.slurm

cd $PRFX
cc cblas_lapacke_test.c -o libsci_c.x
sbatch archer2_libsci_c.slurm

cp $PRFX/archer2_scalapack.slurm $PRFX/GitLab-examples-MT/examples/mpi
cd $PRFX/GitLab-examples-MT/examples/mpi

module restore PrgEnv-gnu
CC scalapack.cpp -o scalapack.x

sbatch archer2_scalapack.slurm



##############################################
# Checking the slurm scripts
##############################################

cd $PRFX
# ftn blas_lapack_test.f90 -o libsci.x (archer2_libsci_f90.slurm)
# ftn cblas_lapacke_test.c -o libsci.x (archer2_libsci_c.slurm)
############### CC simple_hdf5.cpp -o hdf5.x (archer2_hdf5.slurm)

cd $PRFX/benchio/shared-file/source
# benchio (archer2_netcdf_hdf5_parallel.slurm)

cd $PRFX/fftw-example
# cc fftw_example.c -o fftw.x (archer2_fftw.slurm)

cd $PRFX/GitLab-examples-MT/examples/mpi
# CC scalapack.cpp -o libsci.x (archer2_libsci.slurm)
