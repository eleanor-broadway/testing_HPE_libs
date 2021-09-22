##############################################
PRFX=/work/z19/z19/eleanorb/testing_HPE
cd $PRFX

module restore PrgEnv-gnu
##############################################

git clone https://github.com/EPCCed/benchio.git
git clone https://github.com/undees/fftw-example.git

mv archer2_netcdf_hdf5_parallel.slurm $PRFX/benchio/shared-file/source
mv archer2_fftw.slurm $PRFX/fftw-example

##############################################
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel
module load cray-fftw


##############################################
# Parallel NetCDF and HDF5
##############################################

cd $PRFX/benchio/shared-file/source
make clean
make

mkdir -p benchio_files
lfs setstripe -c -1 benchio_files

sbatch archer2_netcdf_hdf5_parallel.slurm


##############################################
# FFTW
##############################################

cd $PRFX/fftw-example
CC fftw_example.c -o fftw.x
sbatch archer2_fftw.slurm
