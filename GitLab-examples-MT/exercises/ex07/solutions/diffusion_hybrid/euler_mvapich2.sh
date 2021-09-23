export MV2_ENABLE_AFFINITY=0

export OMP_NUM_THREADS=1
bsub -n 48 -R "span[ptile=24]" -W 30 -o out.48x1.log mpirun -n 48 -ppn 24 ./run.sh ./diffusion2d_hybrid 14
export OMP_NUM_THREADS=2
bsub -n 48 -R "span[ptile=24]" -W 30 -o out.24x2.log mpirun -n 24 -ppn 12 ./run.sh ./diffusion2d_hybrid 14
export OMP_NUM_THREADS=3
bsub -n 48 -R "span[ptile=24]" -W 30 -o out.16x3.log mpirun -n 16 -ppn 8 ./run.sh ./diffusion2d_hybrid 14
export OMP_NUM_THREADS=6
bsub -n 48 -R "span[ptile=24]" -W 30 -o out.8x6.log mpirun -n 8 -ppn 4 ./run.sh ./diffusion2d_hybrid 14
export OMP_NUM_THREADS=12
bsub -n 48 -R "span[ptile=24]" -W 30 -o out.4x12.log mpirun -n 4 -ppn 2 ./run.sh ./diffusion2d_hybrid 14
export OMP_NUM_THREADS=24
bsub -n 48 -R "span[ptile=24]" -W 30 -o out.2x24.log mpirun -n 2 -ppn 1 ./run.sh ./diffusion2d_hybrid 14
