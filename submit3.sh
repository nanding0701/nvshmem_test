#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -A m2956_g
#####SBATCH --mail-user=nanding@lbl.gov
#####SBATCH --mail-type=ALL

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true

export SLURM_CPU_BIND="cores"
export MPICH_GPU_SUPPORT_ENABLED=0

export CRAY_ACCEL_TARGET=nvidia80
echo MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED

#export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/mpi/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH

export NVSHMEM_USE_GDRCOPY=1
export NVSHMEM_MPI_SUPPORT=1
export MPI_HOME=${MPICH_DIR}
 
export NVSHMEM_LIBFABRIC_SUPPORT=1
export LIBFABRIC_HOME=/opt/cray/libfabric/1.15.2.0
export NVSHMEM_HOME=/global/cfs/cdirs/m2956/nanding/software/nvshmem_src_2.9.0-2/build
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
 
export NVSHMEM_DISABLE_CUDA_VMM=1
export FI_CXI_OPTIMIZED_MRS=false

export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
export NVSHMEM_BOOTSTRAP=MPI

#export NVSHMEM_SYMMETRIC_SIZE=2000000000
### iter, flight_num_id, peer_skip, start_peer, start_size_id
srun -n4 -c32 --cpu-bind=cores -G 4 ./examples/obj/mpi-based-init 100000 0 4 1 1
