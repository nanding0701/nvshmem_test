#!/bin/bash 

source ./nvshmem_env_perlmutter 

export NVSHMEM_HOME=/global/cfs/cdirs/m2956/nanding/software/nvshmem_src_2.9.0-2/build
export MPICH_GPU_SUPPORT_ENABLED=0
export CRAY_ACCEL_TARGET=nvidia80
export MPI_HOME=${MPICH_DIR}
export NVSHMEM_LIBFABRIC_SUPPORT=1
export LIBFABRIC_HOME=/opt/cray/libfabric/1.15.2.0

export NVSHMEM_DISABLE_CUDA_VMM=1
export FI_CXI_OPTIMIZED_MRS=false

export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_DEBUG=0

export CUDA_HOME=${CUDA_HOME}

make -j16

cd ./perftest
make -j16
