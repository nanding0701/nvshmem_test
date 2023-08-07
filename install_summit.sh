module load gdrcopy
module load cuda

SRC_DIR=$PWD

export CUDA_HOME=$CUDA_DIR
export NVSHMEM_MPI_LIB_NAME=libmpi_ibm.so
export MPI_HOME=$MPI_ROOT
export NVSHMEM_PREFIX=/ccs/home/nanding/mysoftware/nvshmem290_gdr23_cuda1103_20230605
export NVSHMEM_HOME=/ccs/home/nanding/mysoftware/nvshmem290_gdr23_cuda1103_20230605
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_LMPI=-lmpi_ibm
export GDRCOPY_HOME=$OLCF_GDRCOPY_ROOT
export NVSHMEM_USE_GDRCOP=1


########### build nvshmem library
cd $SRC_DIR
make -j8 &&
make install

###### build nvshmem example
cd $SRC_DIR/perftest
#cd $SRC_DIR/examples
make -j8 &&
make install
