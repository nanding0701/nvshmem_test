#include <stdio.h>
#include <assert.h>
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cooperative_groups.h>
#include <sys/time.h>
#include <sched.h>
#define RDMA_FLAG_SIZE 1
#define MAX_MSG_SIZE 2 //2 //512 // put in number of doubles
#define MAX_TB 80
#define TB_SIZE 256
#define MSG_NUM 80
#define _SEND_BY_THREAD

#define RES_ROW 9
#define RES_COL 11
int msg_size[9]={0,3,5,8,11,13,16,18,20};
int msg_inflight[11]={1,2,4,6,8,10,100,1000,10000,100000,1000000};

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define MPI_CHECK(stmt)                                                                         \
    do {                                                                                        \
        int result = (stmt);                                                                    \
        if (MPI_SUCCESS != result) {                                                            \
            fprintf(stderr, "[%s:%d] MPI failed with error %d \n", __FILE__, __LINE__, result); \
            exit(-1);                                                                           \
        }                                                                                       \
    } while (0)

__device__ int clockrate;


__global__ void put_data(double *data_d, uint64_t *flag_d,
                          int len, int msg_sync, int pe, int iter, int peer, int peer_skip, int grid_size, int npes) {

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    uint64_t sig = 1;
 
    //if ((pe==0)&&(bid==0)&&(tid==0)) printf("grid_size=%d, msg_syc=%d\n", grid_size, msg_sync);
    if (grid_size>msg_sync){
            for (int i = 0; i < iter; i++) {
                if (pe == 0) {
                    if (bid<msg_sync) nvshmemx_double_put_signal_nbi_block(&data_d[0], &data_d[0],len,(uint64_t*)(flag_d+bid), sig, NVSHMEM_SIGNAL_SET,peer);
                } 
                nvshmem_quiet();
                nvshmem_barrier_all;
            }
    }else{
        int my_send_num=msg_sync/grid_size;
        int delta=msg_sync%grid_size;
        if (bid<delta) my_send_num = my_send_num+1;
        //if (pe==0 && tid==0) printf("pe=%d, bid=%d, tid=%d,my_send_num=%d\n",pe,bid,tid,my_send_num);
	for (int i = 0; i < (iter); i++) {
            if (pe == 0) {
                for (int j = 0; j < my_send_num; j++) {
                    nvshmemx_double_put_signal_nbi_block(&data_d[0], &data_d[0],len,(uint64_t*)(flag_d+j), sig, NVSHMEM_SIGNAL_SET,peer);
                }
            }
            nvshmem_quiet();
            nvshmem_barrier_all;
        }
    }
    
}

__global__ void put_data_1(double *data_d, uint64_t *flag_d,
                          int len, int msg_sync, int pe, int iter, int peer, int peer_skip, int grid_size, int npes) {

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    uint64_t sig = 1;


        for (int i = 0; i < (iter); i++) {
            if (pe == 0) {
                for (int j = 0; j < msg_sync; j++) {
                    nvshmemx_double_put_signal_nbi_block(&data_d[0], &data_d[0],len,(uint64_t*)(flag_d+j), sig, NVSHMEM_SIGNAL_SET,peer);
                }
            }
            nvshmem_quiet();
            nvshmem_barrier_all;
        }

}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


int main(int c, char *v[]) {
    int color, rank, nranks, sub_rank;
    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;
    int mype, npes, mype_node, ndevices;
    int skip = 100;

    MPI_CHECK(MPI_Init(&c, &v));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    //color = rank/6;
    //MPI_Comm SubComm;
    //MPI_Comm_split(MPI_COMM_WORLD, color, rank, &SubComm);
    //MPI_CHECK(MPI_Comm_rank(SubComm, &sub_rank));

    //mpi_comm = SubComm;
    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    
    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlength;
    MPI_Get_processor_name(name, &resultlength);

    // application picks the device each PE will use
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(rank%ndevices));

    //cudaDeviceProp prop;
    //CUDA_CHECK(cudaGetDeviceProperties(&prop,rank));
    //CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *) &prop.clockRate, sizeof(int), 0,
    //                              cudaMemcpyHostToDevice));
    int get_cur_dev;
    CUDA_CHECK(cudaGetDevice(&get_cur_dev));
	
    printf("IN USE - mpi %d/%d, nvshmem %d/%d , ndevices=%d,cur=%d, node=%s\n", rank, nranks, mype, npes, ndevices, get_cur_dev,  name);
    fflush(stdout);

    nvshmem_barrier_all();
    int iter=atoi(v[1]);
    int flight_num_id=atoi(v[2]);
    int peer_skip=atoi(v[3]);
    int start_peer=atoi(v[4]);
    int start_size_id=atoi(v[5]);


    if (rank==0) printf("iter=%d,start_peer=%d, skip_peer=%d,msgSize=%d,msg_per_syn=%d\n",iter, start_peer, peer_skip, (int) pow(2, msg_size[start_size_id]), msg_inflight[flight_num_id]);
    fflush(stdout);
    for (int peer = start_peer; peer < npes; peer=peer+peer_skip) {
        for (int id = start_size_id; id < RES_ROW; id++) {
            int mysize = (int) pow(2, msg_size[id]);
            char *d_buf=NULL;
            int *flag_d = NULL;
            double *data_d = NULL;
            if (id==0){
                int s_buf_size = (mysize + 1);
                d_buf = (char*) nvshmem_malloc (sizeof(char)*s_buf_size*MAX_TB);
                CUDA_CHECK(cudaMemset(d_buf, 0, sizeof(char)*s_buf_size*MAX_TB));
                if (rank == 0) printf("---- size=%d, malloc %f MB\n", mysize, (s_buf_size * MAX_TB* sizeof(char)) / 1e6);
            }else{   
                int elem=mysize/sizeof(double);
                data_d = (double *)nvshmem_malloc(sizeof(double)*elem*MAX_TB);
                CUDA_CHECK(cudaMemset(data_d, 0, sizeof(double)*elem*MAX_TB));
                if (rank == 0) printf("---- size=%d, elem=%d, MAX_TB=%d, malloc %f MB\n", mysize, elem, MAX_TB, (elem*MAX_TB * sizeof(double)) / 1e6);
            }

            fflush(stdout);

            for (int f = flight_num_id; f < RES_COL; f++) {
                int msg_sync = msg_inflight[f];
                int myiter, myskip;
                myiter=iter;
                while (myiter * msg_sync >= 1e6) {
                    myiter = 0.1*myiter;
                }
                if (f==10) myiter=1;
                if (rank == 0) printf("---- messages per sync %d. iter=%d + %d\n", msg_sync, myiter, myskip);
                flag_d = (int *)nvshmem_malloc(sizeof(int)*msg_sync);
                CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(int)*msg_sync));

                float milliseconds;
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaStream_t stream;
                CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                int minGridSize, myblockSize;
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &myblockSize, (const void *) put_data, 0, 0);
                if (rank==0) printf("minGridSize=%d, myblockSize=%d\n",minGridSize,myblockSize);
                fflush(stdout);
                //if (id == 0) {
                //    void *args[] = {&d_buf, &flag_d, &mysize, &msg_sync, &mype, &myiter, &peer, &peer_skip,&minGridSize, &npes};
                //}else{ 
                    void *args[] = {&data_d, &flag_d, &mysize, &msg_sync, &mype, &myiter,  &peer, &peer_skip,&minGridSize, &npes};
                //} 
                CUDA_CHECK(cudaDeviceSynchronize());
		nvshmem_barrier_all();
                
                cudaEventRecord(start, stream);
                
                int status = nvshmemx_collective_launch((const void *) put_data, minGridSize, myblockSize, args, 0, stream);
                if (status != NVSHMEMX_SUCCESS) {
                    fprintf(stderr, "shmemx_collective_launch minGridSize failed %d \n", status);
                    exit(-1);
                }
                cudaEventRecord(stop, stream);
                /* give latency in us */
                CUDA_CHECK(cudaEventSynchronize(stop));
                cudaEventElapsedTime(&milliseconds, start, stop);
                if (rank==0){
                    printf("size=,%d, block,%d, msg_per_sync,%d,time per sync,%f,iter,%d,skip,%d\n",mysize, minGridSize, msg_sync, (milliseconds * 1000) / myiter, myiter, myskip);
                    fflush(stdout);
                }
                CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(int)*msg_sync));
                CUDA_CHECK(cudaDeviceSynchronize());
                nvshmem_barrier_all();

                
                cudaEventRecord(start, stream);
                status = nvshmemx_collective_launch((const void *) put_data_1, 1, myblockSize, args, 0, stream);
                if (status != NVSHMEMX_SUCCESS) {
                    fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
                    exit(-1);
                }
                cudaEventRecord(stop, stream);
                /* give latency in us */
                CUDA_CHECK(cudaEventSynchronize(stop));
                cudaEventElapsedTime(&milliseconds, start, stop);
                if (rank==0){
                    printf("size=,%d, block,1, msg_per_sync,%d,time per sync,%f,iter,%d,skip,%d\n",mysize, msg_sync, (milliseconds * 1000) / myiter, myiter, myskip);
                    fflush(stdout);
                }
                CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(int)*msg_sync));
                CUDA_CHECK(cudaDeviceSynchronize());
                nvshmem_barrier_all();

                cudaEventRecord(start, stream);
                status = nvshmemx_collective_launch((const void *) put_data_1, 1, myblockSize, args, 0, stream);
                if (status != NVSHMEMX_SUCCESS) {
                    fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
                    exit(-1);
                }
                cudaEventRecord(stop, stream);
                /* give latency in us */
                CUDA_CHECK(cudaEventSynchronize(stop));
                cudaEventElapsedTime(&milliseconds, start, stop);
                if (rank==0){
                    printf("size=,%d, block,1, msg_per_sync,%d,time per sync,%f,iter,%d,skip,%d\n",mysize, msg_sync, (milliseconds * 1000) / myiter, myiter, myskip);
                    fflush(stdout);
                }
                CUDA_CHECK(cudaDeviceSynchronize());



            }// end loop msg_per_sync
        } // end loop size
    } // end loop peer


    MPI_Finalize();
    return 0;
}
