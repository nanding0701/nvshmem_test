/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmemi_util.h"
#include "nvshmemi_coll.h"
#include "gpu_coll.h"
#include "barrier.h"
#include "device/coll/barrier.cuh"

__global__ void barrier_on_stream_kernel(int start, int stride, int size, long *pSync,
                                         long *counter);
__global__ void barrier_on_stream_kernel_warp(int start, int stride, int size, long *pSync,
                                              long *counter);
__global__ void barrier_on_stream_kernel_block(int start, int stride, int size, long *pSync,
                                               long *counter);

__global__ void sync_on_stream_kernel(int start, int stride, int size, long *pSync, long *counter);
__global__ void sync_on_stream_kernel_warp(int start, int stride, int size, long *pSync,
                                           long *counter);
__global__ void sync_on_stream_kernel_block(int start, int stride, int size, long *pSync,
                                            long *counter);
__global__ void sync_all_on_stream_kernel();
__global__ void sync_all_on_stream_kernel_warp();
__global__ void sync_all_on_stream_kernel_block();

template <threadgroup_t SCOPE>
__global__ void barrier_on_stream_kernel_threadgroup(nvshmem_team_t team, int in_cuda_graph) {
#ifdef __CUDA_ARCH__
    int myidx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    if (nvshmemi_device_state_d.job_connectivity >= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
        nvshmemi_transfer_quiet<SCOPE>(false);
    }
    if (in_cuda_graph) {
        nvshmemi_threadgroup_sync<SCOPE>();
        if (!myidx) __threadfence_system();
        nvshmemi_threadgroup_sync<SCOPE>();
    }

    nvshmemi_sync_algo_threadgroup<SCOPE>(team);

    if (!myidx) {
        if (nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_PROXY)
            nvshmemi_transfer_enforce_consistency_at_target(false);
    }
#endif
}

template <threadgroup_t SCOPE>
__global__ void sync_on_stream_kernel_threadgroup(nvshmem_team_t team) {
#ifdef __CUDA_ARCH__
    nvshmemi_sync_algo_threadgroup<SCOPE>(team);
#endif
}

int nvshmemi_call_barrier_on_stream_kernel(nvshmem_team_t team, cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
        int size = nvshmemi_team_pool[team]->size;
        num_threads_per_block = size - 1;  // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }

    int in_cuda_graph = 0;
#if CUDART_VERSION >= 10000  // cudaStreamIsCapturing
    cudaStreamCaptureStatus status;
    CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) in_cuda_graph = 1;
#endif

    if (num_threads_per_block <= 32) {
        barrier_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_WARP>
            <<<num_blocks, 32, 0, stream>>>(team, in_cuda_graph);
    } else {
        barrier_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>
            <<<num_blocks, num_threads_per_block, 0, stream>>>(team, in_cuda_graph);
    }
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}

int nvshmemi_call_sync_on_stream_kernel(nvshmem_team_t team, cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
        int size = nvshmemi_team_pool[team]->size;
        num_threads_per_block = size - 1;  // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }

    if (num_threads_per_block <= 32) {
        sync_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_WARP>
            <<<num_blocks, 32, 0, stream>>>(team);
    } else {
        sync_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>
            <<<num_blocks, num_threads_per_block, 0, stream>>>(team);
    }
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}
