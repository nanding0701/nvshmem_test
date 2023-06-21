/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_nvtx.hpp"
#include "cpu_coll.h"
#include "nvshmemi_coll.h"
#include "nvshmem_internal.h"

void nvshmemi_quiesce_internal_streams(cudaStream_t cstrm); /* implemented in quiet.cpp */

static void mps_cpu_barrier(volatile atomic<int>& barrier, volatile atomic<bool>& sense, int n) {
    TRACE(NVSHMEM_COLL, "In MPG CPU barrier");
    int count;

    // Check-in
    count = barrier.fetch_add(1, std::memory_order_release) +
            1;       // equivalent to ++barrier with release memory ordering
    if (count == n)  // Last one in
        sense = 1;
    while (!sense)
        ;

    // Check-out
    count = --barrier;
    if (count == 0)  // Last one out
        sense = 0;
    while (sense.load(std::memory_order_acquire))
        ;
}

void nvshmemi_mps_sync_gpu_on_stream(cudaStream_t stream) {
    nvshmemi_mps_shmdata* shm = (nvshmemi_mps_shmdata*)nvshmemi_state->shm_info.addr;

    CUDA_RUNTIME_CHECK(cudaEventRecord(nvshmemi_state->mps_event, stream));
    mps_cpu_barrier(shm->barrier, shm->sense, (int)shm->nprocesses);
    for (int i = 0; i < nvshmemi_team_same_gpu.size - 1; i++)
        CUDA_RUNTIME_CHECK(
            cudaStreamWaitEvent(stream, nvshmemi_state->same_gpu_other_pe_mps_events[i], 0));
    mps_cpu_barrier(
        shm->barrier, shm->sense,
        (int)shm->nprocesses); /* wait for completion (so tthat next barrier works correctly) */
}

int nvshmemxi_sync_on_stream(nvshmem_team_t team, cudaStream_t stream) {
    nvshmemi_call_sync_on_stream_kernel(team, stream);
    return 0;
}

void nvshmemxi_barrier_all_on_stream(cudaStream_t stream) {
    if (nvshmemi_is_limited_mpg_run) {
        nvshmemx_quiet_on_stream(stream);
        nvshmemxi_sync_all_on_stream(stream);
        nvshmemx_quiet_on_stream(stream); /* to do the consistency op */
    } else {
        nvshmemxi_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream);
    }
}

void nvshmemx_barrier_all_on_stream(cudaStream_t stream) {
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
    (*nvshmemi_check_state_and_init_fn_ptr)();
    nvshmemxi_barrier_all_on_stream(stream);
}

void nvshmemxi_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream) {
    TRACE(NVSHMEM_COLL, "In nvshmemxi_barrier_on_stream");
    nvshmemi_quiesce_internal_streams(stream);
    nvshmemi_call_barrier_on_stream_kernel(team, stream);
}

int nvshmemx_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream) {
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();

    nvshmemxi_barrier_on_stream(team, stream);
    return 0;
}

void nvshmemxi_sync_all_on_stream(cudaStream_t stream) {
    if (nvshmemi_is_limited_mpg_run) {
        // sync PEs on the same GPU
        nvshmemi_mps_sync_gpu_on_stream(stream);
        if (nvshmem_team_my_pe(NVSHMEMI_TEAM_SAME_GPU) == 0) {
            nvshmemxi_sync_on_stream(NVSHMEMI_TEAM_GPU_LEADERS, stream);
        }
        nvshmemi_mps_sync_gpu_on_stream(stream);
    } else {
        nvshmemxi_sync_on_stream(NVSHMEM_TEAM_WORLD, stream);
    }
}

void nvshmemx_sync_all_on_stream(cudaStream_t stream) {
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
    (*nvshmemi_check_state_and_init_fn_ptr)();
    nvshmemxi_sync_all_on_stream(stream);
}

int nvshmemx_team_sync_on_stream(nvshmem_team_t team, cudaStream_t stream) {
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();
    nvshmemxi_sync_on_stream(team, stream);
    return 0;
}
