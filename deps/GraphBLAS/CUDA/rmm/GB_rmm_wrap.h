//------------------------------------------------------------------------------
// CUDA/rmm/GB_rmm_wrap.h: include file for GB_rmm_wrap
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef RMM_WRAP_H
#define RMM_WRAP_H

#ifdef __cplusplus
extern "C" {
#endif

#if 0
typedef enum
{
    rmm_wrap_host = 0,
    rmm_wrap_host_pinned = 1,
    rmm_wrap_device = 2,
    rmm_wrap_managed = 3
}
RMM_MODE ;
#endif

// determine if RMM has been initialized
bool rmm_wrap_is_initialized (void) ;

// create an RMM resource
int rmm_wrap_initialize
(
    int device_id
//  , RMM_MODE mode,
//  size_t init_pool_memsize,
//  size_t max_pool_memsize
) ;

// initialize rmm_wrap_contexts for each device in CUDA_VISIBLE_DEVICES
// (or single device_id 0 if not specified)
int rmm_wrap_initialize_all_same
(
    void
//  RMM_MODE mode,
//  size_t init_pool_memsize,
//  size_t max_pool_memsize
) ;

// destroy an RMM resource
void rmm_wrap_finalize (void) ;

// allocate/deallocate methods
void *rmm_allocate   (int device_id, size_t size) ;
void  rmm_deallocate (int device_id, void *p) ;

#ifdef __cplusplus
}
#endif
#endif

