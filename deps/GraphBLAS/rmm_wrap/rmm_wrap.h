//------------------------------------------------------------------------------
// rmm_wrap/rmm_wrap.h: include file for rmm_wrap
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef RMM_WRAP_H
#define RMM_WRAP_H

//#include <cuda_runtime.h>

// FIXME: consider another way to report the error (not std::cout)
#define cudaSucess 0 
#define RMM_WRAP_CHECK_CUDA(call)                                         \
  do {                                                                    \
    int err = call;                                               \
    if (err != cudaSucess) {                                             \
      printf( "(CUDA runtime) returned %d\n", err);                       \
      printf( " ( %s: %d : %s\n", __FILE__,  __LINE__ ,__func__); \
    }                                                                     \
  } while (0)

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>


// TODO describe the modes
typedef enum
{
    rmm_wrap_host = 0,
    rmm_wrap_host_pinned = 1,
    rmm_wrap_device = 2,
    rmm_wrap_managed = 3
}
RMM_MODE ;

// get id of currently selected device
// FIXME: wrong name.  call it rmm_wrap_get_current_device
int get_current_device();

// determine if RMM has been initialized
bool rmm_wrap_is_initialized (void) ;

// create an RMM resource
int rmm_wrap_initialize
(
    uint32_t device_id,
    RMM_MODE mode,
    size_t init_pool_size,
    size_t max_pool_size,
    size_t stream_pool_size
) ;

// initialize rmm_wrap_contexts for each device in CUDA_VISIBLE_DEVICES
// (or single device_id 0 if not specified)
int rmm_wrap_initialize_all_same
(
    RMM_MODE mode,
    size_t init_pool_size,
    size_t max_pool_size,
    size_t stream_pool_size
) ;

// destroy an RMM resource
void rmm_wrap_finalize (void) ;

// The two PMR-based allocate/deallocate signatures (C-style) (based on current device_id):
void *rmm_wrap_allocate (size_t *size) ;
void  rmm_wrap_deallocate (void *p, size_t size) ;

// The four malloc/calloc/realloc/free signatures (based on current device_id):
void *rmm_wrap_malloc (size_t size) ;
void *rmm_wrap_calloc (size_t n, size_t size) ;
void *rmm_wrap_realloc (void *p, size_t newsize) ;
void  rmm_wrap_free (void *p) ;

// Get streams from context (based on current device_id):
void* rmm_wrap_get_next_stream_from_pool(void);
void* rmm_wrap_get_stream_from_pool(size_t stream_id);
void* rmm_wrap_get_main_stream(void);

#ifdef __cplusplus
}
#endif
#endif

