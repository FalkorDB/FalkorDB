//------------------------------------------------------------------------------
// GB_Global.h: definitions for global data
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These definitions are not visible to the user.  They are used only inside
// GraphBLAS itself.  Note that the GB_Global struct does not appear here.
// It is accessible only by the functions in GB_Global.c.

#ifndef GB_GLOBAL_H
#define GB_GLOBAL_H

typedef void * (*GB_malloc_function_t ) (size_t) ;
typedef void * (*GB_calloc_function_t ) (size_t, size_t) ;
typedef void * (*GB_realloc_function_t) (void *, size_t) ;
typedef void   (*GB_free_function_t) (void *) ;

void     GB_Global_cpu_features_query (void) ;
bool     GB_Global_cpu_features_avx2 (void) ;
bool     GB_Global_cpu_features_avx512f (void) ;
bool     GB_Global_cpu_features_rvv_1_0 (void) ;

void     GB_Global_mode_set (int mode) ;
int      GB_Global_mode_get (void) ;

void     GB_Global_sort_set (int sort) ;
int      GB_Global_sort_get (void) ;

void     GB_Global_GrB_init_called_set (bool init_called) ;
bool     GB_Global_GrB_init_called_get (void) ;

void     GB_Global_hyper_switch_set (float hyper_switch) ;
float    GB_Global_hyper_switch_get (void) ;

void     GB_Global_hyper_hash_set (int64_t hyper_hash) ;
int64_t  GB_Global_hyper_hash_get (void) ;

void     GB_Global_bitmap_switch_set (int k, float b) ;
float    GB_Global_bitmap_switch_get (int k) ;
float    GB_Global_bitmap_switch_matrix_get (int64_t vlen, int64_t vdim) ;
void     GB_Global_bitmap_switch_default (void) ;

void     GB_Global_is_csc_set (bool is_csc) ;
bool     GB_Global_is_csc_get (void) ;

void     GB_Global_abort_set (void (* abort_function) (void)) ;
void     GB_Global_abort (void) ;

void     GB_Global_malloc_function_set (GB_malloc_function_t malloc_function, int memlane) ;
void  *  GB_Global_malloc_function_get (int memlane) ;
void  *  GB_Global_malloc_function (uint64_t memsize, int memlane) ;

void     GB_Global_calloc_function_set (GB_calloc_function_t calloc_function, int memlane) ;
void  *  GB_Global_calloc_function_get (int memlane) ;

void     GB_Global_realloc_function_set (GB_realloc_function_t realloc_function, int memlane) ;
void  *  GB_Global_realloc_function_get (int memlane) ;
void  *  GB_Global_realloc_function (void *p, uint64_t memsize, int memlane) ;
bool     GB_Global_realloc_function_have (int memlane) ;

void     GB_Global_free_function_set (GB_free_function_t free_function, int memlane) ;
void  *  GB_Global_free_function_get (int memlane) ;
void     GB_Global_free_function (void *p, int memlane) ;

void     GB_Global_malloc_is_thread_safe_set (bool malloc_is_thread_safe, int memlane) ;
bool     GB_Global_malloc_is_thread_safe_get (int memlane) ;

void     GB_Global_malloc_tracking_set (bool malloc_tracking) ;
bool     GB_Global_malloc_tracking_get (void) ;

void     GB_Global_nmalloc_clear (void) ;
int64_t  GB_Global_nmalloc_get (void) ;

void     GB_Global_malloc_debug_set (bool malloc_debug) ;
bool     GB_Global_malloc_debug_get (void) ;

void     GB_Global_malloc_debug_count_set (int64_t malloc_debug_count) ;
bool     GB_Global_malloc_debug_count_decrement (void) ;

void *   GB_Global_persistent_malloc (uint64_t memsize) ;
void     GB_Global_persistent_make (void *p) ;
void     GB_Global_persistent_set (void (* persistent_function) (void *)) ;
void     GB_Global_persistent_free (void **p) ;

void     GB_Global_hack_set (int k, int64_t hack) ;
int64_t  GB_Global_hack_get (int k) ;

void     GB_Global_burble_set (bool burble) ;
bool     GB_Global_burble_get (void) ;

void     GB_Global_print_one_based_set (bool onebased) ;
bool     GB_Global_print_one_based_get (void) ;

void     GB_Global_stats_mem_shallow_set (bool mem_shallow) ;
bool     GB_Global_stats_mem_shallow_get (void) ;

void     GB_Global_gpu_count_set (bool enable_cuda) ;
int      GB_Global_gpu_count_get (void) ;
uint64_t GB_Global_gpu_memorysize_get (int device) ;
int      GB_Global_gpu_sm_get (int device) ;
bool     GB_Global_gpu_device_pool_memsize_set (int device, uint64_t gpusize) ;
bool     GB_Global_gpu_device_max_pool_memsize_set (int device, uint64_t gpusize) ;
bool     GB_Global_gpu_device_memory_resource_set (int device, void *resource) ;
void*    GB_Global_gpu_device_memory_resource_get (int device) ;
bool     GB_Global_gpu_device_properties_get (int device) ;
int      GB_Global_gpu_compute_capability_major_get (int device) ;
int      GB_Global_gpu_compute_capability_minor_get (int device) ;

void     GB_Global_timing_clear_all (void) ;
void     GB_Global_timing_clear (int k) ;
void     GB_Global_timing_set (int k, double t) ;
void     GB_Global_timing_add (int k, double t) ; 
double   GB_Global_timing_get (int k) ;

int      GB_Global_memtable_n (void) ;
void     GB_Global_memtable_dump (void) ;
void     GB_Global_memtable_clear (void) ;
void     GB_Global_memtable_add (void *p, uint64_t mem) ;
uint64_t GB_Global_memtable_memsize (void *p) ;
int      GB_Global_memtable_memlane (void *p) ;
void     GB_Global_memtable_remove (void *p) ;
bool     GB_Global_memtable_find (void *p) ;

typedef int (* GB_flush_function_t) (void) ;
typedef int (* GB_printf_function_t) (const char *restrict format, ...) ;

GB_printf_function_t GB_Global_printf_get (void) ;
void     GB_Global_printf_set (GB_printf_function_t p) ;

GB_flush_function_t GB_Global_flush_get (void) ;
void     GB_Global_flush_set (GB_flush_function_t p) ;

void     GB_Global_p_control_set (int8_t p_control) ;
int8_t   GB_Global_p_control_get (void) ;
void     GB_Global_j_control_set (int8_t j_control) ;
int8_t   GB_Global_j_control_get (void) ;
void     GB_Global_i_control_set (int8_t i_control) ;
int8_t   GB_Global_i_control_get (void) ;

void     GB_Global_lock_init (void) ;
void     GB_Global_lock_destroy (void) ;
void     GB_Global_lock_set (int k) ;
void     GB_Global_lock_unset (int k) ;
void     GB_Global_lock_wipe (void) ;
#endif

