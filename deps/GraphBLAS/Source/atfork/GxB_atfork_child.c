//------------------------------------------------------------------------------
// GxB_atfork_child: actions GraphBLAS must take in a forked child process
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If a process using GraphBLAS calls the POSIX fork() method, the new child
// must restrict itself to using only async-signal-safe methods.  See
// https://man7.org/linux/man-pages/man7/signal-safety.7.html .  Thus, the
// child process cannot use the following:

// OpenMP: using omp_lock_set/unset is not possible.  OpenMP is in general not
//      async-signal-safe at all, so no OpenMP parallelism is possible.  The
//      locks protect GraphBLAS global variables from multiple user threads, so
//      without them, GraphBLAS is not thread-safe.  The child process cannot
//      create its own pthreads or OpenMP threads, as a result.  This method
//      disables the use of all omp_lock variables and forces OpenMP to use a
//      single thread in all of the parallel regions in GraphBLAS..

//          FIXME: OpenMP is always used in GraphBLAS via a num_threads clause:
//
//              #pragma omp parallel ... num_threads(nthreads)
//
//          where this method ensures nthreads=1.  This seems to work, but it
//          is still not fully safe.  A complete fix would return GrB_PANIC or
//          some other error if any OpenMP parallel regions are attempted.
//          This will require many tests in many source files.

// malloc/calloc/realloc/free: these are not safe.  This method forces the
//      child to use GB_child_malloc and GB_child_free instead of the existing
//      methods.  The GB_child_malloc always returns NULL, so that any GrB
//      method needing it use will return GrB_OUT_OF_MEMORY.

// thread-local-storage:  this is normally provided by OpenMP and thus is not
//      safe to use.  It is only used by the GxB_Context methods, which
//      controls the # of OpenMP threads to use and which GPUs to use.  Neither
//      are safe to use, so all Context objects are disabled; OpenMP will use
//      a single thread, and no GPUs will be used.

// printf/fflush: these are not safe.  They are only used by the burble and by
//      GxB_*print methods.  This method disables them.

// dlopen/dlclose/dlsym: are not safe; this method disables the loading or
//      compiling of any new JIT kernels.  Existing kernels (loaded JIT kernels
//      or pre-compiled PreJIT kernels) may continue to be safely used.

// CUDA: this is not safe; this method disables all GPUs.  TODO: how does the
//      CPU get access to unified-shared-memory currently on the GPU?

// This method is suitable for passing to pthread_atfork
// (see https://man7.org/linux/man-pages/man3/pthread_atfork.3.html ) as the
// 3rd parameter.  Thus, unlike most GraphBLAS methods, this method returns
// void.  The child must call this immediately after it is created, either
// via pthread_atfork or by calling it directly (which is suitable for use in
// non-POSIX systems such as Windows).

//------------------------------------------------------------------------------

#include "GB_atfork.h"

//------------------------------------------------------------------------------
// GB_child_*:  helper functions for use in the child process
//------------------------------------------------------------------------------

void *GB_child_malloc (size_t size)
{
    // the child may not safely malloc anything.  If it tries, the GrB_* method
    // will return GrB_OUT_OF_MEMORY.
    return (NULL) ;
}

void GB_child_free (void *p)
{
    // the child cannot free anything, so do nothing
    return ;
}

int GB_child_printf (const char *restrict format, ...)
{
    // the child cannot call printf
    return (0) ;
}

int GB_child_flush (void)
{
    // the child cannot call fflush
    return (0) ;
}

//------------------------------------------------------------------------------
// GxB_atfork_child
//------------------------------------------------------------------------------

void GxB_atfork_child (void)
{
    // wipe all OpenMP locks; ensures omp_lock_set/unset will not be called
    GB_Global_lock_wipe ( ) ;

    // use the new malloc/free, defined above
    GB_Global_malloc_function_set (GB_child_malloc, 0) ;
    GB_Global_malloc_function_set (GB_child_malloc, 1) ;
    GB_Global_free_function_set (GB_child_free, 0) ;
    GB_Global_free_function_set (GB_child_free, 1) ;
    GB_Global_malloc_is_thread_safe_set (true, 0) ;
    GB_Global_malloc_is_thread_safe_set (true, 1) ;

    // calloc and realloc can be NULL; GraphBLAS will call malloc instead
    GB_Global_realloc_function_set ((GB_realloc_function_t) NULL, 0) ;
    GB_Global_realloc_function_set ((GB_realloc_function_t) NULL, 1) ;
    GB_Global_calloc_function_set  ((GB_calloc_function_t ) NULL, 0) ;
    GB_Global_calloc_function_set  ((GB_calloc_function_t ) NULL, 1) ;

    // turn off any malloc debugging
    GB_Global_malloc_tracking_set (false) ;
    GB_Global_malloc_debug_set (false) ;
    GB_Global_nmalloc_clear () ;

    // disable the burble
    GB_Global_burble_set (false) ;

    // disable printf and flush, used in the burble and GxB_*print methods
    GB_Global_printf_set (GB_child_printf) ;
    GB_Global_flush_set (GB_child_flush) ;

    // set the JIT so it cannot load or compile and new kernels
    int jit_control = GB_jitifyer_get_control ( ) ;
    jit_control = GB_IMIN (jit_control, GxB_JIT_RUN) ;
    GB_jitifyer_set_control (jit_control) ;

    // ensure only a single OpenMP thread is used in GraphBLAS
    GB_Context_nthreads_max_set (NULL, 1) ;

    // disable all Context objects; use 1 thread, default chunk, no gpus
    GB_Context_disable ( ) ;
}

