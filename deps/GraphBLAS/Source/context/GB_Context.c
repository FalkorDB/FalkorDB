//------------------------------------------------------------------------------
// GB_Context.c: Context object for computational resources
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GxB_Context object contains the set of resources that a user thread
// can use.  There are two kinds of Contexts:

// GxB_CONTEXT_WORLD:  this Context always exists and its contents are always
// defined.  If a user thread has no context, it uses this Context.  It is
// user-visible since its contents may be changed/read by the user application,
// via GxB_Context_set/get.

// GB_CONTEXT_THREAD:  this context is thread-private to each user thread, and
// only visible within this file.  It is not directly accessible by any user
// application.  It is not even visible to other functions inside SuiteSparse:
// GraphBLAS.  If the user thread has not engaged any Context, then
// GB_CONTEXT_THREAD is NULL.  If the compiler does not support
// thread-local-stoage, then GB_CONTEXT_THREAD is always NULL and cannot be
// modified; in this case, GxB_Context_engage can return GrB_NOT_IMPLEMENTED.

#include "GB.h"

#if defined ( _OPENMP )

    // OpenMP threadprivate is preferred
    GxB_Context GB_CONTEXT_THREAD = NULL ;
    #pragma omp threadprivate (GB_CONTEXT_THREAD)

#elif defined ( HAVE_KEYWORD__THREAD )

    // gcc and many other compilers support the __thread keyword
    __thread GxB_Context GB_CONTEXT_THREAD = NULL ;

#elif defined ( HAVE_KEYWORD__DECLSPEC_THREAD )

    // Windows: __declspec (thread)
    __declspec ( thread ) GxB_Context GB_CONTEXT_THREAD = NULL ;

#elif defined ( HAVE_KEYWORD__THREAD_LOCAL )

    // C11 threads
    #include <threads.h>
    _Thread_local GxB_Context GB_CONTEXT_THREAD = NULL ;

#else

    // GraphBLAS will not be thread-safe when using a GxB_Context other than
    // GxB_CONTEXT_WORLD, so GxB_Context_engage returns GrB_NOT_IMPLEMENTED
    // if passed a Context other than GxB_CONTEXT_WORLD or NULL.
    #define NO_THREAD_LOCAL_STORAGE
    #define GB_CONTEXT_THREAD NULL

#endif

//------------------------------------------------------------------------------
// GB_Context_engage: engage the Context for a user thread
//------------------------------------------------------------------------------

GrB_Info GB_Context_engage (GxB_Context Context)
{ 
    if (Context == GxB_CONTEXT_WORLD)
    { 
        // GxB_Context_engage (GxB_CONTEXT_WORLD) is the same as engaging
        // NULL as the user thread context.
        Context = NULL ;
    }
    #if defined ( NO_THREAD_LOCAL_STORAGE )
    return ((Context == NULL) ? GrB_SUCCESS : GrB_NOT_IMPLEMENTED) ;
    #else
    GB_CONTEXT_THREAD = Context ;
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// GB_Context_disengage: disengage the Context for a user thread
//------------------------------------------------------------------------------

GrB_Info GB_Context_disengage (GxB_Context Context)
{
    #if defined ( NO_THREAD_LOCAL_STORAGE )
        // nothing to do
        return (GrB_SUCCESS) ;
    #else
        if (Context == NULL || Context == GB_CONTEXT_THREAD ||
            GB_CONTEXT_THREAD == NULL || Context == GxB_CONTEXT_WORLD)
        { 
            // If no Context provided on input: simply disengage whatever the
            // current Context is for this user thread.  If a non-NULL context
            // is provided and the current GB_CONTEXT_THREAD is not NULL, it
            // must match the Context that is currently engaged to this user
            // thread to be disengaged.
            GB_CONTEXT_THREAD = NULL ;
            return (GrB_SUCCESS) ;
        }
        else
        { 
            // A non-NULL Context was provided on input, but it doesn't match
            // the currently engaged Context.  This is an error.
            return (GrB_INVALID_VALUE) ;
        }
    #endif
}

//------------------------------------------------------------------------------
// Context->nthreads_max: # of OpenMP threads to use
//------------------------------------------------------------------------------

// GB_Context_nthreads_max_get: get max # of threads from a Context
int GB_Context_nthreads_max_get (GxB_Context Context)
{
    int nthreads_max ;
    if (Context == NULL || Context == GxB_CONTEXT_WORLD)
    { 
        GB_ATOMIC_READ
        nthreads_max = GxB_CONTEXT_WORLD->nthreads_max ;
    }
    else
    { 
        nthreads_max = Context->nthreads_max ;
    }
    return (nthreads_max) ;
}

// GB_Context_nthreads_max: get max # of threads from the current Context
int GB_Context_nthreads_max (void)
{ 
    // This method is used by most GraphBLAS functions to determine the # of
    // threads to use.  If a Context is engaged, it uses the engaged context.
    // Otherwise, it uses the default GxB_CONTEXT_WORLD.
    return (GB_Context_nthreads_max_get (GB_CONTEXT_THREAD)) ;
}

// GB_Context_nthreads_max_set: set max # of threads in a Context
void GB_Context_nthreads_max_set
(
    GxB_Context Context,
    int nthreads_max
)
{
    nthreads_max = GB_IMAX (1, nthreads_max) ;
    if (Context == NULL || Context == GxB_CONTEXT_WORLD)
    { 
        GB_ATOMIC_WRITE
        GxB_CONTEXT_WORLD->nthreads_max = nthreads_max ;
    }
    else
    { 
        Context->nthreads_max = nthreads_max ;
    }
}

//------------------------------------------------------------------------------
// Context->chunk: controls # of threads used for small problems
//------------------------------------------------------------------------------

// GB_Context_chunk_get: get chunk from a Context
double GB_Context_chunk_get (GxB_Context Context)
{
    double chunk ;
    if (Context == NULL || Context == GxB_CONTEXT_WORLD)
    { 
        GB_ATOMIC_READ
        chunk = GxB_CONTEXT_WORLD->chunk ;
    }
    else
    { 
        chunk = Context->chunk ;
    }
    return (chunk) ;
}

// GB_Context_chunk: get chunk from the Context of this user thread
double GB_Context_chunk (void)
{ 
    // This method is used by most GraphBLAS functions to determine the chunk
    // parameter.  If a Context is engaged, it uses the engaged context.
    // Otherwise, it uses the default GxB_CONTEXT_WORLD.
    return (GB_Context_chunk_get (GB_CONTEXT_THREAD)) ;
}

// GB_Context_chunk_set: set max # of threads in a Context
void GB_Context_chunk_set
(
    GxB_Context Context,
    double chunk
)
{
    if (chunk < 1)
    { 
        chunk = GB_CHUNK_DEFAULT ;
    }
    if (Context == NULL || Context == GxB_CONTEXT_WORLD)
    { 
        GB_ATOMIC_WRITE
        GxB_CONTEXT_WORLD->chunk = chunk ;
    }
    else
    { 
        Context->chunk = chunk ;
    }
}

//------------------------------------------------------------------------------
// Context->ngpus and Context->gpu_ids: which GPU(s) to use
//------------------------------------------------------------------------------

// GB_Context_gpu_ids_get: get list of GPUs to use from a Context
int32_t GB_Context_gpu_ids_get          // return # of GPUs to use
(
    GxB_Context Context,
    int32_t gpu_ids [GB_MAX_NGPUS]      // list of GPU ids to use
)
{
    if (Context == NULL)
    {
        Context = GxB_CONTEXT_WORLD ;
    }
    int32_t ngpus = Context->ngpus ;
    ngpus = GB_IMIN (ngpus, GB_MAX_NGPUS) ;
    ngpus = GB_IMAX (ngpus, 0) ;
    if (gpu_ids != NULL)
    {
        for (int32_t k = 0 ; k < ngpus ; k++)
        { 
            gpu_ids [k] = (int32_t) Context->gpu_ids [k] ;
        }
    }
    return (ngpus) ;
}

// GB_Context_gpu_ids: get list of GPUs from the current Context
int32_t GB_Context_gpu_ids              // return # of GPUs to use
(
    int32_t gpu_ids [GB_MAX_NGPUS]      // list of GPU ids to use
)
{ 
    // FUTURE: use this in all CUDA kernels
    // This method is used by most GraphBLAS functions to determine the
    // gpu(s) to use.  If a Context is engaged, it uses the engaged context.
    // Otherwise, it uses the default GxB_CONTEXT_WORLD.
    return (GB_Context_gpu_ids_get (GB_CONTEXT_THREAD, gpu_ids)) ;
}

// GB_Context_gpu_ids_set: set list of GPUs in a Context
GrB_Info GB_Context_gpu_ids_set
(
    GxB_Context Context,
    int32_t gpu_ids [GB_MAX_NGPUS],     // list of GPU ids to use
    int32_t ngpus                       // # of GPUs to use
)
{
    if (Context == NULL)
    {
        Context = GxB_CONTEXT_WORLD ;
    }
    int32_t ngpus_max = GB_Global_gpu_count_get ( ) ;
    if (ngpus < 0 || ngpus > ngpus_max)
    { 
        return (GrB_INVALID_VALUE) ;
    }
    Context->ngpus = ngpus ;
    if (gpu_ids == NULL)
    {
        // use GPUs with ids 0 to ngpus-1
        for (int32_t id = 0 ; id < ngpus ; id++)
        {
            Context->gpu_ids [id] = (uint8_t) id ;
        }
    }
    else
    {
        for (int32_t k = 0 ; k < ngpus ; k++)
        {
            // get the GPU id and ensure it is in range 0:ngpus-1
            int32_t id = gpu_ids [k] ;
            if (id < 0 || id >= ngpus_max)
            { 
                return (GrB_INVALID_VALUE) ;
            }
            Context->gpu_ids [k] = (uint8_t) id ;
        }
    }
    return (GrB_SUCCESS) ;
}

