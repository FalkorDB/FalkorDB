//------------------------------------------------------------------------------
// GB_rmm_wrap.cpp: C-callable wrapper for RMM memory resources
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_rmm_wrap.cpp contains a single global object, the RMM_Wrap_Handle that
// holds an RMM (Rapids Memory Manager) memory resource and a hash map (C++
// std:unordered_map).  This allows GB_rmm_wrap to provide the following
// functions to GraphBLAS:

// Create/destroy an RMM resource:
//      rmm_wrap_initialize: create the RMM resource
//      rmm_wrap_is_initialized: query if the RMM resource has been created
//      rmm_wrap_finalize: destroy the RMM resource

// C-style malloc/free methods:
//      GB_rmm_malloc_#:  malloc a block of memory using RMM on device (#)
//      GB_rmm_free_#:    free a block of memory using RMM on device (#)

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

#include "stddef.h"
#include <cuda.h>
//#include <rmm/cuda_stream.hpp>
//#include <rmm/device_buffer.hpp>
//#include <rmm/detail/aligned.hpp>
// #include <rmm/mr/host/host_memory_resource.hpp>
// #include <rmm/mr/host/new_delete_resource.hpp>
// #include <rmm/mr/host/pinned_memory_resource.hpp>
// #include <rmm/mr/device/owning_wrapper.hpp>
// #include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
// #include <rmm/mr/device/polymorphic_allocator.hpp>
// #include <rmm/mr/device/thread_safe_resource_adaptor.hpp>
// #include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
// #include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
// #include <pool_memory_resource.hpp>
// #include <rmm/mr/device/limiting_resource_adaptor.hpp>
// #include <rmm/cuda_stream_view.hpp>
// #include <rmm/cuda_stream_pool.hpp>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <unordered_map>

// typedef rmm::mr::new_delete_resource host_mr;
// typedef rmm::mr::pinned_memory_resource pinned_mr;
// typedef rmm::mr::cuda_memory_resource device_mr;
// typedef rmm::mr::managed_memory_resource managed_mr;
// typedef rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr;
// typedef rmm::mr::pool_memory_resource pool{cuda_mr,size} pool_mr;

// typedef rmm::mr::pool_memory_resource<host_mr> host_pool_mr;
// typedef rmm::mr::pool_memory_resource<pinned_mr> host_pinned_pool_mr;
// typedef rmm::mr::pool_memory_resource<device_mr> device_pool_mr;
// typedef rmm::mr::pool_memory_resource<managed_mr> managed_pool_mr;

typedef std::unordered_map< std::size_t, std::size_t> alloc_map;

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>

//------------------------------------------------------------------------------
// RMM_Wrap_Handle: a global object containing the RMM context
//------------------------------------------------------------------------------

// NOTE: these methods are not thread-safe

// rmm_wrap_context is a pointer to an array of global RMM_Wrap_Handle objects
// (one per GPU) that all methods in this file can access.  The array of
// objects cannot be accessed outside this file.

cudaStream_t rmm_wrap_global_stream = nullptr ;

rmm::mr::managed_memory_resource cuda_mr_default ;
rmm::mr::pool_memory_resource    cuda_pool_default (cuda_mr_default, 0) ;

typedef struct RMM_Wrap_Handle_struct
{
//  std::shared_ptr<rmm::mr::pool_memory_resource>     resource ;
    rmm::mr::pool_memory_resource                      resource ;
    std::shared_ptr<alloc_map>                         size_map ;
    cudaStream_t                                       stream ;

    RMM_Wrap_Handle_struct ( )
        : resource (cuda_mr_default, 0)
        , size_map (nullptr)
        , stream (nullptr)
    {
        // any additional setup code here
    }
}
RMM_Wrap_Handle ;

// rmm_wrap_context: global array of RMM_Wrap_Handle objects, one per GPU
static RMM_Wrap_Handle *rmm_wrap_context [GxB_NARENAS_GPU] ;

static bool rmm_wrap_initialized = false ;

// devices that GraphBLAS can use:
// static uint32_t devices [GxB_NARENAS_GPU] ;
static int ngpus = 0 ;

//------------------------------------------------------------------------------
// make a resource pool
//------------------------------------------------------------------------------

#if 0
inline auto make_cuda()
{
    return std::make_shared<rmm::mr::cuda_memory_resource>() ;
}

inline auto make_managed()
{
    return std::make_shared<rmm::mr::managed_memory_resource>() ;
}

// size_map is an unordered alloc_map that maps allocation address to the size
// of each allocation

inline auto make_and_set_device_pool
(
    std::size_t initial_size,
    std::size_t maximum_size
)
{
    auto resource = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>
                    ( make_cuda(), initial_size, maximum_size ) ;
    rmm::mr::set_current_device_resource( resource.get()) ;
    return resource ;
}

inline auto make_and_set_managed_pool
(
    std::size_t initial_size,
    std::size_t maximum_size
)
{

// Fixme: this method is not used at all; it's broken

// RMM 24.x:
//  rmm::mr::pool_memory_resource resource {
//      rmm::mr::managed_memory_resource{},
//      initial_size } ;
//  rmm::mr::set_current_device_resource ( resource.get() ) ;

#if 1
//  rmm::mr::cuda_memory_resource cuda_mr ;
    rmm::mr::managed_memory_resource cuda_mr ;
    // Construct a resource that uses a coalescing best-fit pool allocator
    // With the pool initially half of available device memory
//  auto initial_size = rmm::percent_of_free_device_memory(50) ;
    rmm::mr::pool_memory_resource pool_mr{cuda_mr, initial_size};
    auto previous = rmm::mr::set_current_device_resource (pool_mr) ;
//  rmm::mr::set_current_device_resource (cuda::mr::any_resource<cuda::mr::device_accessible>) ;
//  auto mr = rmm::mr::get_current_device_resource_ref() ;
    return pool_mr ;
#else

// RMM 26.06.00:
    rmm::mr::managed_memory_resource cuda_mr ;
    rmm::mr::pool_memory_resource resource {cuda_mr, initial_size} ;
    rmm::mr::set_current_device_resource ( resource ) ;
    return resource ;
#endif
}
#endif

//------------------------------------------------------------------------------
// rmm_wrap_is_initialized: determine if this wrapper has been initialized
//------------------------------------------------------------------------------

bool rmm_wrap_is_initialized (void)
{
    return (rmm_wrap_initialized) ;
}

//------------------------------------------------------------------------------
// rmm_wrap_finalize: destroy the global rmm_wrap_context
//------------------------------------------------------------------------------

// Destroy the rmm_wrap_context.  This method allows destroys the contents of
// the rmm_wrap_context:  the memory resource (host or device) and the
// alloc_map.

void rmm_wrap_finalize (void)
{
    if (!rmm_wrap_initialized) return ;
    try
    {
        for (int k = 0 ; k < ngpus ; k++)
        {
            // fixme: destroy the kth stream on device k
            // need cudaSetDevice (k) ;
            delete rmm_wrap_context [k] ;
        }
        // fixme: need one stream per device (not in the stream pool)
        cudaStreamDestroy (rmm_wrap_global_stream) ;
    }
    catch (...)
    {
        // something failed; just return
        return ;
    }
    rmm_wrap_initialized = false ;
}

//------------------------------------------------------------------------------
// rmm_wrap_initialize: initialize rmm_wrap_context[device_id]
//------------------------------------------------------------------------------

int rmm_wrap_initialize     // returns -1 on error, 0 on success
(
    int device_id           // GPU device id to initialize
//  ,
//  // fixme: pool sizes are current not used:
//  size_t init_pool_memsize,  // TODO: describe. Should we default this?
//  size_t max_pool_memsize    // TODO: describe. Should we default this?
)
{

    if (rmm_wrap_initialized) return (-1) ;

    try
    {

        //----------------------------------------------------------------------
        // check inputs
        //----------------------------------------------------------------------

        if (device_id < 0 || device_id > GxB_NARENAS_GPU ||
            rmm_wrap_context [device_id] != NULL)
        {
            return (-1) ;
        }

        cudaSetDevice (device_id) ;

        // create the RMM wrap handle and save it as a global pointer.
        // fixme: does this create a ->resouce on the current device?
        // fixme: new is not a "placement new"; this uses malloc/free;
        // should used arena 0 allocators.
        rmm_wrap_context [device_id] = new RMM_Wrap_Handle ( ) ;

        //----------------------------------------------------------------------
        // Construct a resource that uses a coalescing best-fit pool allocator
        //----------------------------------------------------------------------

#if 0

            // Fixme: allow control of pool sizes

            // std::cout << "Seting managed pool" << std::endl ;
//          rmm_wrap_context[device_id]->resource = make_and_set_managed_pool
//              ( init_pool_memsize, max_pool_memsize) ;

#endif

        //----------------------------------------------------------------------
        // create size_map to lookup size of each allocation
        //----------------------------------------------------------------------

        // fixme: this uses the standard malloc/free, not arena 0
        rmm_wrap_context [device_id]->size_map =
            std::make_shared<alloc_map> ( ) ;
        if (rmm_wrap_context [device_id]->size_map.get ( ) == NULL)
        {
            // failed to create the alloc_map
            return (-1) ;
        }

        return (0) ;    // success

    }
    catch (...)
    {
        return (-1) ;
    }
}

//------------------------------------------------------------------------------
// rmm_wrap_initialize_all: initialize global rmm_wrap_context for all devices
//------------------------------------------------------------------------------

int rmm_wrap_initialize_all_same
(
    void
//  size_t init_pool_memsize,      // TODO: describe. Should we default this?
//  size_t max_pool_memsize        // TODO: describe. Should we default this?
)
{

    if (rmm_wrap_initialized) return (-1) ;

    //--------------------------------------------------------------------------
    // clear the list of GPUs
    //--------------------------------------------------------------------------

    for (int k = 0 ; k < GxB_NARENAS_GPU ; k++)
    {
        rmm_wrap_context [k] = NULL ;
    }
    ngpus = 0 ;

    //--------------------------------------------------------------------------
    // initialize all GPUs
    //--------------------------------------------------------------------------

    try
    {

        //----------------------------------------------------------------------
        // get the list of GPUs
        //----------------------------------------------------------------------

#if 0
        Fixme: this mapping is not yet supported

        const char* cuda_visible_devices =
            std::getenv ("CUDA_VISIBLE_DEVICES" );
        if (cuda_visible_devices != nullptr)
        {

            //------------------------------------------------------------------
            // get the list of GPUs from the environment variable
            //------------------------------------------------------------------

            std::cout << "CUDA_VISIBLE_DEVICES = " << cuda_visible_devices
                << std::endl ;
            std::stringstream check1 ;
            check1 << cuda_visible_devices ;
            std::string intermediate ;
            for (int k = 0 ; getline (check1, intermediate, ',') ; k++)
            {
                // remove any spaces
                intermediate.erase(std::remove_if(intermediate.begin(),
                    intermediate.end(), ::isspace), intermediate.end()) ;

                // GPUs represented by UUIDs from "nvidia-smi -L" or MIG
                if (std::strncmp ("GPU-"    , intermediate.c_str ( ), 4) == 0 ||
                    std::strncmp ("MIG-GPU-", intermediate.c_str ( ), 8) == 0)
                {
                    // device IDs must work with cudaSetDevice() and
                    // as indices for rmm_wrap_context[]
                    devices [ngpus++] = k ;
                }
                else
                {
                    uint32_t device_id = 
                        static_cast<uint32_t> (stoi (intermediate)) ;
                    if (device_id < GxB_NARENAS_GPU)
                    {
                        std::cout << "Found device_id " << device_id
                            << std::endl ;
                        devices [ngpus++] = device_id ;
                    }
                }

                if (ngpus == GxB_NARENAS_GPU)
                {
                    // maximum number of GPUs reached
                    break ;
                }
            }

        }
        else
#endif
        {

            //------------------------------------------------------------------
            // use all GPUs
            //------------------------------------------------------------------

            cudaGetDeviceCount (&ngpus) ;
            ngpus = std::min (ngpus, GxB_NARENAS_GPU) ;
//          for (int k = 0 ; k < ngpus ; k++)
//          {
//              devices [k] = k ;
//          }
        }

        // Fixme: use one stream per device
        cudaStreamCreate (&rmm_wrap_global_stream) ;

        std::cout << "# GPUs: " << ngpus << std::endl ;

        // Allocate rmm_wrap_contexts

        for (int k = 0 ; k < ngpus ; k++)
        {
            uint32_t device_id = k ; // devices [k] ;
            int result = rmm_wrap_initialize (device_id
//              , init_pool_memsize, max_pool_memsize
                ) ;
            if (result < 0)
            {
                return (result) ;
            }
        }

        rmm_wrap_initialized = true ;
        return (0) ;
    }
    catch (...)
    {
        return (-1) ;
    }
}

//------------------------------------------------------------------------------
// malloc/free methods for each GPU device (up to 64 devices)
//------------------------------------------------------------------------------

// create a pair of malloc/free methods for a given device id
#define GB_RMM_MALLOC_FREE(id)                  \
    void *GB_rmm_malloc_ ## id (size_t size)    \
    {                                           \
        return (rmm_allocate (id, size)) ;      \
    }                                           \
    void GB_rmm_free_ ## id (void *p)           \
    {                                           \
        rmm_deallocate (id, p) ;                \
    }

GB_RMM_MALLOC_FREE (0) ;
GB_RMM_MALLOC_FREE (1) ;
GB_RMM_MALLOC_FREE (2) ;
GB_RMM_MALLOC_FREE (3) ;
GB_RMM_MALLOC_FREE (4) ;
GB_RMM_MALLOC_FREE (6) ;
GB_RMM_MALLOC_FREE (7) ;
GB_RMM_MALLOC_FREE (8) ;
GB_RMM_MALLOC_FREE (9) ;

GB_RMM_MALLOC_FREE (10) ;
GB_RMM_MALLOC_FREE (11) ;
GB_RMM_MALLOC_FREE (12) ;
GB_RMM_MALLOC_FREE (13) ;
GB_RMM_MALLOC_FREE (14) ;
GB_RMM_MALLOC_FREE (16) ;
GB_RMM_MALLOC_FREE (17) ;
GB_RMM_MALLOC_FREE (18) ;
GB_RMM_MALLOC_FREE (19) ;

GB_RMM_MALLOC_FREE (20) ;
GB_RMM_MALLOC_FREE (21) ;
GB_RMM_MALLOC_FREE (22) ;
GB_RMM_MALLOC_FREE (23) ;
GB_RMM_MALLOC_FREE (24) ;
GB_RMM_MALLOC_FREE (26) ;
GB_RMM_MALLOC_FREE (27) ;
GB_RMM_MALLOC_FREE (28) ;
GB_RMM_MALLOC_FREE (29) ;

GB_RMM_MALLOC_FREE (30) ;
GB_RMM_MALLOC_FREE (31) ;
GB_RMM_MALLOC_FREE (32) ;
GB_RMM_MALLOC_FREE (33) ;
GB_RMM_MALLOC_FREE (34) ;
GB_RMM_MALLOC_FREE (36) ;
GB_RMM_MALLOC_FREE (37) ;
GB_RMM_MALLOC_FREE (38) ;
GB_RMM_MALLOC_FREE (39) ;

GB_RMM_MALLOC_FREE (40) ;
GB_RMM_MALLOC_FREE (41) ;
GB_RMM_MALLOC_FREE (42) ;
GB_RMM_MALLOC_FREE (43) ;
GB_RMM_MALLOC_FREE (44) ;
GB_RMM_MALLOC_FREE (46) ;
GB_RMM_MALLOC_FREE (47) ;
GB_RMM_MALLOC_FREE (48) ;
GB_RMM_MALLOC_FREE (49) ;

GB_RMM_MALLOC_FREE (50) ;
GB_RMM_MALLOC_FREE (51) ;
GB_RMM_MALLOC_FREE (52) ;
GB_RMM_MALLOC_FREE (53) ;
GB_RMM_MALLOC_FREE (54) ;
GB_RMM_MALLOC_FREE (56) ;
GB_RMM_MALLOC_FREE (57) ;
GB_RMM_MALLOC_FREE (58) ;
GB_RMM_MALLOC_FREE (59) ;

GB_RMM_MALLOC_FREE (60) ;
GB_RMM_MALLOC_FREE (61) ;
GB_RMM_MALLOC_FREE (62) ;
GB_RMM_MALLOC_FREE (63) ;

//------------------------------------------------------------------------------
// rmm_allocate: allocate a block of memory on a given device
//------------------------------------------------------------------------------

void *rmm_allocate (int device_id, size_t size)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    void *p = NULL ;
    // printf ("rmm_allocate (%d, %zu)\n", device_id, size) ;

    GB_OPENMP_LOCK_SET (2) ;    // Fixme: use a different lock for each arena

    if (rmm_wrap_initialized && device_id >= 0 && device_id < GxB_NARENAS_GPU
        && rmm_wrap_context [device_id] != NULL)
    {
        try
        {

            //------------------------------------------------------------------
            // get the hash map for this device and revise the size
            //------------------------------------------------------------------

            alloc_map *am = rmm_wrap_context [device_id]->size_map.get ( ) ;

            // ensure size is nonzero
            if (size == 0) size = 256 ;

            // round-up the allocation to a multiple of 256
            size_t aligned = size % 256 ;
            if (aligned > 0)
            {
                size += (256 - aligned) ;
            }

            //------------------------------------------------------------------
            // allocate the space on the device using Rapids
            //------------------------------------------------------------------

            if (am != NULL)
            {
                cudaSetDevice (device_id) ;
                p = (rmm_wrap_context [device_id]->resource).allocate
                    (rmm_wrap_global_stream, size, 256) ;
            }

            //------------------------------------------------------------------
            // insert p into the hashmap, with its revised size
            //------------------------------------------------------------------

            if (p != NULL)
            {
                am->emplace ((size_t) p, (size_t) size) ;
            }
        }
        catch (...)
        {
            /* an error has occured; leave p as NULL */ ;
        }
    }

    GB_OPENMP_LOCK_UNSET (2) ;

    //--------------------------------------------------------------------------
    // return pointer to allocated block of memory (or NULL on error)
    //--------------------------------------------------------------------------

    return (p) ;
}

//------------------------------------------------------------------------------
// rmm_deallocate: deallocate a block previously allocated by rmm_allocate
//------------------------------------------------------------------------------

void rmm_deallocate (int device_id, void *p)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_OPENMP_LOCK_SET (2) ;    // Fixme: use a different lock for each arena

    if (rmm_wrap_initialized && device_id >= 0 && device_id < GxB_NARENAS_GPU
        && rmm_wrap_context [device_id] != NULL && p != NULL)
    {

        try
        {

            //------------------------------------------------------------------
            // get the size of the block of memory
            //------------------------------------------------------------------

            alloc_map *am = rmm_wrap_context [device_id]->size_map.get ( ) ;
            size_t size = 0 ;
            if (am != NULL)
            {
               // get the size of the block of memory from the hash_map
               auto iter = am->find ((size_t) p)  ;
               if (iter != am->end ( )) size = iter->second ;
            }

            //------------------------------------------------------------------
            // remove p from the hashmap and deallocate it using Rapids
            //------------------------------------------------------------------

            if (size > 0)
            {
                am->erase ((size_t) p) ;
                cudaSetDevice (device_id) ;
                (rmm_wrap_context [device_id]->resource).deallocate
                    (rmm_wrap_global_stream, p, size, 256) ;
            }
        }
        catch (...)
        {
            /* something failed; just catch the error and return */
        }
    }

    GB_OPENMP_LOCK_UNSET (2) ;
}

