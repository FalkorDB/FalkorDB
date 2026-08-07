//------------------------------------------------------------------------------
// rmm_wrap.cpp: C-callable wrapper for an RMM memory resource
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// rmm_wrap.cpp contains a single global object, the RMM_Wrap_Handle that holds
// an RMM (Rapids Memory Manager) memory resource and a hash map (C++
// std:unordered_map).  This allows rmm_wrap to provide 7 functions to a C
// application:

// Create/destroy an RMM resource:
//      rmm_wrap_initialize: create the RMM resource
//      rmm_wrap_is_initialized: query if the RMM resource has been created
//      rmm_wrap_finalize: destroy the RMM resource

// C-style malloc/free methods:
//      rmm_wrap_malloc:  malloc a block of memory using RMM
//      rmm_wrap_free:    free a block of memory allocated by this RMM wrapper

// PMR-based allocate/deallocate methods (C-callable):
//      rmm_wrap_allocate (std::size_t *size)
//      rmm_wrap_deallocate (void *p, std::size_t size)

#include "rmm_wrap.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>

//------------------------------------------------------------------------------
// RMM_Wrap_Handle: a global object containing the RMM context
//------------------------------------------------------------------------------

// NOTE: this is not thread-safe

// rmm_wrap_context is a pointer to an array of global RMM_Wrap_Handle objects
// (one per GPU) that all methods in this file can access.  The array of
// objects cannot be accessed outside this file.

cudaStream_t rmm_wrap_global_stream = nullptr ;

rmm::mr::managed_memory_resource cuda_mr_default ;
rmm::mr::pool_memory_resource    cuda_pool_default (cuda_mr_default, 0) ;

typedef struct RMM_Wrap_Handle_struct
{
    uint32_t device_id;
    RMM_MODE mode;
//  std::shared_ptr<rmm::mr::pool_memory_resource>     resource;
    rmm::mr::pool_memory_resource                      resource ;
//  std::shared_ptr<std::pmr::memory_resource>         host_resource;
    std::shared_ptr<alloc_map>                         size_map ;

#if 1
//  RMM_Wrap_Handle_struct() : resource() { } ; // FAILS

    RMM_Wrap_Handle_struct ( )
        : device_id (0)
        , mode (rmm_wrap_managed)
//      , resource (nullptr)        // FAILS
        , resource (cuda_mr_default, 0)     // OK
        , size_map (nullptr)        // FAILS
//      , size_map ()               // FAILS
    {
        // any additional setup code here
    }
#endif
}
RMM_Wrap_Handle ;



// rmm_wrap_context: global pointer to the single array of RMM_Wrap_Handle
// objects, one per GPU
static RMM_Wrap_Handle **rmm_wrap_context = NULL ;
static std::vector<uint32_t> devices;


//------------------------------------------------------------------------------
// make a resource pool
//------------------------------------------------------------------------------

#if 0
inline auto make_cuda()
{
    return std::make_shared<rmm::mr::cuda_memory_resource>() ;
}
#endif

#if 0
inline auto make_managed()
{
    return std::make_shared<rmm::mr::managed_memory_resource>() ;
}
#endif

// size_map is an unordered alloc_map that maps allocation address to the size
// of each allocation

#if 0
inline auto make_and_set_device_pool
(
    std::size_t initial_size,
    std::size_t maximum_size
)
{
    auto resource = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>
                    ( make_cuda(), initial_size, maximum_size ) ;
    rmm::mr::set_current_device_resource( resource.get()) ;
    return resource;
}
#endif

inline auto make_and_set_managed_pool
(
    std::size_t initial_size,
    std::size_t maximum_size
)
{

// RMM 24.x:
//  rmm::mr::pool_memory_resource resource {
//      rmm::mr::managed_memory_resource{},
//      initial_size } ;
//  rmm::mr::set_current_device_resource ( resource.get() ) ;

#if 1
//  rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::managed_memory_resource cuda_mr ;
    // Construct a resource that uses a coalescing best-fit pool allocator
    // With the pool initially half of available device memory
//  auto initial_size = rmm::percent_of_free_device_memory(50);
    rmm::mr::pool_memory_resource pool_mr{cuda_mr, initial_size};
    auto previous = rmm::mr::set_current_device_resource (pool_mr);
//  rmm::mr::set_current_device_resource (cuda::mr::any_resource<cuda::mr::device_accessible>) ;
//  auto mr = rmm::mr::get_current_device_resource_ref();
    return pool_mr ;
#else

// RMM 26.06.00:
    rmm::mr::managed_memory_resource cuda_mr ;
    rmm::mr::pool_memory_resource resource {cuda_mr, initial_size} ;
    rmm::mr::set_current_device_resource ( resource ) ;
    return resource;
#endif
}

//------------------------------------------------------------------------------
// rmm_wrap_is_initialized: determine if rmm_wrap_context exists
//------------------------------------------------------------------------------

bool rmm_wrap_is_initialized (void)
{
    return (rmm_wrap_context != NULL) ;
}

//------------------------------------------------------------------------------
// rmm_wrap_finalize: destroy the global rmm_wrap_context
//------------------------------------------------------------------------------

// Destroy the rmm_wrap_context.  This method allows destroys the contents of
// the rmm_wrap_context:  the memory resource (host or device) and the
// alloc_map.

// fixme for CUDA: GraphBLAS currently does not call this method ...

void rmm_wrap_finalize (void)
{
    try
    {
        if (rmm_wrap_context != NULL)
        {
            for (int device_id = 0; device_id < devices.size(); ++device_id)
            {
                delete rmm_wrap_context[device_id];
            }
            delete rmm_wrap_context ;
            rmm_wrap_context = NULL ;
        }
        cudaStreamDestroy (rmm_wrap_global_stream) ;
    }
    catch (...)
    {
        // something failed; just return
        return ;
    }
}

//------------------------------------------------------------------------------
// get_current_device: helper to get id for currently selected device
//------------------------------------------------------------------------------

int get_current_device(void)
{
    // fixme for CUDA: return an error code if this method fails
    int device_id;
    cudaGetDevice(&device_id);
    return device_id;
}

//------------------------------------------------------------------------------
// rmm_wrap_initialize: initialize rmm_wrap_context[device_id]
//------------------------------------------------------------------------------

int rmm_wrap_initialize     // returns -1 on error, 0 on success
(
    uint32_t device_id,     // GPU device id, for cudaSetDevice
    RMM_MODE mode,          // TODO: describe. Should we default this?
    size_t init_pool_memsize,  // TODO: describe. Should we default this?
    size_t max_pool_memsize    // TODO: describe. Should we default this?
)
{
    
    try
    {

        //----------------------------------------------------------------------
        // check inputs
        //----------------------------------------------------------------------

        if (rmm_wrap_context[device_id] != NULL)
        {
            return (-1) ;
        }

        RMM_WRAP_CHECK_CUDA (cudaSetDevice (device_id)) ;

        // create the RMM wrap handle and save it as a global pointer.
        rmm_wrap_context [device_id] = new RMM_Wrap_Handle() ;
        // fixme for CUDA: check for error?

        //  std::cout<< " init called with mode "<<mode<<" init_size "
        // <<init_pool_memsize<<" max_size "<<max_pool_memsize<<"\n";

        //----------------------------------------------------------------------
        // Construct a resource that uses a coalescing best-fit pool allocator
        //----------------------------------------------------------------------

#if 0
        if (mode == rmm_wrap_host )
        {
            // rmm_wrap_context->host_resource =
            //  std::pmr::synchronized_pool_resource() ;
            //  // (init_pool_memsize, max_pool_memsize) ;
            // rmm_wrap_context->host_resource =  make_and_set_host_pool() ;
            //  // (init_pool_memsize, max_pool_memsize) ;
        }
        else if (mode == rmm_wrap_host_pinned )
        {
            // rmm_wrap_context->host_resource =
            //  std::pmr::synchronized_pool_resource() ;
            //  // (init_pool_memsize, max_pool_memsize) ;
        }
        else if (mode == rmm_wrap_device )
        {
//          rmm_wrap_context[device_id]->resource =
//              make_and_set_device_pool( init_pool_memsize, max_pool_memsize) ;
        }
        else 
#endif
        if ( mode == rmm_wrap_managed )
        {
            // std::cout << "Seting managed pool" << std::endl;
//          rmm_wrap_context[device_id]->resource = make_and_set_managed_pool( init_pool_memsize, max_pool_memsize);
        }
        else
        {
            // invalid mode
            return (-1) ;
        }

        // std::cout << "Setting mode for rmm_wrap context" << std::endl;
        // Mark down the mode for reference later
        rmm_wrap_context[device_id]->mode = mode;

        //----------------------------------------------------------------------
        // create size map to lookup size of each allocation
        //----------------------------------------------------------------------

        // std::cout << "Setting size_map for rmm_wrap context" << std::endl;
        rmm_wrap_context[device_id]->size_map = std::make_shared<alloc_map> () ;
        if (rmm_wrap_context[device_id]->size_map.get() == NULL)
        {
            // std::cout << "Failed to create size_map" << std::endl;
            // failed to create the alloc_map
            return (-1) ;
        }
        return (0) ;
    }
    catch (...)
    {
        // something failed; return an error code
        return (-1) ;
    }
}

//------------------------------------------------------------------------------
// rmm_wrap_initialize_all: initialize global rmm_wrap_context for all devices
//------------------------------------------------------------------------------

int rmm_wrap_initialize_all_same
(
    RMM_MODE mode,              // TODO: describe. Should we default this?
    size_t init_pool_memsize,      // TODO: describe. Should we default this?
    size_t max_pool_memsize        // TODO: describe. Should we default this?
)
{
    try
    {

        if (rmm_wrap_context != NULL)
        {
            return (-1);
        }

        devices.clear();

        cudaStreamCreate (&rmm_wrap_global_stream) ;

        const char* cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
        if (cuda_visible_devices != nullptr)
        {
            std::cout << "CUDA_VISIBLE_DEVICES = " << cuda_visible_devices
                << std::endl;
        }

        /**
         * Start with "CUDA_VISIBLE_DEVICES" var if it's defined.
         */
        if(cuda_visible_devices != nullptr) {
            std::cout << "getting cuda visible devices" << std::endl;
            std::stringstream check1;
            check1 << cuda_visible_devices;
            std::string intermediate;
            for (int i = 0; getline(check1, intermediate, ','); ++i)
            {
                intermediate.erase(std::remove_if(intermediate.begin(),
                    intermediate.end(), ::isspace), intermediate.end());

                // GPUs represented by UUIDs from "nvidia-smi -L" or MIG
                if (std::strncmp("GPU-", intermediate.c_str(), 4) == 0 ||
                    std::strncmp("MIG-GPU-", intermediate.c_str(), 8) == 0)
                {
                    // device IDs must work with cudaSetDevice() and
                    // as indices for rmm_wrap_context[]
                    devices.push_back(i);
                    continue;
                }

                uint32_t device_id = static_cast<uint32_t>(stoi(intermediate));
                std::cout << "Found device_id " << device_id << std::endl;
                devices.push_back(device_id);
            }
        /**
         * If CUDA_VISIBLE_DEVICES not explicitly specified,
         * default to device 0.
         */
        } else {
            int ngpus = 0 ;
            cudaGetDeviceCount (&ngpus) ;
            std::cout << "Using all devices: " << ngpus << std::endl;
            for (int i = 0 ; i < ngpus ; i++)
            {
                devices.push_back(i);
            }
        }

        // Allocate rmm_wrap_contexts
//      printf ("\ndevices.size %ld\n", devices.size()) ;
        std::cout << "devices.size is " << devices.size() << std::endl ;
        rmm_wrap_context = (RMM_Wrap_Handle**)malloc(devices.size() * sizeof(RMM_Wrap_Handle*));
        for(int i = 0; i < devices.size(); ++i) {
            rmm_wrap_context[i] = NULL;
            uint32_t device_id = devices[i];
            int ret = rmm_wrap_initialize(device_id, mode, init_pool_memsize, max_pool_memsize ) ;
            if(ret < 0) {
                return ret;
            }
        }

        return 0;
    }
    catch (...)
    {
        // something failed; return an error state
        printf ("rmm_wrap_initialize_all_same failed!\n") ;
        return (-1) ;
    }
}

//------------------------------------------------------------------------------
// rmm_wrap_malloc: malloc-equivalent method using RMM
//------------------------------------------------------------------------------

// rmm_wrap_malloc is identical to the C11 malloc function, except that
// it uses RMM underneath to allocate the space.

void *rmm_wrap_malloc (std::size_t size)
{
    return (rmm_wrap_allocate (&size)) ;
}

//------------------------------------------------------------------------------
// rmm_wrap_free: free a block of memory, size not needed
//------------------------------------------------------------------------------

// rmm_wrap_free is identical to the C11 free function, except that
// it uses RMM underneath to allocate the space.

void rmm_wrap_free (void *p)
{
    rmm_wrap_deallocate (p, 0) ;
}

//------------------------------------------------------------------------------
// rmm_wrap_allocate: allocate a block of memory using RMM
//------------------------------------------------------------------------------

void *rmm_wrap_allocate( std::size_t *size)
{
    try
    {
        void *p = NULL ;

        if (rmm_wrap_context == NULL)
        {
            return (NULL) ;
        }

        // fixme for CUDA: check for failure of get_current_device
        uint32_t device_id = get_current_device();

        alloc_map *am = rmm_wrap_context[device_id]->size_map.get() ;
        if (am == NULL)
        {
            // PANIC!
            // std::cout<< "Uh oh, can't allocate before initializing RMM"
            // << std::endl;
            return (NULL) ;
        }

        // ensure size is nonzero
        if (*size == 0) *size = 256 ;
        // round-up the allocation to a multiple of 256
        std::size_t aligned = (*size) % 256 ;
        if (aligned > 0)
        {
            *size += (256 - aligned) ;
        }

        #if 0
        rmm::mr::pool_memory_resource memoryresource =
            rmm::mr::get_current_device_resource_ref() ;
        p = memoryresource->allocate( *size ) ;
        #endif
    
#if 1
// segfault:
        p = (rmm_wrap_context [device_id]->resource).allocate( rmm_wrap_global_stream, *size , 256) ;
#else
        p = cuda_pool_default.allocate( rmm_wrap_global_stream, *size , 256) ;
#endif

        if (p == NULL)
        {
            // out of memory
            *size = 0 ;
            return (NULL) ;
        }

        // insert p into the hashmap
        am->emplace ((std::size_t)p, (std::size_t)(*size)) ;

        // return the allocated block
        return (p) ;

    }
    catch (...)
    {
        // something failed; just return NULL
        return (NULL) ;
    }
}

//------------------------------------------------------------------------------
// rmm_wrap_deallocate: deallocate a block previously allocated by RMM
//------------------------------------------------------------------------------

void rmm_wrap_deallocate( void *p, std::size_t size)
{
    try
    {
        if (rmm_wrap_context == NULL)
        {
            return ;
        }

        // Note: there are 3 PANIC cases below.  The API of rmm_wrap_deallocate
        // does not allow an error condition to be returned.  These PANICs
        // could be logged, or they could terminate the program if debug mode
        // enabled, etc.  In production, all we can do is ignore the PANIC.

        if (p == NULL)
        {
            // nothing to do; ignore a double-free
            if (size > 0)
            {
                // PANIC!  Why does a NULL pointer have a nonzero size??
            }
            return ;
        }

        uint32_t device_id = get_current_device();

        // check the size given.  If the input size is zero, then the size is
        // unknown (say rmm_wrap_free(p)).  In that case, just trust the
        // hashmap.  Otherwise, double-check to make sure the size is correct.
        alloc_map *am = rmm_wrap_context[device_id]->size_map.get() ;
        size_t actual_size = 0 ;
        if (am == NULL)
        {
            // PANIC!
            // std::cout<< "Uh oh, can't deallocate before initializing RMM"
            // << std::endl;
            return ;
        }
        else
        {
           //actual_size = am->at( (std::size_t)(p) )  ;
           auto iter = am->find( (std::size_t)(p) )  ;
           if (iter != am->end() ) actual_size = iter->second;
           // else std::cout <<
           // " rmm_wrap:: tried to free unallocated pointer !" << p ;
        }

        if (actual_size == 0)
        {
            // PANIC!  oops, p is not in the hashmap.  Ignore it.  TODO: could
            // add a printf here, write to a log file, etc.  if debug mode,
            // abort, etc.
            return ;
        }

        if (size > 0 && size != actual_size)
        {
            // PANIC!  oops, invalid old size.  Ignore the input size, and free
            // p anyway.  TODO: could add a printf here, write to a log file,
            // etc.  if debug mode, abort, etc.
        }

        // remove p from the hashmap
        am->erase ( (std::size_t)(p) ) ;

        // deallocate the block of memory
//      rmm::mr::pool_memory_resource memoryresource =
//          rmm::mr::get_current_device_resource_ref() ;
//      memoryresource->deallocate( p, actual_size ) ;
        cuda_pool_default.deallocate( rmm_wrap_global_stream, p, actual_size , 256 ) ;


    }
    catch (...)
    {
        // something failed; just catch the error and return
        return ;
    }
}

