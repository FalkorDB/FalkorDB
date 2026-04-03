//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_get_device_properties: get the properties of a GPU
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These functions are not to be called by any *.cu method.  Instead,
// they are to be solely used by C functions in GraphBLAS/Source.

#include "GB_cuda.hpp"

#define CU_OK(cudaMethod)                               \
{                                                       \
    if ((cudaMethod) != cudaSuccess)                    \
    {                                                   \
        printf ("cuda failed, file: %s, line: %d\n", __FILE__, __LINE__) ; \
        return (false) ;   \
    }   \
}

//------------------------------------------------------------------------------
// GB_cuda_get_device: get the current GPU
//------------------------------------------------------------------------------

bool GB_cuda_get_device (int *device)
{
    if (device == NULL)
    {
        // invalid inputs
        return (false) ;
    }
    CU_OK (cudaGetDevice (device)) ;
    return (true) ;
}

//------------------------------------------------------------------------------
// GB_cuda_set_device: set the current GPU
//------------------------------------------------------------------------------

bool GB_cuda_set_device (int device)
{
    if (device < 0)
    {
        // invalid inputs
        return (false) ;
    }
    CU_OK (cudaSetDevice (device)) ;
    return (true) ;
}

//------------------------------------------------------------------------------
// GB_cuda_get_device_properties: determine all properties of a single GPU
//------------------------------------------------------------------------------

bool GB_cuda_get_device_properties  // true if OK, false if failure
(
    int device,
    GB_cuda_device *prop
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (prop == NULL || device < 0)
    {
        // invalid inputs
        return (false) ;
    }

    // clear the GPU settings
    memset (prop, 0, sizeof (GB_cuda_device)) ;

    int original_device ;
    CU_OK (cudaGetDevice (&original_device)) ;

    //--------------------------------------------------------------------------
    // get the properties
    //--------------------------------------------------------------------------

    int num_sms, compute_capability_major, compute_capability_minor ;
    size_t memfree, memtotal ;

    CU_OK (cudaDeviceGetAttribute (&num_sms,
                cudaDevAttrMultiProcessorCount, device)) ;
    CU_OK (cudaDeviceGetAttribute (&compute_capability_major,
                cudaDevAttrComputeCapabilityMajor, device)) ;
    CU_OK (cudaDeviceGetAttribute (&compute_capability_minor,
                cudaDevAttrComputeCapabilityMinor, device)) ;

    CU_OK (cudaSetDevice (device)) ;
    CU_OK (cudaMemGetInfo (&memfree, &memtotal)) ;
    CU_OK (cudaSetDevice (original_device)) ;

    prop->total_global_memory = memtotal ;
    prop->number_of_sms = num_sms ;
    prop->compute_capability_major = compute_capability_major ;
    prop->compute_capability_minor = compute_capability_minor ;

    // FIXME: remove this printf
    printf ("\nDevice: %d: memory: %ld SMs: %d compute: %d.%d\n",
        device, prop->total_global_memory, prop->number_of_sms,
        prop->compute_capability_major, prop->compute_capability_minor) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (true) ;
}

