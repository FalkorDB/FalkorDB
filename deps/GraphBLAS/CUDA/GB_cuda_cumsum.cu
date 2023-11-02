//------------------------------------------------------------------------------
// GB_cuda_cumsum: cumlative sum of an array using GPU acceleration
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compute the cumulative sum of an array count[0:n], of size n+1
// in pseudo-MATLAB notation:

//      k = sum (count [0:n-1] != 0) ;

//      count = cumsum ([0 count[0:n-1]]) ;

// That is, count [j] on input is overwritten with the value of
// sum (count [0..j-1]).  count [n] is implicitly zero on input.
// On output, count [n] is the total sum.

#include "GB_cuda.h"
// #include <local_cub/device/device_scan.cuh>
#include <cub/device/device_scan.cuh>

GrB_Info GB_cuda_cumsum             // compute the cumulative sum of an array
(
    int64_t *restrict count,    // size n+1, input/output
    const int64_t n
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (count != NULL) ;
    ASSERT (n >= 0) ;

    //--------------------------------------------------------------------------
    // count = cumsum ([0 count[0:n-1]]) ;
    //--------------------------------------------------------------------------

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, count,
        count, (int) n) ;
    size_t size ;
    d_temp_storage  = GB_malloc_memory( temp_storage_bytes, 1, &size);
    if ( d_temp_storage == NULL){
       return GrB_OUT_OF_MEMORY;
    } 

    // Run
    CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, count, count, n));

    // Check for correctness (and display results, if specified)
    #if 0
    #ifdef GB_DEBUG
    int compare = CompareDeviceResults(h_reference, count, num_items, true, g_verbose);
    ASSERT( compare == 0);
    #endif
    #endif

    // Cleanup
    GB_free_memory (&d_temp_storage, size) ;

    return GrB_SUCCESS;
}

