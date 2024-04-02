//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_type_branch: decide if GPU can be used on a type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The CUDA kernels require that the type sizes are 1, 2, or a multiple of 4
// bytes.  In addition, the shfl_down primitives require the type to be 32
// bytes or less.  If user-defined type has a different size, it cannot be done
// on the GPU.

// FIXME: get the CUDA kernels to work on large types

// All built-in types pass this rule.

#include "GB_cuda.hpp"

bool GB_cuda_type_branch            // return true if the type is OK on GPU
(
    const GrB_Type type             // type to query
)
{

    if (type == NULL)
    {
        // if the type is NULL, it's ignored anyway, so it's fine
        return (true) ;
    }

    size_t size = type->size ;

    if (size == sizeof (uint8_t) || size == sizeof (uint16_t))
    {
        // size is 1 or 2 bytes
        return (true) ;
    }

    if (size % sizeof (uint32_t) == 0 && size <= 32)
    {
        // size is 4, 16, 20, 24, 28, or 32
        return (true) ;
    }

    // the type is not supported on the GPU
    return (false) ;
}

