//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_type_branch: decide if GPU can be used on a type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The CUDA kernels require that the type sizes are 1, 2, or a multiple of 4
// bytes.  All built-in types pass this rule.

// This method does not check type->hash.  If it is UINT64_MAX, then it cannot
// be JIT'd, but this is accounted for by checking the relevant higher-level
// op, monoid, or semiring.  Those will have a hash of UINT64_MAX if any of
// their consituient parts (the types and ops in a semiring for example) have a
// hash of UINT64_MAX.

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

    if (type == GxB_FC32 || type == GxB_FC64)
    {
        // FIXME: complex types not yet supported in CUDA
        return (false) ;
    }

    size_t size = type->size ;

    if (size == sizeof (uint8_t) || size == sizeof (uint16_t))
    {
        // size is 1 or 2 bytes
        return (true) ;
    }

    if (size % sizeof (uint32_t) == 0)
    {
        // size is 4, 16, 20, 24, 28, or 32: small ztypes.
        // If the size is larger than 32 bytes, it still must be a multiple of
        // 4 bytes.  The only difference will be warp-level reductions, which
        // will use GB_cuda_shfl_down_large_ztype instead of tile.shfl_down.
        return (true) ;
    }

    // the type is not supported on the GPU
    return (false) ;
}

