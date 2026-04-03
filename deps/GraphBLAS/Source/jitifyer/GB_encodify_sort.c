//------------------------------------------------------------------------------
// GB_encodify_sort: encode a sort problem, including types and op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

uint64_t GB_encodify_sort       // encode a sort problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    // input/output
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp binaryop
)
{ 

    //--------------------------------------------------------------------------
    // check if the binaryop is JIT'able
    //--------------------------------------------------------------------------

    if (binaryop != NULL && binaryop->hash == UINT64_MAX)
    { 
        // cannot JIT this binaryop
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;
        (*suffix) = NULL ;
        return (UINT64_MAX) ;
    }

    //--------------------------------------------------------------------------
    // primary encoding of the problem
    //--------------------------------------------------------------------------

    GB_encodify_kcode (encoding, kcode) ;
    GB_enumify_sort (&encoding->code, C, binaryop) ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    // if hash is zero, it denotes a builtin binary operator
    uint64_t hash = binaryop->hash ;
    encoding->suffix_len = (hash == 0) ? 0 : binaryop->name_len ;
    (*suffix) = (hash == 0) ? NULL : binaryop->name ;

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    hash = hash ^ GB_jitifyer_hash_encoding (encoding) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

