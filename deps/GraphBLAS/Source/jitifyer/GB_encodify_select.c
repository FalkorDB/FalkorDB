//------------------------------------------------------------------------------
// GB_encodify_select: encode a select problem, including types and op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

uint64_t GB_encodify_select     // encode an select problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    const GrB_Matrix C,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A
)
{ 

    //--------------------------------------------------------------------------
    // check if the op is JIT'able
    //--------------------------------------------------------------------------

    if (op != NULL && op->hash == UINT64_MAX)
    { 
        // cannot JIT this op
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;
        (*suffix) = NULL ;
        return (UINT64_MAX) ;
    }

    //--------------------------------------------------------------------------
    // primary encoding of the problem
    //--------------------------------------------------------------------------

    GB_encodify_kcode (encoding, kcode) ;
    GB_enumify_select (&encoding->code, C, op, flipij, A) ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    // if hash is zero, it denotes a builtin operator
    uint64_t hash = op->hash ;
    encoding->suffix_len = (hash == 0) ? 0 : op->name_len ;
    (*suffix) = (hash == 0) ? NULL : op->name ;

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    hash = hash ^ GB_jitifyer_hash_encoding (encoding) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

