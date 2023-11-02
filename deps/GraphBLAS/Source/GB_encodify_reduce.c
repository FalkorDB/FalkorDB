//------------------------------------------------------------------------------
// GB_encodify_reduce: encode a GrB_reduce problem, including types and ops
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

uint64_t GB_encodify_reduce // encode a GrB_reduce problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    GrB_Monoid monoid,      // the monoid to enumify
    GrB_Matrix A            // input matrix to reduce
)
{ 

    //--------------------------------------------------------------------------
    // check if the monoid is JIT'able
    //--------------------------------------------------------------------------

    if (monoid->hash == UINT64_MAX)
    { 
        // cannot JIT this monoid
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;
        (*suffix) = NULL ;
        return (UINT64_MAX) ;
    }

    //--------------------------------------------------------------------------
    // primary encoding of the problem
    //--------------------------------------------------------------------------

    GB_enumify_reduce (&encoding->code, monoid, A) ;
    bool builtin = (monoid->hash == 0) ;
    encoding->kcode = GB_JIT_KERNEL_REDUCE ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    int32_t name_len = monoid->op->name_len ;
    encoding->suffix_len = (builtin) ? 0 : name_len ;
    (*suffix) = (builtin) ? NULL : monoid->op->name ;

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    uint64_t hash = GB_jitifyer_hash_encoding (encoding) ;
    hash = hash ^ monoid->hash ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

