//------------------------------------------------------------------------------
// GB_encodify_user_type: encode a user type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

uint64_t GB_encodify_user_type      // encode a user defined type
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GrB_Type type
)
{ 

    //--------------------------------------------------------------------------
    // check if the type is JIT'able
    //--------------------------------------------------------------------------

    if (type != NULL && type->hash == UINT64_MAX)
    { 
        // cannot JIT this type
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;
        (*suffix) = NULL ;
        return (UINT64_MAX) ;
    }

    //--------------------------------------------------------------------------
    // primary encoding of the user type
    //--------------------------------------------------------------------------

    GB_encodify_kcode (encoding, GB_JIT_KERNEL_USERTYPE) ;
    encoding->code = 0 ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    // if hash is zero, it denotes a builtin type
    uint64_t hash = type->hash ;
    encoding->suffix_len = (hash == 0) ? 0 : type->name_len ;
    (*suffix) = (hash == 0) ? NULL : type->name ;

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    hash = hash ^ GB_jitifyer_hash_encoding (encoding) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

