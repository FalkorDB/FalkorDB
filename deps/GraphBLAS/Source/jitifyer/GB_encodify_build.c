//------------------------------------------------------------------------------
// GB_encodify_build: encode a build problem, including types and op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

uint64_t GB_encodify_build      // encode an build problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    const GrB_BinaryOp dup,     // operator for summing up duplicates
    const GrB_Type ttype,       // type of Tx array
    const GrB_Type stype,       // type of Sx array
    bool Ti_is_32,              // if true, Ti is uint32_t, else uint64_t
    bool I_is_32,               // if true, I_work is uint32_t else uint64_t
    bool K_is_32,               // if true, K_work is uint32_t else uint64_t
    bool K_is_null,             // if true, K_work is NULL
    bool no_duplicates          // if true, no duplicates appear
)
{ 

    //--------------------------------------------------------------------------
    // check if the dup operator is JIT'able
    //--------------------------------------------------------------------------

    if (dup != NULL && dup->hash == UINT64_MAX)
    { 
        // cannot JIT this dup operator
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;
        (*suffix) = NULL ;
        return (UINT64_MAX) ;
    }

    //--------------------------------------------------------------------------
    // primary encoding of the problem
    //--------------------------------------------------------------------------

    GB_encodify_kcode (encoding, kcode) ;
    GB_enumify_build (&encoding->code, dup, ttype, stype,
        Ti_is_32, I_is_32, K_is_32, K_is_null, no_duplicates) ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    // if hash is zero, it denotes a builtin binary operator
    uint64_t hash = dup->hash ;
    encoding->suffix_len = (hash == 0) ? 0 : dup->name_len ;
    (*suffix) = (hash == 0) ? NULL : dup->name ;

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    hash = hash ^ GB_jitifyer_hash_encoding (encoding) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

