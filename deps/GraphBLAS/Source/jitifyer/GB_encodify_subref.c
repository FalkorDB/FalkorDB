//------------------------------------------------------------------------------
// GB_encodify_subref: encode a subref problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

uint64_t GB_encodify_subref     // encode an subref problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    // C matrix:
    GrB_Matrix C,
    // index types:
    bool I_is_32,           // if true, I is 32-bits; else 64
    bool J_is_32,           // if true, J is 32-bits; else 64 (0 if not used)
    int Ikind,              // 0: all (no I), 1: range, 2: stride, 3: list
    int Jkind,              // ditto, or 0 if not used
    bool need_qsort,        // true if qsort needs to be called
    GrB_Matrix R,
    // A matrix:
    GrB_Matrix A
)
{

    //--------------------------------------------------------------------------
    // check if the C->type is JIT'able
    //--------------------------------------------------------------------------

    if (C->type->hash == UINT64_MAX)
    { 
        // cannot JIT this type
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;
        (*suffix) = NULL ;
        return (UINT64_MAX) ;
    }

    //--------------------------------------------------------------------------
    // primary encoding of the problem
    //--------------------------------------------------------------------------

    GB_encodify_kcode (encoding, kcode) ;
    GB_enumify_subref (&encoding->code,
        C, I_is_32, J_is_32, Ikind, Jkind, need_qsort, R, A) ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    // if hash is zero, it denotes a builtin type
    uint64_t hash = C->type->hash ;
    encoding->suffix_len = (hash == 0) ? 0 : C->type->name_len ;
    (*suffix) = (hash == 0) ? NULL : C->type->name ;

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    hash = hash ^ GB_jitifyer_hash_encoding (encoding) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

