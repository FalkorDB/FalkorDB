//------------------------------------------------------------------------------
// GB_encodify_masker: encode a masker problem, including types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// FUTURE: allow the types of R, C, and Z to differ.

#include "GB.h"
#include "jitifyer/GB_stringify.h"

uint64_t GB_encodify_masker     // encode a masker problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    const GrB_Matrix R,         // may be NULL, for phase1
    const bool Rp_is_32,        // if true, R->p is 32 bit; else 64 bit
    const bool Rj_is_32,        // if true, R->h is 32 bit; else 64 bit
    const bool Ri_is_32,        // if true, R->i is 32 bit; else 64 bit
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix C,
    const GrB_Matrix Z
)
{ 

    //--------------------------------------------------------------------------
    // check if the R->type is JIT'able
    //--------------------------------------------------------------------------

    GrB_Type rtype = (R == NULL) ? NULL : R->type ;
    if (R != NULL && rtype->hash == UINT64_MAX)
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
    GB_enumify_masker (&encoding->code, R, Rp_is_32, Rj_is_32, Ri_is_32,
        M, Mask_struct, Mask_comp, C, Z) ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    // if hash is zero, it denotes a builtin type
    uint64_t hash = (rtype == NULL) ? 0 : rtype->hash ;
    encoding->suffix_len = (hash == 0) ? 0 : rtype->name_len ;
    (*suffix) = (hash == 0) ? NULL : rtype->name ;

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    hash = hash ^ GB_jitifyer_hash_encoding (encoding) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

