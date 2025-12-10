//------------------------------------------------------------------------------
// GB_encodify_mxm: encode a GrB_mxm problem, including types and ops
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

uint64_t GB_encodify_mxm        // encode a GrB_mxm problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    // C matrix:
    const bool C_iso,
    const bool C_in_iso,
    const int C_sparsity,
    const GrB_Type ctype,
    bool Cp_is_32,          // if true, C->p is 32-bit; else 64
    bool Cj_is_32,          // if true, C->h is 32-bit; else 64
    bool Ci_is_32,          // if true, C->i is 32-bit; else 64
    // M matrix:
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    // semiring:
    const GrB_Semiring semiring,
    const bool flipxy,
    // A and B:
    const GrB_Matrix A,
    const GrB_Matrix B
)
{ 

    //--------------------------------------------------------------------------
    // check if the semiring is JIT'able
    //--------------------------------------------------------------------------

    if (semiring->hash == UINT64_MAX)
    { 
        // cannot JIT this semiring
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;
        (*suffix) = NULL ;
        return (UINT64_MAX) ;
    }

    //--------------------------------------------------------------------------
    // primary encoding of the problem
    //--------------------------------------------------------------------------

    GB_encodify_kcode (encoding, kcode) ;
    GB_enumify_mxm (&encoding->code, C_iso, C_in_iso, C_sparsity, ctype,
        Cp_is_32, Cj_is_32, Ci_is_32, M, Mask_struct, Mask_comp, semiring,
        flipxy, A, B) ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    // if hash is zero, it denotes a builtin semiring
    uint64_t hash = semiring->hash ;
    encoding->suffix_len = (hash == 0) ? 0 : semiring->name_len ;
    (*suffix) = (hash == 0) ? NULL : semiring->name ;

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    hash = hash ^ GB_jitifyer_hash_encoding (encoding) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

