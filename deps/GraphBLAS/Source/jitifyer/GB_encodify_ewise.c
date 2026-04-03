//------------------------------------------------------------------------------
// GB_encodify_ewise: encode a ewise problem, including types and op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

uint64_t GB_encodify_ewise      // encode an ewise problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    const bool is_eWiseMult,    // if true, method is emult
    // C matrix:
    const bool C_iso,
    const bool C_in_iso,
    const int C_sparsity,
    const GrB_Type ctype,
    const bool Cp_is_32,
    const bool Cj_is_32,
    const bool Ci_is_32,
    // M matrix:
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    // operator:
    const GrB_BinaryOp binaryop,
    const bool flipij,
    const bool flipxy,
    // A and B:
    const GrB_Matrix A,         // NULL for apply bind1st
    const GrB_Matrix B          // NULL for apply bind2nd
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

    bool is_eWiseAdd   = (kcode == GB_JIT_KERNEL_ADD) ;
    bool is_eWiseUnion = (kcode == GB_JIT_KERNEL_UNION) ;
    bool is_kron       = (kcode == GB_JIT_KERNEL_KRONER) ;

    GB_encodify_kcode (encoding, kcode) ;
    GB_enumify_ewise (&encoding->code, is_eWiseMult, is_eWiseUnion, is_kron,
        is_eWiseAdd, C_iso, C_in_iso, C_sparsity, ctype,
        Cp_is_32, Cj_is_32, Ci_is_32,
        M, Mask_struct, Mask_comp, binaryop, flipij, flipxy, A, B) ;

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

