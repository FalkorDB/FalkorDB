//------------------------------------------------------------------------------
// GB_encodify_assign: encode an assign problem, including types and op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

uint64_t GB_encodify_assign     // encode an assign problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    // C matrix:
    GrB_Matrix C,
    bool C_replace,
    // index types:
    int Ikind,              // 0: all (no I), 1: range, 2: stride, 3: list
    int Jkind,              // ditto
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // operator:
    GrB_BinaryOp accum,     // the accum operator (may be NULL)
    // A matrix or scalar
    GrB_Matrix A,           // NULL for scalar assignment
    GrB_Type scalar_type,
    int assign_kind         // 0: assign, 1: subassign, 2: row, 3: col
)
{

    //--------------------------------------------------------------------------
    // check if the accum operator or C->type is JIT'able
    //--------------------------------------------------------------------------

    if ((accum != NULL && accum->hash == UINT64_MAX) ||
        (accum == NULL && C->type->hash == UINT64_MAX))
    { 
        // cannot JIT this accum operator or type
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;
        (*suffix) = NULL ;
        return (UINT64_MAX) ;
    }

    //--------------------------------------------------------------------------
    // primary encoding of the problem
    //--------------------------------------------------------------------------

    encoding->kcode = kcode ;
    GB_enumify_assign (&encoding->code, C, C_replace, Ikind, Jkind,
        M, Mask_struct, Mask_comp, accum, A, scalar_type, assign_kind) ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    // if hash is zero, it denotes a builtin binary operator or type
    uint64_t hash = 0 ;
    if (accum == NULL)
    { 
        // use the hash and name of the C->type
        hash = C->type->hash ;
        encoding->suffix_len = (hash == 0) ? 0 : C->type->name_len ;
        (*suffix) = (hash == 0) ? NULL : C->type->name ;
    }
    else
    { 
        // use the hash and name of the accum operator
        hash = accum->hash ;
        encoding->suffix_len = (hash == 0) ? 0 : accum->name_len ;
        (*suffix) = (hash == 0) ? NULL : accum->name ;
    }

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    hash = hash ^ GB_jitifyer_hash_encoding (encoding) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

