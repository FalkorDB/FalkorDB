//------------------------------------------------------------------------------
// GB_compatible: check input and operators for type compatibility
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Check if the types for C<M> = accum (C,T) are all compatible,
// and (if present) make sure the size of C and M match.

#define GB_FREE_ALL ;
#include "GB.h"

GrB_Info GB_compatible          // SUCCESS if all is OK, *_MISMATCH otherwise
(
    const GrB_Type ctype,       // the type of C (matrix or scalar)
    const GrB_Matrix C,         // the output matrix C; NULL if C is a scalar
    const GrB_Matrix M,         // optional mask, NULL if no mask
    const bool Mask_struct,     // true if M is structural
    const GrB_BinaryOp accum,   // C<M> = accum(C,T) is computed
    const GrB_Type ttype,       // type of T
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M

    GrB_Info info ;

    //--------------------------------------------------------------------------
    // check accum compatibility
    //--------------------------------------------------------------------------

    if (accum != NULL)
    { 
        // Results T are accumlated via C<M>=accum(C,T)

        // For entries in C and T, c=z=accum(c,t) is computed, so C must
        // be compatible with both the ztype and xtype of accum, and T
        // must be compatible with the ytype of accum.

        // For entries in T but not C, c=t is assigned, so C and T must
        // be compatible.  This is the same as the condition below
        // when accum is NULL.

        GB_OK (GB_BinaryOp_compatible (accum, ctype, ctype, ttype,
            GB_ignore_code, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // check the types of C and T
    //--------------------------------------------------------------------------

    // C<M> = T, so C and T must be compatible.
    // also C<M> = accum(C,T) for entries in T but not C
    if (!GB_Type_compatible (ctype, ttype))
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Result of computation of type [%s]\n"
            "cannot be typecast to final output of type [%s]",
            ttype->name, ctype->name) ;
    }

    //--------------------------------------------------------------------------
    // check the mask
    //--------------------------------------------------------------------------

    return (GB_Mask_compatible (M, Mask_struct, C, 1, 1, Werk)) ;
}

