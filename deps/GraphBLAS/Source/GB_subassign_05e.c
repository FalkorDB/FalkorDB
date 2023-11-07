//------------------------------------------------------------------------------
// GB_subassign_05e: C(:,:)<M,struct> = scalar ; no S, C empty, M structural
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.

// Method 05e: C(:,:)<M,struct> = scalar ; no S

// M:           present
// Mask_comp:   false
// Mask_struct: true
// C_replace:   false
// accum:       NULL
// A:           scalar
// S:           none

// C and M can have any sparsity on input.  The content of C is replace with
// the structure of M, and the values of C are all set to the scalar.  If M is
// bitmap, only assignments where (Mb [pC] == 1) are needed, but it's faster to
// just assign all entries.

// C is always iso, and its iso value has been assigned by GB_assign_prep.

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL

GrB_Info GB_subassign_05e
(
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M
    ASSERT (C->iso) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for subassign method_05e", GB0) ;
    ASSERT_MATRIX_OK (M, "M for subassign method_05e", GB0) ;
    ASSERT (GB_nnz (C) == 0) ;

    // M can be jumbled, in which case C is jumbled on output 
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_PENDING (M)) ;

    //--------------------------------------------------------------------------
    // Method 05e: C(:,:)<M> = x ; C is empty, x is a scalar, M is structural
    //--------------------------------------------------------------------------

    // Time: Optimal:  the method must iterate over all entries in M,
    // and the time is O(nnz(M)).  This is also the size of C.

    //--------------------------------------------------------------------------
    // allocate C and create its pattern
    //--------------------------------------------------------------------------

    // clear prior content and then create a copy of the pattern of M.  Keep
    // the same type and CSR/CSC for C.  Allocate C->x and assign to it the
    // scalar.

    bool C_is_csc = C->is_csc ;
    GB_phybix_free (C) ;
    // set C->iso = true    OK
    GB_OK (GB_dup_worker (&C, true, M, false, C->type)) ;
    C->is_csc = C_is_csc ;
    GB_cast_scalar (C->x, C->type->code, scalar, scalar_type->code, scalar_type->size) ;

    C->jumbled = M->jumbled ;       // C is jumbled if M is jumbled
    ASSERT_MATRIX_OK (C, "C output for subassign method_05e", GB0) ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    return (GrB_SUCCESS) ;
}

