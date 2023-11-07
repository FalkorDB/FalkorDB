//------------------------------------------------------------------------------
// GB_subassign_24: make a deep copy of a sparse or dense matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A, making a deep copy into an existing non-shallow matrix C, but
// possibly reusing parts of C if C is dense.  See also GB_dup.
// C can have any sparsity structure on input.  C takes on the same sparsity
// structure as A.

// Handles arbitrary typecasting.

// if C sparse and A dense/full, C is converted to full, ignoring
// C->sparsity_control.  C is conformed to its desired sparsity structure later.

// A can be jumbled, in which case C is also jumbled.
// A can have any sparsity structure (sparse, hyper, bitmap, or full).

#include "GB_subassign_dense.h"
#include "GB_Pending.h"
#define GB_FREE_ALL ;

GrB_Info GB_subassign_24    // C = A, copy A into an existing matrix C
(
    GrB_Matrix C,           // output matrix to modify
    const GrB_Matrix A,     // input matrix to copy
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A
    ASSERT (!GB_is_shallow (C)) ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for GB_subassign_24", GB0) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (GB_PENDING_OK (C)) ;

    ASSERT_MATRIX_OK (A, "A for GB_subassign_24", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;

    C->jumbled = false ;        // prior contents of C are discarded
    const bool C_iso = A->iso ; // C is iso if A is iso

    // save the sparsity control of C
    int C_sparsity_control = C->sparsity_control ;
    float C_hyper_switch = C->hyper_switch ;
    float C_bitmap_switch = C->bitmap_switch ;

    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // C = A
    //--------------------------------------------------------------------------

    bool copy_dense_A_to_C =            // copy from dense A to dense C if:
        (
            GB_is_dense (C)             // both A and C are dense
            && GB_is_dense (A)
            && !(A->jumbled)            // A cannot be jumbled
            && C->vdim == A->vdim       // A and C have the same size
            && C->vlen == A->vlen
            && C->is_csc == A->is_csc   // A and C have the same format
            && C->x != NULL             // C->x exists
        ) ;

    if (copy_dense_A_to_C)
    { 

        //----------------------------------------------------------------------
        // discard the pattern of C
        //----------------------------------------------------------------------

        // make C full, if not full already
        C->nzombies = 0 ;                   // overwrite any zombies
        GB_Pending_free (&(C->Pending)) ;   // abandon all pending tuples
        C->iso = C_iso ;
        GB_convert_any_to_full (C) ;        // ensure C is full

    }
    else
    { 

        //----------------------------------------------------------------------
        // copy the pattern from A to C
        //----------------------------------------------------------------------

        // clear prior content of C, but keep the CSR/CSC format and its type
        bool C_is_csc = C->is_csc ;
        GB_phybix_free (C) ;
        // copy the pattern, not the values
        // set C->iso = C_iso   OK
        GB_OK (GB_dup_worker (&C, C_iso, A, false, C->type)) ;
        C->is_csc = C_is_csc ;      // do not change the CSR/CSC format of C
        // GB_assign_prep has assigned the C->x iso value, but this has just
        // been cleared, so it needs to be reassigned below by GB_cast_matrix.
    }

    //--------------------------------------------------------------------------
    // copy the values from A to C, typecasting as needed
    //--------------------------------------------------------------------------

    if (C->type != A->type)
    { 
        GBURBLE ("(typecast) ") ;
    }

    GB_OK (GB_cast_matrix (C, A)) ;

    //--------------------------------------------------------------------------
    // restore the sparsity control of C
    //--------------------------------------------------------------------------

    C->sparsity_control = C_sparsity_control ;
    C->hyper_switch = C_hyper_switch ;
    C->bitmap_switch = C_bitmap_switch ;

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C result for GB_subassign_24", GB0) ;
    return (GrB_SUCCESS) ;
}

