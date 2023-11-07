//------------------------------------------------------------------------------
// GB_add: C = A+B, C<M>=A+B, and C<!M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_add computes C=A+B, C<M>=A+B, or C<!M>=A+B using the given operator
// element-wise on the matrices A and B.  The result is typecasted as needed.
// The pattern of C is the union of the pattern of A and B, intersection with
// the mask M, if present.  On input, the contents of C are undefined; it is
// an output-only matrix in a static header.

// Let the op be z=f(x,y) where x, y, and z have type xtype, ytype, and ztype.
// If both A(i,j) and B(i,j) are present, then:

//      C(i,j) = (ctype) op ((xtype) A(i,j), (ytype) B(i,j))

// If just A(i,j) is present but not B(i,j), then:

//      C(i,j) = (ctype) A (i,j)

// If just B(i,j) is present but not A(i,j), then:

//      C(i,j) = (ctype) B (i,j)

// ctype is the type of matrix C.  The pattern of C is the union of A and B.

// If A_and_B_are_disjoint is true, the intersection of A and B must be empty.
// This is used by GB_wait only, for merging the pending tuple matrix T into A.
// In this case, the result C is always sparse or hypersparse, not bitmap or
// full.  Any duplicate pending tuples have already been summed in T, so the
// intersection of T and A is always empty.

// Some methods should not exploit the mask, but leave it for later.
// See GB_ewise and GB_accum_mask: the only places where this function is
// called with a non-null mask M.  Both of those callers can handle the
// mask being applied later.  GB_add_sparsity determines whether or not the
// mask should be applied now, or later.

// If A and B are iso, the op is not positional, and op(A,B) == A == B, then C
// is iso.  If A and B are known to be disjoint, then op(A,B) is ignored when
// determining if C is iso.

// C on input is empty, see GB_add_phase2.c.

#include "GB_add.h"

#define GB_FREE_ALL ;

GrB_Info GB_add             // C=A+B, C<M>=A+B, or C<!M>=A+B
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask for C, unused if NULL
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use !M
    bool *mask_applied,     // if true, the mask was applied
    const GrB_Matrix A,     // input A matrix
    const GrB_Matrix B,     // input B matrix
    const bool is_eWiseUnion,   // if true, eWiseUnion, else eWiseAdd
    const GrB_Scalar alpha, // alpha and beta ignored for eWiseAdd,
    const GrB_Scalar beta,  // nonempty scalars for GxB_eWiseUnion
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    const bool A_and_B_are_disjoint,   // if true, A and B are disjoint
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    ASSERT (mask_applied != NULL) ;
    (*mask_applied) = false ;

    ASSERT_MATRIX_OK (A, "A for add", GB0) ;
    ASSERT_MATRIX_OK (B, "B for add", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for add", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for add", GB0) ;
    ASSERT (A->vdim == B->vdim && A->vlen == B->vlen) ;
    ASSERT (GB_IMPLIES (M != NULL, A->vdim == M->vdim && A->vlen == M->vlen)) ;

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    // TODO: some cases can allow M, A, and/or B to be jumbled
    GB_MATRIX_WAIT (M) ;        // cannot be jumbled
    GB_MATRIX_WAIT (A) ;        // cannot be jumbled
    GB_MATRIX_WAIT (B) ;        // cannot be jumbled

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    bool apply_mask ;
    int C_sparsity = GB_add_sparsity (&apply_mask, M, Mask_struct, Mask_comp,
        A, B) ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    int64_t Cnvec = 0 , Cnvec_nonempty = 0  ;
    int64_t *Cp     = NULL ; size_t Cp_size = 0 ;
    int64_t *Ch     = NULL ; size_t Ch_size = 0 ; 
    int64_t *C_to_M = NULL ; size_t C_to_M_size = 0 ;
    int64_t *C_to_A = NULL ; size_t C_to_A_size = 0 ;
    int64_t *C_to_B = NULL ; size_t C_to_B_size = 0 ;
    bool Ch_is_Mh ;
    int C_ntasks = 0, C_nthreads ;
    GB_task_struct *TaskList = NULL ; size_t TaskList_size = 0 ;

    //--------------------------------------------------------------------------
    // phase0: finalize the sparsity C and find the vectors in C
    //--------------------------------------------------------------------------

    info = GB_add_phase0 (
        // computed by by phase0:
        &Cnvec, &Ch, &Ch_size, &C_to_M, &C_to_M_size, &C_to_A, &C_to_A_size,
        &C_to_B, &C_to_B_size, &Ch_is_Mh,
        // input/output to phase0:
        &C_sparsity,
        // original input:
        (apply_mask) ? M : NULL, A, B, Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (info) ;
    }

    GBURBLE ("add:(%s<%s%s>=%s+%s) ",
        GB_sparsity_char (C_sparsity),
        GB_sparsity_char_matrix (M),
        ((M != NULL) && !apply_mask) ? " (mask later)" : "",
        GB_sparsity_char_matrix (A),
        GB_sparsity_char_matrix (B)) ;

    //--------------------------------------------------------------------------
    // phase1: split C into tasks, and count entries in each vector of C
    //--------------------------------------------------------------------------

    if (C_sparsity == GxB_SPARSE || C_sparsity == GxB_HYPERSPARSE)
    {

        //----------------------------------------------------------------------
        // C is sparse or hypersparse: slice and analyze the C matrix
        //----------------------------------------------------------------------

        // phase1a: split C into tasks
        info = GB_ewise_slice (
            // computed by phase1a
            &TaskList, &TaskList_size, &C_ntasks, &C_nthreads,
            // computed by phase0:
            Cnvec, Ch, C_to_M, C_to_A, C_to_B, Ch_is_Mh,
            // original input:
            (apply_mask) ? M : NULL, A, B, Werk) ;
        if (info != GrB_SUCCESS)
        { 
            // out of memory; free everything allocated by GB_add_phase0
            GB_FREE (&Ch, Ch_size) ;
            GB_FREE_WORK (&C_to_M, C_to_M_size) ;
            GB_FREE_WORK (&C_to_A, C_to_A_size) ;
            GB_FREE_WORK (&C_to_B, C_to_B_size) ;
            return (info) ;
        }

        // count the number of entries in each vector of C
        info = GB_add_phase1 (
            // computed or used by phase1:
            &Cp, &Cp_size, &Cnvec_nonempty, A_and_B_are_disjoint,
            // from phase1a:
            TaskList, C_ntasks, C_nthreads,
            // from phase0:
            Cnvec, Ch, C_to_M, C_to_A, C_to_B, Ch_is_Mh,
            // original input:
            (apply_mask) ? M : NULL, Mask_struct, Mask_comp, A, B, Werk) ;
        if (info != GrB_SUCCESS)
        { 
            // out of memory; free everything allocated by GB_add_phase0
            GB_FREE_WORK (&TaskList, TaskList_size) ;
            GB_FREE (&Ch, Ch_size) ;
            GB_FREE_WORK (&C_to_M, C_to_M_size) ;
            GB_FREE_WORK (&C_to_A, C_to_A_size) ;
            GB_FREE_WORK (&C_to_B, C_to_B_size) ;
            return (info) ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // C is bitmap or full: only determine how many threads to use
        //----------------------------------------------------------------------

        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        C_nthreads = GB_nthreads (A->vlen * A->vdim, chunk, nthreads_max) ;
    }

    //--------------------------------------------------------------------------
    // phase2: compute the entries (indices and values) in each vector of C
    //--------------------------------------------------------------------------

    // Cp and Ch are either freed by phase2, or transplanted into C.
    // Either way, they are not freed here.

    info = GB_add_phase2 (
        // computed or used by phase2:
        C, ctype, C_is_csc, op, A_and_B_are_disjoint,
        // from phase1
        &Cp, Cp_size, Cnvec_nonempty,
        // from phase1a:
        TaskList, C_ntasks, C_nthreads,
        // from phase0:
        Cnvec, &Ch, Ch_size, C_to_M, C_to_A, C_to_B, Ch_is_Mh, C_sparsity,
        // original input:
        (apply_mask) ? M : NULL, Mask_struct, Mask_comp, A, B,
        is_eWiseUnion, alpha, beta, Werk) ;

    // Ch and Cp must not be freed; they are now C->h and C->p.
    // If the method failed, Cp and Ch have already been freed.

    // free workspace
    GB_FREE_WORK (&TaskList, TaskList_size) ;
    GB_FREE_WORK (&C_to_M, C_to_M_size) ;
    GB_FREE_WORK (&C_to_A, C_to_A_size) ;
    GB_FREE_WORK (&C_to_B, C_to_B_size) ;

    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C output for add", GB0) ;
    (*mask_applied) = apply_mask ;
    return (GrB_SUCCESS) ;
}

