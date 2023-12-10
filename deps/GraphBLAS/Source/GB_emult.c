//------------------------------------------------------------------------------
// GB_emult: C = A.*B, C<M>=A.*B, or C<!M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_emult, does C=A.*B, C<M>=A.*B, C<!M>=A.*B, using the given operator
// element-wise on the matrices A and B.  The result is typecasted as needed.
// The pattern of C is the intersection of the pattern of A and B, intersection
// with the mask M or !M, if present.

// Let the op be z=f(x,y) where x, y, and z have type xtype, ytype, and ztype.
// If both A(i,j) and B(i,j) are present, then:

//      C(i,j) = (ctype) op ((xtype) A(i,j), (ytype) B(i,j))

// If just A(i,j) is present but not B(i,j), or
// if just B(i,j) is present but not A(i,j), then C(i,j) is not present.

// ctype is the type of matrix C, and currently it is always op->ztype,
// but this might change in the future.

// The pattern of C is the intersection of A and B, and also intersection with
// M if present and not complemented.

// TODO: if C is bitmap on input and C_sparsity is GxB_BITMAP, then C=A.*B,
// C<M>=A.*B and C<M>+=A.*B can all be done in-place.  Also, if C is bitmap
// but T<M>=A.*B is sparse (M sparse, with A and B bitmap), then it too can
// be done in place.

#include "GB_emult.h"
#include "GB_add.h"
#include "GB_binop.h"

#define GB_FREE_WORKSPACE                       \
{                                               \
    GB_FREE_WORK (&TaskList, TaskList_size) ;   \
    GB_FREE_WORK (&C_to_M, C_to_M_size) ;       \
    GB_FREE_WORK (&C_to_A, C_to_A_size) ;       \
    GB_FREE_WORK (&C_to_B, C_to_B_size) ;       \
}

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORKSPACE ;         \
    GB_phybix_free (C) ;        \
}

GrB_Info GB_emult           // C=A.*B, C<M>=A.*B, or C<!M>=A.*B
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask, unused if NULL
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use !M
    bool *mask_applied,     // if true, the mask was applied
    const GrB_Matrix A,     // input A matrix
    const GrB_Matrix B,     // input B matrix
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    ASSERT_MATRIX_OK (A, "A for emult", GB0) ;
    ASSERT_MATRIX_OK (B, "B for emult", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for emult", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for emult", GB0) ;
    ASSERT (A->vdim == B->vdim && A->vlen == B->vlen) ;
    ASSERT (GB_IMPLIES (M != NULL, A->vdim == M->vdim && A->vlen == M->vlen)) ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    GB_task_struct *TaskList = NULL ; size_t TaskList_size = 0 ;
    int64_t *restrict C_to_M = NULL ; size_t C_to_M_size = 0 ;
    int64_t *restrict C_to_A = NULL ; size_t C_to_A_size = 0 ;
    int64_t *restrict C_to_B = NULL ; size_t C_to_B_size = 0 ;

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    // some cases can allow M, A, and/or B to be jumbled
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (M) ;
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (B) ;

    //--------------------------------------------------------------------------
    // determine the sparsity of C and the method to use
    //--------------------------------------------------------------------------

    bool apply_mask ;           // if true, mask is applied during emult
    int ewise_method ;          // method to use
    int C_sparsity = GB_emult_sparsity (&apply_mask, &ewise_method,
        M, Mask_comp, A, B) ;

    //--------------------------------------------------------------------------
    // get the opcode and determine if f(x,y) == f(y,x)
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    if (op->xtype == GrB_BOOL)
    { 
        opcode = GB_boolean_rename (opcode) ;
    }

    bool op_is_commutative ;
    switch (opcode)
    {
        case GB_MIN_binop_code     :    // z = min(x,y)
        case GB_MAX_binop_code     :    // z = max(x,y)
        case GB_PLUS_binop_code    :    // z = x + y
        case GB_TIMES_binop_code   :    // z = x * y
        case GB_PAIR_binop_code    :    // z = 1
        case GB_ISEQ_binop_code    :    // z = (x == y)
        case GB_ISNE_binop_code    :    // z = (x != y)
        case GB_EQ_binop_code      :    // z = (x == y)
        case GB_NE_binop_code      :    // z = (x != y)
        case GB_LOR_binop_code     :    // z = x || y
        case GB_LAND_binop_code    :    // z = x && y
        case GB_LXOR_binop_code    :    // z = x != y
        case GB_HYPOT_binop_code   :    // z = hypot (x,y)
        case GB_BOR_binop_code     :    // z = (x | y), bitwise or
        case GB_BAND_binop_code    :    // z = (x & y), bitwise and
        case GB_BXOR_binop_code    :    // z = (x ^ y), bitwise xor
        case GB_BXNOR_binop_code   :    // z = ~(x ^ y), bitwise xnor
            op_is_commutative = true ;
            break ;
        default : 
            op_is_commutative = false ;
    }

    //--------------------------------------------------------------------------
    // C<M or !M> = A.*B
    //--------------------------------------------------------------------------

    switch (ewise_method)
    {

        case GB_EMULT_METHOD1_ADD :  // A and B both full (or as-if-full)

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      full    .           full            full    (GB_add)
            //      ------------------------------------------
            //      C       <M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      full            full    (GB_add or 4)
            //      bitmap  bitmap      full            full    (GB_add or 7)
            //      bitmap  full        full            full    (GB_add or 7)
            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      bitmap  sparse      full            full    (GB_add or 6)
            //      bitmap  bitmap      full            full    (GB_add or 7)
            //      bitmap  full        full            full    (GB_add or 7)

            // A and B are both full (or as-if-full).  The mask M may be
            // anything.  GB_add computes the same thing in this case, so it is
            // used instead, to reduce the code needed for GB_emult.

            return (GB_add (C, ctype, C_is_csc, M, Mask_struct,
                Mask_comp, mask_applied, A, B, false, NULL, NULL,
                op, false, Werk)) ;

        case GB_EMULT_METHOD2 :  // A sparse/hyper, B bitmap/full

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      sparse  .           sparse          bitmap  (method: 2)
            //      sparse  .           sparse          full    (method: 2)
            //      ------------------------------------------
            //      C       <M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  bitmap      sparse          bitmap  (method: 2)
            //      sparse  bitmap      sparse          full    (method: 2)
            //      sparse  full        sparse          bitmap  (method: 2)
            //      sparse  full        sparse          full    (method: 2)
            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          bitmap  (2: M later)
            //      sparse  sparse      sparse          full    (2: M later)
            //      ------------------------------------------
            //      C       <!M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  bitmap      sparse          bitmap  (method: 2)
            //      sparse  bitmap      sparse          full    (method: 2)
            //      sparse  full        sparse          bitmap  (method: 2)
            //      sparse  full        sparse          full    (method: 2)

            // A is sparse/hyper and B is bitmap/full.  M is either not
            // present, not applied (!M when sparse/hyper), or bitmap/full.
            // This method does not handle the case when M is sparse/hyper,
            // unless M is ignored and applied later.

            return (GB_emult_02 (C, ctype, C_is_csc,
                (apply_mask) ? M : NULL, Mask_struct, Mask_comp,
                A, B, op, Werk)) ;

        case GB_EMULT_METHOD3 :  // A bitmap/full, B sparse/hyper

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      sparse  .           bitmap          sparse  (method: 3)
            //      sparse  .           full            sparse  (method: 3)
            //      ------------------------------------------
            //      C       <M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  bitmap      bitmap          sparse  (method: 3)
            //      sparse  bitmap      full            sparse  (method: 3)
            //      sparse  full        bitmap          sparse  (method: 3)
            //      sparse  full        full            sparse  (method: 3)
            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      bitmap          sparse  (3: M later)
            //      sparse  sparse      full            sparse  (3: M later)
            //      ------------------------------------------
            //      C       <!M> =      A       .*      B
            //      ------------------------------------------
            //      sparse  bitmap      bitmap          sparse  (method: 3)
            //      sparse  bitmap      full            sparse  (method: 3)
            //      sparse  full        bitmap          sparse  (method: 3)
            //      sparse  full        full            sparse  (method: 3)

            // A is bitmap/full and B is sparse/hyper.
            // M is not present, not applied, or bitmap/full
            // Note that A and B are flipped.

            if (op_is_commutative)
            {
                // replace A.*B with B.*A, and use GB_emult_02, since the op is
                // commutative.  No need to change the op or flip it by using
                // f(y,x).  Just swap A and B.  This allows GB_emult_03 to
                // cover fewer cases via GB_NO_COMMUTATIVE_BINARY_OPS in the
                // GB_binop_factory.
                return (GB_emult_02 (C, ctype, C_is_csc,
                    (apply_mask) ? M : NULL, Mask_struct, Mask_comp,
                    B, A, op, Werk)) ;
            }
            else
            {
                // the op is not commutative: use GB_emult_03
                return (GB_emult_03 (C, ctype, C_is_csc,
                    (apply_mask) ? M : NULL, Mask_struct, Mask_comp,
                    A, B, op, Werk)) ;
            }

        case GB_EMULT_METHOD8 : 

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      sparse  .           sparse          sparse  (method: 8)
            //      ------------------------------------------
            //      C       <M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          sparse  (method: 8)
            //      sparse  bitmap      sparse          sparse  (method: 8)
            //      sparse  full        sparse          sparse  (method: 8)
            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          sparse  (8: M later)
            //      sparse  bitmap      sparse          sparse  (method: 8)
            //      sparse  full        sparse          sparse  (method: 8)

            // TODO: break Method8 into different methods
            break ;

        case GB_EMULT_METHOD5 :   // C is bitmap, M is not present

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      bitmap  .           bitmap          bitmap  (method: 5)
            //      bitmap  .           bitmap          full    (method: 5)
            //      bitmap  .           full            bitmap  (method: 5)

        case GB_EMULT_METHOD6 :   // C is bitmap, !M is sparse/hyper

            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      bitmap  sparse      bitmap          bitmap  (method: 6)
            //      bitmap  sparse      bitmap          full    (method: 6)
            //      bitmap  sparse      full            bitmap  (method: 6)
            //      bitmap  sparse      full            full    (GB_add or 6)

        case GB_EMULT_METHOD7 :   // C is bitmap, M is bitmap/full

            //      ------------------------------------------
            //      C      <M> =        A       .*      B
            //      ------------------------------------------
            //      bitmap  bitmap      bitmap          bitmap  (method: 7)
            //      bitmap  bitmap      bitmap          full    (method: 7)
            //      bitmap  bitmap      full            bitmap  (method: 7)
            //      bitmap  bitmap      full            full    (GB_add or 7)
            //      bitmap  full        bitmap          bitmap  (method: 7)
            //      bitmap  full        bitmap          full    (method: 7)
            //      bitmap  full        full            bitmap  (method: 7)
            //      bitmap  full        full            full    (GB_add or 7)
            //      ------------------------------------------
            //      C      <!M> =       A       .*      B
            //      ------------------------------------------
            //      bitmap  bitmap      bitmap          bitmap  (method: 7)
            //      bitmap  bitmap      bitmap          full    (method: 7)
            //      bitmap  bitmap      full            bitmap  (method: 7)
            //      bitmap  bitmap      full            full    (GB_add or 7)
            //      bitmap  full        bitmap          bitmap  (method: 7)
            //      bitmap  full        bitmap          full    (method: 7)
            //      bitmap  full        full            bitmap  (method: 7)
            //      bitmap  full        full            full    (GB_add or 7)

            // For methods 5, 6, and 7, C is constructed as bitmap.
            // Both A and B are bitmap/full.  M is either not present,
            // complemented, or not complemented and bitmap/full.  The
            // case when M is not complemented and sparse/hyper is handled
            // by method 4, which constructs C as sparse/hyper (the same
            // structure as M), not bitmap.

            return (GB_emult_bitmap (C, ewise_method, ctype, C_is_csc,
                M, Mask_struct, Mask_comp, mask_applied, A, B,
                op, Werk)) ;

        case GB_EMULT_METHOD4 : 

            //      ------------------------------------------
            //      C       <M>=        A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      bitmap          bitmap  (method: 4)
            //      sparse  sparse      bitmap          full    (method: 4)
            //      sparse  sparse      full            bitmap  (method: 4)
            //      sparse  sparse      full            full    (GB_add or 4)

            return (GB_emult_04 (C, ctype, C_is_csc, M, Mask_struct,
                mask_applied, A, B, op, Werk)) ;

        case GB_EMULT_METHOD9 : break ; // punt

            //      ------------------------------------------
            //      C       <M>=        A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          bitmap  (method: 9)
            //      sparse  sparse      sparse          full    (method: 9)

            // TODO: this will use Method9 (M,A,B, flipxy=false)

            // The method will compute the 2-way intersection of M and A,
            // using the same parallization as C=A.*B when both A and B are
            // both sparse.  It will then lookup B in O(1) time.
            // M and A must not be jumbled.

        case GB_EMULT_METHOD10 : break ; // punt

            //      ------------------------------------------
            //      C       <M>=        A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      bitmap          sparse  (method: 10)
            //      sparse  sparse      full            sparse  (method: 10)

            // TODO: this will use Method10 (M,B,A, flipxy=true)
            // M and B must not be jumbled.

        default:;
    }

    //--------------------------------------------------------------------------
    // Method8 (and for now, Method9 and Method10)
    //--------------------------------------------------------------------------

    ASSERT (C_sparsity == GxB_SPARSE || C_sparsity == GxB_HYPERSPARSE) ;

    GB_MATRIX_WAIT (M) ;
    GB_MATRIX_WAIT (A) ;
    GB_MATRIX_WAIT (B) ;

    GBURBLE ("emult:(%s<%s>=%s.*%s) ",
        GB_sparsity_char (C_sparsity),
        GB_sparsity_char_matrix (M),
        GB_sparsity_char_matrix (A),
        GB_sparsity_char_matrix (B)) ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    int64_t Cnvec, Cnvec_nonempty ;
    int64_t *Cp = NULL ; size_t Cp_size = 0 ;
    const int64_t *Ch = NULL ; size_t Ch_size = 0 ;
    int C_ntasks = 0, C_nthreads ;

    //--------------------------------------------------------------------------
    // phase0: finalize the sparsity C and find the vectors in C
    //--------------------------------------------------------------------------

    GB_OK (GB_emult_08_phase0 (
        // computed by phase0:
        &Cnvec, &Ch, &Ch_size, &C_to_M, &C_to_M_size, &C_to_A, &C_to_A_size,
        &C_to_B, &C_to_B_size,
        // input/output to phase0:
        &C_sparsity,
        // original input:
        (apply_mask) ? M : NULL, A, B, Werk)) ;

    // C is still sparse or hypersparse, not bitmap or full
    ASSERT (C_sparsity == GxB_SPARSE || C_sparsity == GxB_HYPERSPARSE) ;

    //--------------------------------------------------------------------------
    // phase1: split C into tasks, and count entries in each vector of C
    //--------------------------------------------------------------------------

    // phase1a: split C into tasks
    GB_OK (GB_ewise_slice (
        // computed by phase1a:
        &TaskList, &TaskList_size, &C_ntasks, &C_nthreads,
        // computed by phase0:
        Cnvec, Ch, C_to_M, C_to_A, C_to_B, false,
        // original input:
        (apply_mask) ? M : NULL, A, B, Werk)) ;

    // count the number of entries in each vector of C
    GB_OK (GB_emult_08_phase1 (
        // computed by phase1:
        &Cp, &Cp_size, &Cnvec_nonempty,
        // from phase1a:
        TaskList, C_ntasks, C_nthreads,
        // from phase0:
        Cnvec, Ch, C_to_M, C_to_A, C_to_B,
        // original input:
        (apply_mask) ? M : NULL, Mask_struct, Mask_comp, A, B, Werk)) ;

    //--------------------------------------------------------------------------
    // phase2: compute the entries (indices and values) in each vector of C
    //--------------------------------------------------------------------------

    // Cp is either freed by phase2, or transplanted into C.
    // Either way, it is not freed here.

    GB_OK (GB_emult_08_phase2 (
        // computed or used by phase2:
        C, ctype, C_is_csc, op,
        // from phase1:
        &Cp, Cp_size, Cnvec_nonempty,
        // from phase1a:
        TaskList, C_ntasks, C_nthreads,
        // from phase0:
        Cnvec, Ch, Ch_size, C_to_M, C_to_A, C_to_B, C_sparsity,
        // from GB_emult_sparsity:
        ewise_method,
        // original input:
        (apply_mask) ? M : NULL, Mask_struct, Mask_comp, A, B, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (C, "C output for emult", GB0) ;
    (*mask_applied) = apply_mask ;
    return (GrB_SUCCESS) ;
}

