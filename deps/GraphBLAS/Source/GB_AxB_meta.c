//------------------------------------------------------------------------------
// GB_AxB_meta: C<M>=A*B meta algorithm
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C, C<M>, C<!M> = A*B, A'*B, A*B', or A'*B' : both symbolic and numeric, with
// the optional mask matrix.  This function is called by GB_mxm only.  If the
// mask matrix is present, it can be regular or complemented, and either valued
// or structural.

// This algorithm may decide that it is more efficient to apply the mask later,
// in GB_accum_mask, after this matrix C is computed, in GB_mxm.  The result is
// either the T matrix in GB_mxm, or (if done in-place), the final output
// matrix C passed in from the user (C_in).

// The method is chosen automatically:  a gather/scatter saxpy method
// (Gustavson), or a dot product method.

// FUTURE:: an outer-product method for C=A*B'

#define GB_FREE_WORKSPACE       \
{                               \
    GB_Matrix_free (&AT) ;      \
    GB_Matrix_free (&BT) ;      \
}

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORKSPACE ;         \
    GB_phybix_free (C) ;        \
    GB_phybix_free (MT) ;       \
}

#include "GB_mxm.h"
#include "GB_transpose.h"

GrB_Info GB_AxB_meta                // C<M>=A*B meta algorithm
(
    GrB_Matrix C,                   // output, static header (if not in-place)
    GrB_Matrix C_in,                // input/output matrix, if done in-place
    bool C_replace,                 // C matrix descriptor
    const bool C_is_csc,            // desired CSR/CSC format of C
    GrB_Matrix MT,                  // return MT = M' (static header)
    bool *M_transposed,             // true if MT = M' was computed
    const GrB_Matrix M_in,          // mask for C<M> (not complemented)
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // accum operator for C_in += A*B
    const GrB_Matrix A_in,          // input matrix
    const GrB_Matrix B_in,          // input matrix
    const GrB_Semiring semiring_in, // semiring that defines C=A*B
    bool A_transpose,               // if true, use A', else A
    bool B_transpose,               // if true, use B', else B
    bool flipxy,                    // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // if true, mask was applied
    bool *done_in_place,            // if true, C was computed in-place
    GrB_Desc_Value AxB_method,      // for auto vs user selection of methods
    const int do_sort,              // if nonzero, try to return C unjumbled
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK_OR_NULL (C_in, "C_in for meta A*B", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M_in, "M for meta A*B", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for meta A*B", GB0) ;
    ASSERT_MATRIX_OK (A_in, "A_in for meta A*B", GB0) ;
    ASSERT_MATRIX_OK (B_in, "B_in for meta A*B", GB0) ;

    ASSERT (!GB_ZOMBIES (M_in)) ;
    ASSERT (GB_JUMBLED_OK (M_in)) ;
    ASSERT (!GB_PENDING (M_in)) ;

    ASSERT (!GB_ZOMBIES (A_in)) ;
    ASSERT (GB_JUMBLED_OK (A_in)) ;
    ASSERT (!GB_PENDING (A_in)) ;

    ASSERT (!GB_ZOMBIES (B_in)) ;
    ASSERT (GB_JUMBLED_OK (B_in)) ;
    ASSERT (!GB_PENDING (B_in)) ;

    ASSERT_SEMIRING_OK (semiring_in, "semiring_in for numeric A*B", GB0) ;
    ASSERT (mask_applied != NULL) ;
    ASSERT (C  != NULL && ( C->static_header || GBNSTATIC)) ;
    ASSERT (MT != NULL && (MT->static_header || GBNSTATIC)) ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    GrB_Info info ;
    struct GB_Matrix_opaque AT_header, BT_header ;
    GrB_Matrix AT = NULL, BT = NULL ;

    (*mask_applied) = false ;
    (*done_in_place) = false ;

    //--------------------------------------------------------------------------
    // get the semiring_in
    //--------------------------------------------------------------------------

    GB_Opcode opcode = semiring_in->multiply->opcode  ;
    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
    bool op_is_first  = (opcode == GB_FIRST_binop_code) ;
    bool op_is_second = (opcode == GB_SECOND_binop_code) ;
    bool op_is_pair   = (opcode == GB_PAIR_binop_code) ;
    bool allow_scale = true ;
    if (semiring_in->multiply->binop_function == NULL &&
        (op_is_first || op_is_second))
    { 
        // GB_rowscale and GB_colscale do not handle the implicit FIRST
        // operator for GB_reduce_to_vector.  They do handle any other
        // positional operator (FIRSTI, FIRSTJ, SECONDI, SECONDJ, etc).
        allow_scale = false ;
    }

    //--------------------------------------------------------------------------
    // estimate the work to transpose A, B, and C
    //--------------------------------------------------------------------------

    double A_work = GB_nnz_held (A_in) ;    // work to transpose A
    double B_work = GB_nnz_held (B_in) ;    // work to transpose B
    // work to transpose C cannot be determined; assume it is full
    double C_work =
        (double) (A_transpose ? GB_NCOLS (A_in) : GB_NROWS (A_in)) *
        (double) (B_transpose ? GB_NROWS (B_in) : GB_NCOLS (B_in)) ;

    //--------------------------------------------------------------------------
    // see if the work can be done in-place
    //--------------------------------------------------------------------------

    // If C is hypersparse, sparse, or full:
    //
    //      C can be computed in-place if it is already dense, and if it is
    //      guaranteed to remain dense after the computation is done.  This
    //      case requires the accum operator to be present and it must match
    //      the monoid of the semiring.  C_replace must be false, or
    //      effectively false.
    //
    //      todo:  if C is full and accum is not present, it can be quickly
    //      converted to bitmap and then done in-place.
    //
    // If C is bitmap:
    //
    //      C can be computed in-place if its type is the same as the semiring
    //      monoid.  The accum must not be present, or if present it must match
    //      the semiring monoid.  C_replace can be true or false.
    //
    //      todo: modify GB_AxB_dot2 so it can compute C in-place,
    //      or add a bitmap dot product method.  Also modify GB_AxB_saxpy
    //      so it can compute a C bitmap in-place.
    //
    // In both cases, C must not be transposed, nor can it be aliased with any
    // input matrix.

    bool can_do_in_place = false ;

    if (C_in != NULL)
    {
        #if 0
        // disabled: this will work for most methods in the future is too
        // aggressive for dot4
        if (GB_IS_BITMAP (C_in))
        {
            // C is bitmap
            ASSERT (!GB_PENDING (C_in)) ; // no pending tuples in bitmap
            ASSERT (!GB_ZOMBIES (C_in)) ; // bitmap never has zombies
            can_do_in_place = (C_in->type == semiring_in->add->op->ztype)
                && ((accum == NULL) || (accum == semiring_in->add->op)) ;
        }
        else
        #endif
        if (accum != NULL)
        { 
            // accum is present; check if C_in is full.
            bool C_is_full = GB_IS_FULL (C_in) ;

            // accum must be present, and must match the monoid of the
            // semiring, and the ztype of the monoid must match the type of C
            bool accum_is_monoid = (accum == semiring_in->add->op)
                && (C_in->type == accum->ztype) ;

            // C += A*B with C_replace ignored (effectively false)
            // C<M> += A*B with C_replace false
            // C<!M> += A*B with C_replace false
            can_do_in_place = C_is_full && accum_is_monoid
                && ((M_in == NULL) || (M_in != NULL && !C_replace)) ;
        }

        // C must also not be transposed on output; see below.  Nor can it be
        // aliased with any input matrix.  This test is done after handling the
        // CSR/CSC formats since the input matrices may be transposed (thus
        // breaking the alias with C).
    }

    //--------------------------------------------------------------------------
    // handle the CSR/CSC formats of C, M, A, and B
    //--------------------------------------------------------------------------

    // On input, A and/or B can be transposed, and all four matrices can be in
    // either CSR or CSC format, in any combination.  This gives a total of 64
    // possible combinations.  However, a CSR matrix that is transposed is just
    // the same as a non-transposed CSC matrix.

    // Use transpose to handle the CSR/CSC format.  If C is desired in CSR
    // format, treat it as if it were in format CSC but transposed.
    bool C_transpose = !C_is_csc ;

    // If the mask is not present, then treat it as having the same CSR/CSC
    // format as C.
    bool M_is_csc = (M_in == NULL) ? C_is_csc : M_in->is_csc ;

    // Treat M just like C.  If M is in CSR format, treat it as if it were CSC
    // but transposed, since there are no descriptors that transpose C or M.
    bool M_transpose = !M_is_csc ;

    // A can be transposed, and can also be in CSR or CSC format.  If A is in
    // CSR, treat it as A' in CSC, and if A' is in CSR, treat it as A in CSC.
    if (!A_in->is_csc)
    { 
        // Flip the sense of A_transpose
        A_transpose = !A_transpose ;
    }

    // B is treated just like A
    if (!B_in->is_csc)
    { 
        // Flip the sense of B_transpose
        B_transpose = !B_transpose ;
    }

        // Now all matrices C, M_in, A_in, and B_in, can be treated as if they
        // were all in CSC format, except any of them can be transposed.  There
        // are now 16 cases to handle, where M, A, and B are M_in, A_in, and
        // B_in and all matrices are CSR/CSC agnostic, and where C has not yet
        // been created.

        //      C <M > = A  * B
        //      C <M'> = A  * B
        //      C'<M > = A  * B
        //      C'<M'> = A  * B

        //      C <M > = A  * B'
        //      C <M'> = A  * B'
        //      C'<M > = A  * B'
        //      C'<M'> = A  * B'

        //      C <M > = A' * B
        //      C <M'> = A' * B
        //      C'<M > = A' * B
        //      C'<M'> = A' * B

        //      C <M > = A' * B'
        //      C <M'> = A' * B'
        //      C'<M > = A' * B'
        //      C'<M'> = A' * B'

    //==========================================================================
    // swap_rule: decide if C or C' should be computed
    //==========================================================================

    // This function can compute C or C', by setting C->is_csc as the negation
    // of the desired format C_is_csc.  This ensures that GB_accum_mask will
    // transpose C when this function is done.

    bool swap_rule = false ;
    int A_in_is_diagonal = -1 ;            // not yet computed
    int B_in_is_diagonal = -1 ;            // not yet computed

    if (C_transpose && A_transpose && B_transpose)
    { 

        //----------------------------------------------------------------------
        // C' = A'*B'       becomes C = B*A, never stays as-is
        //----------------------------------------------------------------------

        swap_rule = true ;

    }
    else if (!C_transpose && A_transpose && B_transpose)
    { 

        //----------------------------------------------------------------------
        // C  = A'*B'       becomes C = (B*A)', never stays as-is
        //----------------------------------------------------------------------

        swap_rule = true ;

    }
    else if (C_transpose && A_transpose && !B_transpose)
    { 

        //----------------------------------------------------------------------
        // C' = A'*B        becomes C = B'*A via swap rule, or stays C'=A'*B
        //----------------------------------------------------------------------

        // by default, use the swap rule and compute C=B'*A instead
        swap_rule = true ;

        // In v6.1.1 and earlier, this method always chose swap_rule = true.
        // The heuristic has been modified in v6.1.2 by adding the following
        // refinement, possibly selecting swap_rule as false instead:

        // see what the swap_rule == true would do for C=B'*A
        A_in_is_diagonal = GB_is_diagonal (A_in) ;

        int tentative_axb_method ;
        GB_AxB_meta_adotb_control (&tentative_axb_method, C_in, M_in,
            Mask_comp, B_in, A_in, accum, semiring_in, flipxy, can_do_in_place,
            allow_scale, A_in_is_diagonal, AxB_method) ;

        if (tentative_axb_method == GB_USE_SAXPY)
        { 
            // reconsider and do not use swap rule if saxpy C=B'*A is too
            // expensive.  C'=A'*B is either computed as-is with swap_rule
            // false, requiring a transpose of C and A.  Or, C=B'*A is
            // computed, with swap_rule true, requiring a transpose of B.
            swap_rule = (B_work < A_work + C_work) ;
        }

    }
    else if (C_transpose && !A_transpose && B_transpose)
    { 

        //----------------------------------------------------------------------
        // C' = A*B'        becomes C = B*A', or stays as C' = A*B'
        //----------------------------------------------------------------------

        // C'=A*B' is either computed as-is with C'=A*B', or C=B*A' with
        // swap_rule true.  Both require explicit transpose(s).
        // C'=A*B' requires B to be transposed, then C on output.
        // C=B*A' requires A to be transposed.

        // In v5.1 and earlier swap_rule = true was used for this case.
        // If C is very large, this will still be true.  swap_rule can only be
        // false if C is small.

        swap_rule = (A_work < B_work + C_work) ;

    }
    else if (!C_transpose && !A_transpose && B_transpose)
    { 

        //----------------------------------------------------------------------
        // C  = A*B'        becomes C'=B*A' or stays C=A*B'
        //----------------------------------------------------------------------

        // C=A*B' is either computed as-is with C=A*B', or C'=B*A' with
        // swap_rule true.  Both require explicit transpose(s).
        // C'=B*A' requires A to be transposed, then C on output.
        // C=A*B' requires B to be transposed.

        // In v5.1 and earlier swap_rule = false was used for this case.
        // If C is very large, this will still be false.  swap_rule can only be
        // true if C is small.

        swap_rule = (B_work > A_work + C_work) ;

    }
    else if (!C_transpose && A_transpose && !B_transpose)
    { 

        //----------------------------------------------------------------------
        // C  = A'*B        becomes C'=B'*A or stays C=A'*B
        //----------------------------------------------------------------------

        // by default, do not use the swap rule and compute C=A'*B as-is
        swap_rule = false ;

        // In v6.1.1 and earlier, this method always chose swap_rule = false.
        // The heuristic has been modified in v6.1.2 by adding the following
        // refinement, possibly selecting swap_rule as true instead:

        // see what method C=A'*B would use if swap_rule is false
        B_in_is_diagonal = GB_is_diagonal (B_in) ;

        int tentative_axb_method ;
        GB_AxB_meta_adotb_control (&tentative_axb_method, C_in, M_in,
            Mask_comp, A_in, B_in, accum, semiring_in, flipxy, can_do_in_place,
            allow_scale, B_in_is_diagonal, AxB_method) ;

        if (tentative_axb_method == GB_USE_SAXPY)
        { 
            // reconsider and use swap rule if saxpy C=(A')*B is too expensive.
            // C=(A')*B is either computed as-is, requiring a transpose of A,
            // or it is computed as C'=(B')*A using the swap rule, requiring a
            // transpose of C and B.
            swap_rule = (A_work > B_work + C_work) ;
        }

    }
    else if (C_transpose && !A_transpose && !B_transpose)
    { 

        //----------------------------------------------------------------------
        // C' = A*B         stays as-is
        //----------------------------------------------------------------------

        swap_rule = false ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // C  = A*B         stays as-is
        //----------------------------------------------------------------------

        swap_rule = false ;
    }

    //--------------------------------------------------------------------------
    // apply the swap_rule
    //--------------------------------------------------------------------------

    GrB_Matrix A, B ;
    bool atrans, btrans ;
    int A_is_diagonal = -1 ;            // not yet computed
    int B_is_diagonal = -1 ;            // not yet computed

    if (swap_rule)
    { 
        // Replace C'=(A'*B') with C=B*A, and so on.  Swap A and B and transose
        // them, transpose M, negate flipxy, and transpose M and C.
        A = B_in ; atrans = !B_transpose ;
        B = A_in ; btrans = !A_transpose ;
        flipxy = !flipxy ;              // flipxy is modified here
        M_transpose = !M_transpose ;
        C_transpose = !C_transpose ;
        A_is_diagonal = B_in_is_diagonal ;
        B_is_diagonal = -1 ;
    }
    else
    { 
        // use the input matrices as-is
        A = A_in ; atrans = A_transpose ;
        B = B_in ; btrans = B_transpose ;
        A_is_diagonal = -1 ;
        B_is_diagonal = B_in_is_diagonal ;
    }

    ASSERT_MATRIX_OK (A, "final A for A*B", GB0) ;
    ASSERT_MATRIX_OK (B, "final B for A*B", GB0) ;

    //--------------------------------------------------------------------------
    // finalize the semiring after flipping the binary multiplicative operator
    //--------------------------------------------------------------------------

    struct GB_Semiring_opaque semiring_struct ;
    GrB_Semiring semiring = &semiring_struct ;
    memcpy (semiring, semiring_in, sizeof (struct GB_Semiring_opaque)) ;
    semiring->header_size = 0 ;
    semiring->multiply = GB_flip_binop (semiring_in->multiply, &flipxy) ;

    opcode = semiring->multiply->opcode  ;
    op_is_first  = (opcode == GB_FIRST_binop_code) ;
    op_is_second = (opcode == GB_SECOND_binop_code) ;

    // flipxy is now false for all built-in semirings, and for all user-defined
    // semirings that use built-in multiplicative operators that are handled by
    // GB_flip_binop.

    //--------------------------------------------------------------------------
    // explicitly transpose the mask
    //--------------------------------------------------------------------------

    GrB_Matrix M ;

    if (M_transpose && M_in != NULL)
    { 
        // MT = M_in' also typecasting to boolean.  It is not freed here
        // unless an error occurs, but is returned to the caller.
        // If Mask_struct is true, MT = one(M') is iso.
        GBURBLE ("(M transpose) ") ;
        GB_OK (GB_transpose_cast (MT, GrB_BOOL, C_is_csc, M_in, Mask_struct,
            Werk)) ;
        M = MT ;
        (*M_transposed) = true ;
    }
    else
    { 
        // M_in can be used as-is; it may be NULL
        M = M_in ;
        (*M_transposed) = false ;
    }

    ASSERT_MATRIX_OK_OR_NULL (M, "final M for A*B", GB0) ;

    //--------------------------------------------------------------------------
    // check additional conditions for in-place computation of C
    //--------------------------------------------------------------------------

    if (can_do_in_place)
    {
        // C cannot be done in-place if it is aliased with any input matrix.
        // Also cannot compute C in-place if it is to be transposed.
        bool C_aliased = GB_any_aliased (C_in, M) || GB_any_aliased (C_in, A) ||
            GB_any_aliased (C_in, B) ;
        if (C_transpose || C_aliased)
        { 
            can_do_in_place = false ;
        }
        // todo: A and B can be transposed below, so this check should be
        // done after any such transposings.
    }

    //--------------------------------------------------------------------------
    // burble
    //--------------------------------------------------------------------------

    const char *M_str = (M == NULL) ? "" : (Mask_comp ?  "<!M>" : "<M>") ;
    #define GB_PROP_LEN (GxB_MAX_NAME_LEN+128)
    char A_str [GB_PROP_LEN+1] ;
    char B_str [GB_PROP_LEN+1] ;
    if (GB_Global_burble_get ( ))
    {
        int64_t anz = GB_nnz (A) ;
        int64_t bnz = GB_nnz (B) ;
        snprintf (A_str, GB_PROP_LEN, "A: " GBd "-by-" GBd ", %s, " GBd
            " entries", GB_NROWS (A), GB_NCOLS (A), A->type->name, anz) ;
        snprintf (B_str, GB_PROP_LEN, "B: " GBd "-by-" GBd ", %s, " GBd
            " entries", GB_NROWS (B), GB_NCOLS (B), B->type->name, bnz) ;
    }

    //--------------------------------------------------------------------------
    // typecast A and B when transposing them, if needed
    //--------------------------------------------------------------------------

    bool A_is_pattern ;
    bool B_is_pattern ;
    GrB_Type atype_cast, btype_cast ;
    if (flipxy)
    { 
        // A is passed as y, and B as x, in z = mult(x,y)
        // The built-in first, second, pair, and positional ops have all been
        // renamed, so A and B are not pattern-only if flipxy is still true.
        A_is_pattern = false ;
        B_is_pattern = false ;
        atype_cast = semiring->multiply->ytype ;
        btype_cast = semiring->multiply->xtype ;
    }
    else
    { 
        // A is passed as x, and B as y, in z = mult(x,y)
        A_is_pattern = op_is_second || op_is_pair || op_is_positional ;
        B_is_pattern = op_is_first  || op_is_pair || op_is_positional ;
        atype_cast = semiring->multiply->xtype ;
        btype_cast = semiring->multiply->ytype ;
    }

    //==========================================================================
    // select the final algorithm and perform the matrix multiply
    //==========================================================================

    // use GB_AxB_saxpy3 by default
    int axb_method = GB_USE_SAXPY ;

    if (atrans)
    {

        //----------------------------------------------------------------------
        // C<M> = A'*B' or A'*B
        //----------------------------------------------------------------------

        if (B_is_diagonal == -1)
        {
            B_is_diagonal = GB_is_diagonal (B) ;
        }

        // explicitly transpose B
        if (btrans && !B_is_diagonal)
        {
            // B = B', or B = one(B') if only the pattern of B is needed.
            // This is currently unused, since C=A'*B' and C'=A'*B' are always
            // converted to C=(B*A)' and C=B*A, respectively.  It is left here
            // in case the swap_rule changes.
            GB_CLEAR_STATIC_HEADER (BT, &BT_header) ;
            GB_OK (GB_transpose_cast (BT, btype_cast, true, B, B_is_pattern,
                Werk)) ;
            B = BT ;
        }

        //----------------------------------------------------------------------
        // select the method for C<M>=A'*B
        //----------------------------------------------------------------------

        GB_AxB_meta_adotb_control (&axb_method, C_in, M,
            Mask_comp, A, B, accum, semiring, flipxy, can_do_in_place,
            allow_scale, B_is_diagonal, AxB_method) ;

        //----------------------------------------------------------------------
        // AT = A'
        //----------------------------------------------------------------------

        if (axb_method == GB_USE_COLSCALE || axb_method == GB_USE_SAXPY)
        {
            // AT = A', or AT=one(A') if only the pattern is needed.
            GB_CLEAR_STATIC_HEADER (AT, &AT_header) ;
            GB_OK (GB_transpose_cast (AT, atype_cast, true, A, A_is_pattern,
                Werk)) ;
            // do not use colscale if AT is now bitmap
            if (GB_IS_BITMAP (AT))
            { 
                axb_method = GB_USE_SAXPY ;
            }
        }

        //----------------------------------------------------------------------
        // C<M>=A'*B
        //----------------------------------------------------------------------

        switch (axb_method)
        {
            case GB_USE_ROWSCALE : 
                // C = D*B using rowscale
                GBURBLE ("C%s=A'*B, rowscale ", M_str) ;
                GB_OK (GB_rowscale (C, A, B, semiring, flipxy, Werk)) ;
                break ;

            case GB_USE_COLSCALE : 
                // C = A'*D using colscale
                GBURBLE ("C%s=A'*B, colscale (transposed %s) ", M_str, A_str) ;
                GB_OK (GB_colscale (C, AT, B, semiring, flipxy, Werk)) ;
                break ;

            case GB_USE_DOT : 
                // C<M>=A'*B via dot, or C_in<M>+=A'*B if in-place
                GBURBLE ("C%s=A'*B, %sdot_product ", M_str,
                    (M != NULL && !Mask_comp) ? "masked_" : "") ;
                GB_OK (GB_AxB_dot (C, can_do_in_place ? C_in : NULL, M,
                    Mask_comp, Mask_struct, accum, A, B, semiring, flipxy,
                    mask_applied, done_in_place, Werk)) ;
                break ;

            default : 
                // C = A'*B via saxpy: Gustavson + Hash method
                GBURBLE ("C%s=A'*B, saxpy (transposed %s) ", M_str, A_str) ;
                GB_OK (GB_AxB_saxpy (C, can_do_in_place ? C_in : NULL, M,
                    Mask_comp, Mask_struct, accum, AT, B, semiring, flipxy,
                    mask_applied, done_in_place, AxB_method, do_sort, Werk)) ;
                break ;
        }

    }
    else if (btrans)
    {

        //----------------------------------------------------------------------
        // select the method for C<M> = A*B'
        //----------------------------------------------------------------------

        if (allow_scale && M == NULL
            && !GB_IS_BITMAP (A)     // todo: A*D colscale with A bitmap
            && ((B_is_diagonal == -1) ? GB_is_diagonal (B) : B_is_diagonal))
        { 
            // C = A*D, column scale
            axb_method = GB_USE_COLSCALE ;
        }
        else if (allow_scale && M == NULL
            && !GB_IS_BITMAP (B)     // todo: D*B' rowscale with B bitmap
            && ((A_is_diagonal == -1) ? GB_is_diagonal (A) : A_is_diagonal))
        { 
            // C = D*B', row scale
            axb_method = GB_USE_ROWSCALE ;
        }
        else if (AxB_method == GxB_AxB_DOT)
        { 
            // only use the dot product method if explicitly requested
            axb_method = GB_USE_DOT ;
        }

        //----------------------------------------------------------------------
        // BT = B'
        //----------------------------------------------------------------------

        if (axb_method != GB_USE_COLSCALE)
        {
            // BT = B', or BT=one(B') if only the pattern of B is needed
            GB_CLEAR_STATIC_HEADER (BT, &BT_header) ;
            GB_OK (GB_transpose_cast (BT, btype_cast, true, B, B_is_pattern,
                Werk)) ;
            // do not use rowscale if BT is now bitmap
            if (axb_method == GB_USE_ROWSCALE && GB_IS_BITMAP (BT))
            { 
                axb_method = GB_USE_SAXPY ;
            }
        }

        //----------------------------------------------------------------------
        // C<M> = A*B'
        //----------------------------------------------------------------------

        switch (axb_method)
        {
            case GB_USE_COLSCALE : 
                // C = A*D
                GBURBLE ("C%s=A*B', colscale ", M_str) ;
                GB_OK (GB_colscale (C, A, B, semiring, flipxy, Werk)) ;
                break ;

            case GB_USE_ROWSCALE : 
                // C = D*B'
                GBURBLE ("C%s=A*B', rowscale (transposed %s) ", M_str, B_str) ;
                GB_OK (GB_rowscale (C, A, BT, semiring, flipxy, Werk)) ;
                break ;

            case GB_USE_DOT : 
                // C<M>=A*B' via dot product, or C_in<M>+=A*B' if in-place
                GBURBLE ("C%s=A*B', dot_product (transposed %s) "
                    "(transposed %s) ", M_str, A_str, B_str) ;
                GB_CLEAR_STATIC_HEADER (AT, &AT_header) ;
                GB_OK (GB_transpose_cast (AT, atype_cast, true, A, A_is_pattern,
                    Werk)) ;
                GB_OK (GB_AxB_dot (C, can_do_in_place ? C_in : NULL, M,
                    Mask_comp, Mask_struct, accum, AT, BT, semiring, flipxy,
                    mask_applied, done_in_place, Werk)) ;
                break ;

            default : 
                // C = A*B' via saxpy: Gustavson + Hash method
                GBURBLE ("C%s=A*B', saxpy (transposed %s) ", M_str, B_str) ;
                GB_OK (GB_AxB_saxpy (C, can_do_in_place ? C_in : NULL, M,
                    Mask_comp, Mask_struct, accum, A, BT, semiring, flipxy,
                    mask_applied, done_in_place, AxB_method, do_sort, Werk)) ;
                break ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // select the method for C<M> = A*B
        //----------------------------------------------------------------------

        if (allow_scale && M == NULL
            && !GB_IS_BITMAP (A)     // todo: A*D colscale with A bitmap
            && ((B_is_diagonal == -1) ? GB_is_diagonal (B) : B_is_diagonal))
        { 
            // C = A*D, column scale
            axb_method = GB_USE_COLSCALE ;
        }
        else if (allow_scale && M == NULL
            && !GB_IS_BITMAP (B)     // todo: D*B rowscale with B bitmap
            && ((A_is_diagonal == -1) ? GB_is_diagonal (A) : A_is_diagonal))
        { 
            // C = D*B, row scale
            axb_method = GB_USE_ROWSCALE ;
        }
        else if (AxB_method == GxB_AxB_DOT)
        { 
            // C<M>=A*B via dot product, or C_in<M>+=A*B if in-place.
            axb_method = GB_USE_DOT ;
        }
        else if (AxB_method == GxB_AxB_SAXPY
              || AxB_method == GxB_AxB_HASH
              || AxB_method == GxB_AxB_GUSTAVSON)
        { 
            // C<M>=A*B via saxpy
            axb_method = GB_USE_SAXPY ;
        }
        else
        {
            // C = A*B: auto selection: select saxpy or dot
            if (GB_IS_HYPERSPARSE (A) && (GB_IS_BITMAP (B) || GB_IS_FULL (B)))
            {
                // If A is hyper and B is bitmap/full, then saxpy will compute
                // C as sparse or bitmap.  If bitmap, use saxpy; if sparse, use
                // dot product instead.
                int ignore, saxpy_method ;
                GB_AxB_saxpy_sparsity (&ignore, &saxpy_method, M, Mask_comp,
                    A, B) ;
                if (saxpy_method == GB_SAXPY_METHOD_BITMAP)
                { 
                    // bitmap = hyper * (bitmap or full) is very efficient
                    // to do via GB_AxB_saxbit.
                    axb_method = GB_USE_SAXPY ;
                }
                else
                { 
                    // sparse = hyper * (bitmap or full) would use
                    // GB_AxB_saxpy3, which can be slow, so use dot instead.
                    axb_method = GB_USE_DOT ;
                }
            }
            else
            { 
                // otherwise, always use GB_AxB_saxpy
                axb_method = GB_USE_SAXPY ;
            }
        }

        //----------------------------------------------------------------------
        // C<M> = A*B
        //----------------------------------------------------------------------

        switch (axb_method)
        {
            case GB_USE_COLSCALE : 
                // C = A*D, column scale
                GBURBLE ("C%s=A*B, colscale ", M_str) ;
                GB_OK (GB_colscale (C, A, B, semiring, flipxy, Werk)) ;
                break ;

            case GB_USE_ROWSCALE : 
                // C = D*B, row scale
                GBURBLE ("C%s=A*B, rowscale ", M_str) ;
                GB_OK (GB_rowscale (C, A, B, semiring, flipxy, Werk)) ;
                break ;

            case GB_USE_DOT : 
                // C<M>=A*B via dot product, or C_in<M>+=A*B if in-place.
                GBURBLE ("C%s=A*B', dot_product (transposed %s) ",
                    M_str, A_str) ;
                GB_CLEAR_STATIC_HEADER (AT, &AT_header) ;
                GB_OK (GB_transpose_cast (AT, atype_cast, true, A, A_is_pattern,
                    Werk)) ;
                GB_OK (GB_AxB_dot (C, can_do_in_place ? C_in : NULL, M,
                    Mask_comp, Mask_struct, accum, AT, B, semiring, flipxy,
                    mask_applied, done_in_place, Werk)) ;
                break ;

            default : 
                // C = A*B via saxpy: Gustavson + Hash method
                GBURBLE ("C%s=A*B, saxpy ", M_str) ;
                GB_OK (GB_AxB_saxpy (C, can_do_in_place ? C_in : NULL, M,
                    Mask_comp, Mask_struct, accum, A, B, semiring, flipxy,
                    mask_applied, done_in_place, AxB_method, do_sort, Werk)) ;
                break ;
        }
    }

    if (*M_transposed) { GBURBLE ("(M transposed) ") ; }
    if ((M != NULL) && !(*mask_applied)) { GBURBLE ("(mask later) ") ; }

    //--------------------------------------------------------------------------
    // handle C_transpose and assign the CSR/CSC format
    //--------------------------------------------------------------------------

    // If C_transpose is true, then C' has been computed.  In this case, negate
    // the desired C_is_csc so that GB_accum_mask transposes the result before
    // applying the accum operator and/or writing the result back to the user's
    // C.

    if (*done_in_place)
    { 
        GBURBLE ("(C in place) ") ;
        // C can be done in-place only if C is not transposed on output
        ASSERT_MATRIX_OK (C_in, "C_in output for all C=A*B", GB0) ;
        ASSERT (C_in->is_csc == C_is_csc) ;
    }
    else
    { 
        C->is_csc = C_transpose ? !C_is_csc : C_is_csc ;
        ASSERT_MATRIX_OK (C, "C output for all C=A*B", GB0) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    // do not free MT; return it to the caller
    #ifdef GB_DEBUG
    if (*M_transposed) ASSERT_MATRIX_OK (MT, "MT computed", GB0) ;
    #endif
    return (GrB_SUCCESS) ;
}

