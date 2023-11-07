//------------------------------------------------------------------------------
// GB_ewise: C<M> = accum (C, A+B) or A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C,A+B), A.*B and variations.  The input matrices A and B are
// optionally transposed.  Does the work for GrB_eWiseAdd_* and
// GrB_eWiseMult_*.  Handles all cases of the mask.

#define GB_FREE_ALL         \
{                           \
    GB_Matrix_free (&T) ;   \
    GB_Matrix_free (&AT) ;  \
    GB_Matrix_free (&BT) ;  \
    GB_Matrix_free (&MT) ;  \
}

#include "GB_ewise.h"
#include "GB_add.h"
#include "GB_emult.h"
#include "GB_transpose.h"
#include "GB_accum_mask.h"
#include "GB_binop.h"

GrB_Info GB_ewise                   // C<M> = accum (C, A+B) or A.*B
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // if true, clear C before writing to it
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, complement the mask M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op_in,       // defines '+' for C=A+B, or .* for A.*B
    const GrB_Matrix A,             // input matrix
    bool A_transpose,               // if true, use A' instead of A
    const GrB_Matrix B,             // input matrix
    bool B_transpose,               // if true, use B' instead of B
    bool eWiseAdd,                  // if true, do set union (like A+B),
                                    // otherwise do intersection (like A.*B)
    const bool is_eWiseUnion,       // if true, eWiseUnion, else eWiseAdd
    const GrB_Scalar alpha,         // alpha and beta ignored for eWiseAdd,
    const GrB_Scalar beta,          // nonempty scalars for GxB_eWiseUnion
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M, A, and/or B

    GrB_Info info ;
    GrB_Matrix MT = NULL, T = NULL, AT = NULL, BT = NULL ;
    struct GB_Matrix_opaque T_header, MT_header, AT_header, BT_header ;

    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;

    ASSERT_MATRIX_OK (C, "C input for GB_ewise", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_ewise", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_ewise", GB0) ;
    ASSERT_BINARYOP_OK (op_in, "op for GB_ewise", GB0) ;
    ASSERT_MATRIX_OK (A, "A for GB_ewise", GB0) ;
    ASSERT_MATRIX_OK (B, "B for GB_ewise", GB0) ;

    // T has the same type as the output z for z=op(a,b)
    GrB_BinaryOp op = op_in ;
    GrB_Type T_type = op->ztype ;

    // check domains and dimensions for C<M> = accum (C,T)
    GB_OK (GB_compatible (C->type, C, M, Mask_struct, accum, T_type, Werk)) ;

    // T=op(A,B) via op operator, so A and B must be compatible with z=op(a,b)
    GB_OK (GB_BinaryOp_compatible (op, NULL, A->type, B->type,
        GB_ignore_code, Werk)) ;

    if (eWiseAdd)
    {
        if (is_eWiseUnion)
        {
            // alpha and beta scalars must be present
            GB_RETURN_IF_NULL_OR_FAULTY (alpha) ;
            GB_RETURN_IF_NULL_OR_FAULTY (beta) ;
            GB_MATRIX_WAIT (alpha) ;
            GB_MATRIX_WAIT (beta) ;
            if (GB_nnz ((GrB_Matrix) alpha) == 0)
            { 
                GB_ERROR (GrB_EMPTY_OBJECT, "%s\n",
                    "alpha cannot be an empty scalar") ;
            }
            if (GB_nnz ((GrB_Matrix) beta) == 0)
            { 
                GB_ERROR (GrB_EMPTY_OBJECT, "%s\n",
                    "beta cannot be an empty scalar") ;
            }
            // C = op (A, beta) is done for entries in A but not B
            if (!GB_Type_compatible (op->ytype, beta->type))
            { 
                GB_ERROR (GrB_DOMAIN_MISMATCH,
                    "beta scalar of type [%s]\n"
                    "cannot be typecast to op input of type [%s]",
                    beta->type->name, op->ytype->name) ;
            }
            // C = op (alpha, B) is done for entries in B but not A
            if (!GB_Type_compatible (op->xtype, alpha->type))
            { 
                GB_ERROR (GrB_DOMAIN_MISMATCH,
                    "alpha scalar of type [%s]\n"
                    "cannot be typecast to op input of type [%s]",
                    alpha->type->name, op->xtype->name) ;
            }
        }
        else
        {
            // C = A is done for entries in A but not B
            if (!GB_Type_compatible (C->type, A->type))
            { 
                GB_ERROR (GrB_DOMAIN_MISMATCH,
                    "First input of type [%s]\n"
                    "cannot be typecast to final output of type [%s]",
                    A->type->name, C->type->name) ;
            }
            // C = B is done for entries in B but not A
            if (!GB_Type_compatible (C->type, B->type))
            { 
                GB_ERROR (GrB_DOMAIN_MISMATCH,
                    "Second input of type [%s]\n"
                    "cannot be typecast to final output of type [%s]",
                    B->type->name, C->type->name) ;
            }
        }
    }

    // check the dimensions
    int64_t anrows = (A_transpose) ? GB_NCOLS (A) : GB_NROWS (A) ;
    int64_t ancols = (A_transpose) ? GB_NROWS (A) : GB_NCOLS (A) ;
    int64_t bnrows = (B_transpose) ? GB_NCOLS (B) : GB_NROWS (B) ;
    int64_t bncols = (B_transpose) ? GB_NROWS (B) : GB_NCOLS (B) ;
    int64_t cnrows = GB_NROWS (C) ;
    int64_t cncols = GB_NCOLS (C) ;
    if (anrows != bnrows || ancols != bncols ||
        cnrows != anrows || cncols != bncols)
    { 
        GB_ERROR (GrB_DIMENSION_MISMATCH,
            "Dimensions not compatible:\n"
            "output is " GBd "-by-" GBd "\n"
            "first input is " GBd "-by-" GBd "%s\n"
            "second input is " GBd "-by-" GBd "%s",
            cnrows, cncols,
            anrows, ancols, A_transpose ? " (transposed)" : "",
            bnrows, bncols, B_transpose ? " (transposed)" : "") ;
    }

    // quick return if an empty mask M is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp, Mask_struct) ;

    //--------------------------------------------------------------------------
    // handle CSR and CSC formats
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;

    // CSC/CSR format of T is same as C.  Conform A and B to the format of C.
    bool T_is_csc = C->is_csc ;
    if (T_is_csc != A->is_csc)
    { 
        // Flip the sense of A_transpose.  For example, if C is CSC and A is
        // CSR, and A_transpose is true, then C=A'+B is being computed.  But
        // this is the same as C=A+B where A is treated as if it is CSC.
        A_transpose = !A_transpose ;
    }

    if (T_is_csc != B->is_csc)
    { 
        // Flip the sense of B_transpose.
        B_transpose = !B_transpose ;
    }

    if (A_transpose && B_transpose)
    { 
        // T=A'+B' is not computed.  Instead, T=A+B is computed first,
        // and then C = T' is computed.
        A_transpose = false ;
        B_transpose = false ;
        // The CSC format of T and C now differ.
        T_is_csc = !T_is_csc ;
    }

    if (!T_is_csc)
    {
        if (op_is_positional)
        { 
            // positional ops must be flipped, with i and j swapped
            op = GB_positional_binop_ijflip (op) ;
            opcode = op->opcode ;
        }
    }

    //--------------------------------------------------------------------------
    // decide when to apply the mask
    //--------------------------------------------------------------------------

    // GB_add and GB_emult can apply any non-complemented mask, but it is
    // faster to exploit the mask in GB_add / GB_emult only when it is very
    // sparse compared with A and B, or (in special cases) when it is easy
    // to apply.

    // check the CSR/CSC format of M
    bool M_is_csc = (M == NULL) ? T_is_csc : M->is_csc ;

    //--------------------------------------------------------------------------
    // transpose M if needed
    //--------------------------------------------------------------------------

    GrB_Matrix M1 = M ;
    bool M_transpose = (T_is_csc != M_is_csc) ;
    if (M_transpose)
    { 
        // MT = (bool) M'
        GBURBLE ("(M transpose) ") ;
        GB_CLEAR_STATIC_HEADER (MT, &MT_header) ;
        GB_OK (GB_transpose_cast (MT, GrB_BOOL, T_is_csc, M, Mask_struct,
            Werk)) ;
        M1 = MT ;
    }

    //--------------------------------------------------------------------------
    // transpose A and/or B if needed:
    //--------------------------------------------------------------------------

    bool A_is_pattern = false, B_is_pattern = false ;
    if (!eWiseAdd)
    { 
        // eWiseMult can create AT and BT as iso if the op is FIRST, SECOND, or
        // PAIR; eWiseAdd cannot.
        GB_binop_pattern (&A_is_pattern, &B_is_pattern, false, opcode) ;
    }

    GrB_Matrix A1 = A ;
    if (A_transpose)
    { 
        // AT = (xtype) A' or AT = (xtype) one (A')
        GBURBLE ("(A transpose) ") ;
        GB_CLEAR_STATIC_HEADER (AT, &AT_header) ;
        GB_OK (GB_transpose_cast (AT, op->xtype, T_is_csc, A, A_is_pattern,
            Werk)) ;
        A1 = AT ;
        ASSERT_MATRIX_OK (AT, "AT from transpose", GB0) ;
    }

    GrB_Matrix B1 = B ;
    if (B_transpose)
    { 
        // BT = (ytype) B' or BT = (ytype) one (B')
        GBURBLE ("(B transpose) ") ;
        GB_CLEAR_STATIC_HEADER (BT, &BT_header) ;
        GB_OK (GB_transpose_cast (BT, op->ytype, T_is_csc, B, B_is_pattern,
            Werk)) ;
        B1 = BT ;
        ASSERT_MATRIX_OK (BT, "BT from transpose", GB0) ;
    }

    //--------------------------------------------------------------------------
    // special cases
    //--------------------------------------------------------------------------

    // FUTURE::: handle more special cases:
    // C<M>+=A+B when C and A are dense, B is sparse.  M can be sparse.
    // C<M>+=A+B when C and B are dense, A is sparse.  M can be sparse.
    // C<M>+=A+B when C, A, and B are dense.  M can be sparse.
    // In all cases above, C remains dense and can be updated in-place
    // C_replace must be false.  M can be valued or structural.

    bool no_typecast =
        (op->ztype == C->type)              // no typecasting of C
        && (op->xtype == A1->type)          // no typecasting of A
        && (op->ytype == B1->type) ;        // no typecasting of B

    bool any_bitmap =
        GB_IS_BITMAP (C) ||
        GB_IS_BITMAP (M) ||
        GB_IS_BITMAP (A1) ||
        GB_IS_BITMAP (B1) ;

    bool any_pending_work =
        GB_ANY_PENDING_WORK (M1) ||
        GB_ANY_PENDING_WORK (A1) ||
        GB_ANY_PENDING_WORK (B1) ;

    bool any_iso = (A1->iso || B1->iso) ;

        // FUTURE: for sssp12:
        // C<A> = A+B where C is sparse and B is dense;
        // mask is structural, not complemented, C_replace is false.
        // C is not empty.  Use a kernel that computes T<A>=A+B
        // where T starts out empty; just iterate over the entries in A.

    if (GB_IS_FULL (A1)                     // A and B are full
        && GB_IS_FULL (B1)
        && !any_iso                         // A and B are not iso
        && (M == NULL) && !Mask_comp        // no mask
        && (C->is_csc == T_is_csc)          // no transpose of C
        && no_typecast                      // no typecasting
        && !op_is_positional                // op is not positional
        && !any_bitmap                      // no bitmap matrices
        && !any_pending_work)               // no matrix has pending work
    {

        if (GB_IS_FULL (C)                  // C is full
        && !C->iso                          // C is not iso
        && accum == op)                     // accum is same as the op
        { 

            //------------------------------------------------------------------
            // C += A+B where all 3 matrices are dense
            //------------------------------------------------------------------

            // C_replace is ignored
            GBURBLE ("dense C+=A+B ") ;
            info = GB_ewise_fulla (C, op, A1, B1) ;
            if (info != GrB_NO_VALUE)
            { 
                GB_FREE_ALL ;
                return (info) ;
            }

        }
        else if (accum == NULL)             // no accum
        { 

            //------------------------------------------------------------------
            // C = A+B where A and B are dense (C is anything)
            //------------------------------------------------------------------

            // C_replace is ignored
            GBURBLE ("dense C=A+B ") ;
            info = GB_ewise_fulln (C, op, A1, B1) ;

            if (info != GrB_NO_VALUE)
            { 
                GB_FREE_ALL ;
                return (info) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // T = A+B or A.*B, or with any mask M
    //--------------------------------------------------------------------------

    bool mask_applied = false ;
    GB_CLEAR_STATIC_HEADER (T, &T_header) ;

    if (eWiseAdd)
    { 

        //----------------------------------------------------------------------
        // T<any mask> = A+B
        //----------------------------------------------------------------------

        // TODO: check the mask condition in GB_add_sparsity.
        // Only exploit the mask in GB_add if it's more efficient than
        // exploiting it later, probably this condition:

            // (accum == NULL) && (C->is_csc == T->is_csc)
            // && (C_replace || GB_NNZ_UPPER_BOUND (C) == 0))

        // If that is true and the mask is applied, then T is transplanted as
        // the final C and the mask is no longer needed.  In this case, it
        // could be faster to exploit the mask duing GB_add.

        GB_OK (GB_add (T, T_type, T_is_csc, M1, Mask_struct, Mask_comp,
            &mask_applied, A1, B1, is_eWiseUnion, alpha, beta, op, false,
            Werk)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // T<any mask> = A.*B
        //----------------------------------------------------------------------

        // T can be returned with shallow components derived from its inputs A1
        // and/or B1.  In particular, if T is hypersparse, T->h may be a
        // shallow copy of A1->h, B1->h, or M1->h.  T is hypersparse if any
        // matrix A1, B1, or M1 are hypersparse.  Internally, T->h always
        // starts as a shallow copy of A1->h, B1->h, or M1->h, but it may be
        // pruned by GB_hypermatrix_prune, and thus no longer shallow.

        GB_OK (GB_emult (T, T_type, T_is_csc, M1, Mask_struct, Mask_comp,
            &mask_applied, A1, B1, op, Werk)) ;

        //----------------------------------------------------------------------
        // transplant shallow content from AT, BT, or MT
        //----------------------------------------------------------------------

        // If T is hypersparse, T->h is always a shallow copy of A1->h, B1->h,
        // or M1->h.  Any of the three matrices A1, B1, or M1 may be temporary
        // transposes, AT, BT, and MT respectively.  If T->h is a shallow cpoy
        // of a temporary matrix, then change the ownership of the T->h array,
        // from the temporary matrix into T, so that T->h is not freed when AT,
        // BT, and MT are freed.

        // GB_transpose can return all kinds of shallow components, particularly
        // when transposing vectors.  It can return AT->h as shallow copy of
        // A->i, for example.

        if (T->h_shallow)
        {
            // T->h is shallow and T is hypersparse
            ASSERT (GB_IS_HYPERSPARSE (T)) ;

            // one of A1, B1, or M1 is hypersparse
            ASSERT (GB_IS_HYPERSPARSE (A1) || GB_IS_HYPERSPARSE (B1) ||
                    GB_IS_HYPERSPARSE (M1))
            if (A_transpose && T->h == A1->h)
            { 
                // A1 is the temporary matrix AT.  AT->h might itself be a
                // shallow copy of A->h or A->i, from GB_transpose.
                ASSERT (A1 == AT) ;
                T->h_shallow = AT->h_shallow ;
                T->h_size = AT->h_size ;
                AT->h_shallow = true ;
            }
            else if (B_transpose && T->h == B1->h)
            { 
                // B1 is the temporary matrix BT.  BT->h might itself be a
                // shallow copy of B->h or B->i, from GB_transpose.
                ASSERT (B1 == BT) ;
                T->h_shallow = BT->h_shallow ;
                T->h_size = BT->h_size ;
                BT->h_shallow = true ;
            }
            else if (M_transpose && T->h == M1->h)
            { 
                // M1 is the temporary matrix MT.  MT->h might itself be a
                // shallow copy of M->h or M->i, from GB_transpose.
                ASSERT (M1 == MT) ;
                T->h_shallow = MT->h_shallow ;
                T->h_size = MT->h_size ;
                MT->h_shallow = true ;
            }

            // T->h may still be shallow, but if so, it is a shallow copy of
            // some component of the user input matrices A, B, or M, and must
            // remain shallow.  A deep copy of it will be made when T->h is
            // transplanted into the result C.
            ASSERT (GB_IMPLIES (T->h_shallow,
                (T->h == A1->h || T->h == B1->h ||
                 (M1 != NULL && T->h == M1->h)))) ;
        }
    }

    //--------------------------------------------------------------------------
    // free the transposed matrices
    //--------------------------------------------------------------------------

    GB_Matrix_free (&AT) ;
    GB_Matrix_free (&BT) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T): accumulate the results into C via the mask
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (T, "T from GB_ewise, prior to C<M>=accum(C,T)", GB0) ;

    if ((accum == NULL) && (C->is_csc == T->is_csc)
        && (M == NULL || (M != NULL && mask_applied))
        && (C_replace || GB_NNZ_UPPER_BOUND (C) == 0))
    { 
        // C = 0 ; C = (ctype) T ; with the same CSR/CSC format.  The mask M
        // (if any) has already been applied.  If C is also empty, or to be
        // cleared anyway, and if accum is not present, then T can be
        // transplanted directly into C, as C = (ctype) T, typecasting if
        // needed.  If no typecasting is done then this takes no time at all
        // and is a pure transplant.  Also conform C to its desired
        // hypersparsity.
        GB_Matrix_free (&MT) ;
        GB_OK (GB_transplant_conform (C, C->type, &T, Werk)) ;
        return (GB_block (C, Werk)) ;
    }
    else
    { 
        // C<M> = accum (C,T)
        // GB_accum_mask also conforms C to its desired hypersparsity
        info = GB_accum_mask (C, M, MT, accum, &T, C_replace, Mask_comp,
            Mask_struct, Werk) ;
        GB_Matrix_free (&MT) ;
        return (info) ;
    }
}

