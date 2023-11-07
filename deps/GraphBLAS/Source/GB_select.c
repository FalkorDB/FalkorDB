//------------------------------------------------------------------------------
// GB_select: apply a select operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C, select(A,Thunk)) or select(A,Thunk)')

#define GB_FREE_ALL         \
{                           \
    GB_Matrix_free (&T) ;   \
}

#include "GB_select.h"
#include "GB_accum_mask.h"
#include "GB_transpose.h"
#include "GB_scalar_wrap.h"

GrB_Info GB_select          // C<M> = accum (C, select(A,k)) or select(A',k)
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // C descriptor
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // descriptor for M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op_in,
    const GrB_Matrix A,             // input matrix
    const GrB_Scalar Thunk,         // always present
    const bool A_transpose,         // A matrix descriptor
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M and/or A

    GrB_IndexUnaryOp op = op_in ;
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    GB_RETURN_IF_NULL_OR_FAULTY (Thunk) ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;

    ASSERT_MATRIX_OK (C, "C input for GB_select", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_select", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_select", GB0) ;
    ASSERT_INDEXUNARYOP_OK (op, "indexunaryop for GB_select", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for GB_select", GB0) ;
    ASSERT_SCALAR_OK (Thunk, "Thunk for GB_select", GB0) ;

    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;

    // check domains and dimensions for C<M> = accum (C,T)
    GrB_Info info ;
    GB_OK (GB_compatible (C->type, C, M, Mask_struct, accum, A->type, Werk));

    GB_Type_code xcode = (op->xtype == NULL) ? GB_ignore_code : op->xtype->code;
    GB_Opcode opcode = op->opcode ;
    ASSERT (GB_IS_INDEXUNARYOP_CODE (opcode)) ;
    ASSERT (opcode != GB_FLIPDIAGINDEX_idxunop_code) ;

    // A must also be compatible with op->xtype
    if (!GB_Type_compatible (A->type, op->xtype))
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Incompatible type for C=%s(A,Thunk):\n"
            "input A type [%s]\n"
            "cannot be typecast to operator input of type [%s]",
            op->name, A->type->name, op->xtype->name) ;
    }

    // check the dimensions
    int64_t tnrows = (A_transpose) ? GB_NCOLS (A) : GB_NROWS (A) ;
    int64_t tncols = (A_transpose) ? GB_NROWS (A) : GB_NCOLS (A) ;
    if (GB_NROWS (C) != tnrows || GB_NCOLS (C) != tncols)
    { 
        GB_ERROR (GrB_DIMENSION_MISMATCH,
            "Dimensions not compatible:\n"
            "output is " GBd "-by-" GBd "\n"
            "input is " GBd "-by-" GBd "%s",
            GB_NROWS (C), GB_NCOLS (C),
            tnrows, tncols, A_transpose ? " (transposed)" : "") ;
    }

    // finish any pending work on the Thunk
    GrB_Type ttype = Thunk->type ;
    GB_MATRIX_WAIT (Thunk) ;

    // check the GrB_IndexUnaryOp
    if (GB_nnz ((GrB_Matrix) Thunk) == 0)
    { 
        // Thunk cannot be empty for GrB_select
        GB_ERROR (GrB_EMPTY_OBJECT, "Thunk for C=%s(A,Thunk)"
            " cannot be an empty scalar\n", op->name) ;
    }

    if (!GB_Type_compatible (GrB_BOOL, op->ztype))
    { 
        // GrB_IndexUnaryOp ztype must be compatible with GrB_BOOL
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Output of user-defined IndexUnaryOp %s is %s\n"
            "which cannot be typecasted to bool\n",
            op->name, op->ztype->name) ;
    }

    if (!GB_Type_compatible (ttype, op->ytype))
    { 
        // Thunk must be typecasted to the op->ytype
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Incompatible type for C=%s(A,Thunk):\n"
            "input Thunk type [%s] and op thunk type [%s]"
            " not compatible",
            op->name, ttype->name, op->ytype->name) ;
    }

    // quick return if an empty mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp, Mask_struct) ;

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (M) ;        // TODO: delay until accum/mask phase
    GB_MATRIX_WAIT (A) ;        // TODO: could tolerate jumbled in some cases

    GB_BURBLE_DENSE (C, "(C %s) ") ;
    GB_BURBLE_DENSE (M, "(M %s) ") ;
    GB_BURBLE_DENSE (A, "(A %s) ") ;

    //--------------------------------------------------------------------------
    // handle the CSR/CSC format and the transposed case
    //--------------------------------------------------------------------------

    // A and C can be in CSR or CSC format (in any combination), and A can be
    // transposed first via A_transpose.  However, A is not explicitly
    // transposed first.  Instead, the selection operation is modified by
    // changing the operator, and the resulting matrix T is transposed, if
    // needed.

    // Instead of explicitly transposing the input matrix A and output T:
    // If A in CSC format and not transposed: treat as if A and T were CSC
    // If A in CSC format and transposed:     treat as if A and T were CSR
    // If A in CSR format and not transposed: treat as if A and T were CSR
    // If A in CSR format and transposed:     treat as if A and T were CSC

    bool A_csc = (A->is_csc == !A_transpose) ;

    // The final transpose, if needed, is accomplished in GB_accum_mask, by
    // tagging T as the same CSR/CSC format as A_csc.  If the format of T and C
    // do not match, GB_accum_mask transposes T, computing C<M>=accum(C,T').

    //--------------------------------------------------------------------------
    // change the op if needed
    //--------------------------------------------------------------------------

    bool flipij = !A_csc ;

    ASSERT_SCALAR_OK (Thunk, "Thunk now GB_select", GB0) ;

    bool make_copy = false ;
    bool is_empty = false ;
    bool negate_thunk = false ;
    bool bthunk = false ;
    bool op_is_bool_valued = (xcode == GB_BOOL_code &&
      (opcode >= GB_VALUENE_idxunop_code && opcode <= GB_VALUELE_idxunop_code)) ;
    if (op_is_bool_valued)
    { 
        GB_cast_scalar (&bthunk, GB_BOOL_code, Thunk->x, ttype->code,
            sizeof (bool)) ;
    }

    if (flipij && GB_IS_INDEXUNARYOP_CODE_POSITIONAL (opcode))
    { 

        //----------------------------------------------------------------------
        // tril, triu, diag, offdiag, ...: handle the flip
        //----------------------------------------------------------------------

        // The built-in operators are modified so they can always work as if A
        // were in CSC format.  If A is not in CSC, then the operation is
        // flipped.

        switch (opcode)
        {
            // TRIL becomes TRIU with thunk negated
            case GB_TRIL_idxunop_code : 
                negate_thunk = true ;
                op = GrB_TRIU ;
                break ;

            // TRIU becomes TRIL with thunk negated
            case GB_TRIU_idxunop_code : 
                negate_thunk = true ;
                op = GrB_TRIL ;
                break ;

            // DIAG, OFFDIAG, DIAGINDEX: same op, but negate the thunk
            case GB_DIAG_idxunop_code : 
            case GB_OFFDIAG_idxunop_code : 
            case GB_DIAGINDEX_idxunop_code : 
                negate_thunk = true ;
                break ;

            // ROWINDEX becomes COLINDEX
            case GB_ROWINDEX_idxunop_code  : 
                // i+thunk becomes j+thunk: no change to thunk
                op = (xcode == GB_INT32_code) ? GrB_COLINDEX_INT32
                                              : GrB_COLINDEX_INT64 ;
                break ;

            // COLINDEX becomes ROWINDEX
            case GB_COLINDEX_idxunop_code  : 
                // j+thunk becomes i+thunk: no change to thunk
                op = (xcode == GB_INT32_code) ? GrB_ROWINDEX_INT32
                                              : GrB_ROWINDEX_INT64 ;
                break ;

            // COLLE becomes ROWLE
            case GB_COLLE_idxunop_code : 
                // j <= thunk becomes i <= thunk: no change to thunk
                op = GrB_ROWLE ;
                break ;

            // COLGT becomes ROWGT
            case GB_COLGT_idxunop_code : 
                // j > thunk becomes i > thunk: no change to thunk
                op = GrB_ROWGT ;
                break ;

            // ROWLE becomes COLLE
            case GB_ROWLE_idxunop_code : 
                // i <= thunk becomes j <= thunk: no change to thunk
                op = GrB_COLLE ;
                break ;

            // ROWGT becomes COLGT
            case GB_ROWGT_idxunop_code : 
                // i > thunk becomes j > thunk: no change to thunk
                op = GrB_COLGT ;
                break ;

            default:;
        }

        // flipij is now false for any positional operator
        flipij = false ;

    }
    else if (op_is_bool_valued)
    {

        //----------------------------------------------------------------------
        // convert all VALUE* bool cases to VALUEEQ
        //----------------------------------------------------------------------

        op = GrB_VALUEEQ_BOOL ;
        switch (opcode)
        {

            case GB_VALUENE_idxunop_code   : // A(i,j) != thunk

                // use A(i,j) == !thunk
                bthunk = !bthunk ;
                break ;

            case GB_VALUEGT_idxunop_code   : // A(i,j) > thunk

                if (bthunk)
                { 
                    // if thunk is true,  return an empty matrix
                    is_empty = true ;
                }
                else
                { 
                    // otherwise, use A(i,j) == true
                    bthunk = true ;
                }
                break ;

            case GB_VALUEGE_idxunop_code   : // A(i,j) >= thunk

                if (!bthunk)
                { 
                    // if thunk is false, make a copy
                    make_copy = true ;
                }
                else
                { 
                    // otherwise, use A(i,j) == true
                    bthunk = true ;
                }
                break ;

            case GB_VALUELT_idxunop_code   : // A(i,j) < thunk

                // if thunk is false, return an empty matrix
                if (!bthunk)
                { 
                    is_empty = true ;
                }
                else
                { 
                    // otherwise, use A(i,j) == false
                    bthunk = false ;
                }
                break ;

            case GB_VALUELE_idxunop_code   : // A(i,j) <= thunk

                // if thunk is true, make a copy
                if (bthunk)
                { 
                    make_copy = true ;
                }
                else
                { 
                    // otherwise, use A(i,j) == false
                    bthunk = false ;
                }
                break ;

            default : ;
        }
    }

    if (opcode != GB_USER_idxunop_code)
    { 
        // flipij can still be true but is only needed for if the
        // GrB_IndexUnaryOp is user-defined.  So set here it to false for all
        // but user-defined ops.
        flipij = false ;
    }

    //--------------------------------------------------------------------------
    // negate the Thunk if needed
    //--------------------------------------------------------------------------

    GrB_Scalar Thunk2 ;
    struct GB_Scalar_opaque Thunk2_header ;
    int64_t ithunk = 0 ;
    if (negate_thunk)
    { 
        // Thunk = -(int64_t) Thunk
        GB_cast_scalar (&ithunk, GB_INT64_code, Thunk->x, ttype->code,
            sizeof (int64_t)) ;
        ithunk = -ithunk ;
        Thunk2 = GB_Scalar_wrap (&Thunk2_header, GrB_INT64, &ithunk) ;
    }
    else if (op_is_bool_valued)
    { 
        // Thunk = bthunk
        Thunk2 = GB_Scalar_wrap (&Thunk2_header, GrB_BOOL, &bthunk) ;
    }
    else
    { 
        // use Thunk as-is
        Thunk2 = Thunk ;
    }

    //--------------------------------------------------------------------------
    // create T
    //--------------------------------------------------------------------------

    GB_CLEAR_STATIC_HEADER (T, &T_header) ;

    if (make_copy)
    { 
        // T = A
        GB_OK (GB_shallow_copy (T, A_csc, A, Werk)) ;
    }
    else if (is_empty)
    { 
        // T is an empty non-iso matrix
        GB_OK (GB_new (&T, // auto (sparse or hyper), existing header
            A->type, A->vlen, A->vdim, GB_Ap_calloc, A_csc,
            GxB_SPARSE + GxB_HYPERSPARSE, GB_Global_hyper_switch_get ( ), 1)) ;
    }
    else
    { 
        // T = select (A, Thunk)
        GB_OK (GB_selector (T, op, flipij, A, Thunk2, Werk)) ;
    }

    T->is_csc = A_csc ;
    ASSERT_MATRIX_OK (T, "T=select(A,Thunk) output", GB0) ;
    ASSERT_MATRIX_OK (C, "C for accum; T=select(A,Thunk) output", GB0) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T): accumulate the results into C via the mask
    //--------------------------------------------------------------------------

    return (GB_accum_mask (C, M, NULL, accum, &T, C_replace, Mask_comp,
        Mask_struct, Werk)) ;
}

