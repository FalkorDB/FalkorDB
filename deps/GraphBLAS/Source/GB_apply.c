//------------------------------------------------------------------------------
// GB_apply: apply a unary operator; optionally transpose a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C, op(A)) or accum (C, op(A)')

// GB_apply does the work for GrB_*_apply, including the binary op variants.

#define GB_FREE_ALL ;

#include "GB_apply.h"
#include "GB_binop.h"
#include "GB_transpose.h"
#include "GB_accum_mask.h"
#include "GB_scalar_wrap.h"

GrB_Info GB_apply                   // C<M> = accum (C, op(A)) or op(A')
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // C descriptor
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // M descriptor
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
        const GB_Operator op_in,        // unary/idxunop/binop to apply
        const GrB_Scalar scalar_in,     // scalar to bind to binop, or thunk
        bool binop_bind1st,             // if true, binop(x,A) else binop(A,y)
    const GrB_Matrix A,             // first or 2nd input:  matrix A
    bool A_transpose,               // A matrix descriptor
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M and/or A

    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;

    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    GB_RETURN_IF_NULL_OR_FAULTY (op_in) ;
    ASSERT_MATRIX_OK (C, "C input for GB_apply", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_apply", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_apply", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for GB_apply", GB0) ;
    ASSERT_OP_OK (op_in, "op for GB_apply", GB0) ;

    GB_Operator op = op_in ;
    GB_Opcode opcode = op->opcode ;
    GrB_Type T_type = op->ztype ;
    GrB_Scalar scalar = (GrB_Scalar) scalar_in ;

    bool op_is_unop = GB_IS_UNARYOP_CODE (opcode) ;
    bool op_is_binop = GB_IS_BINARYOP_CODE (opcode) ;
    bool op_is_idxunop = GB_IS_INDEXUNARYOP_CODE (opcode) ;
    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
    struct GB_Scalar_opaque Thunk_header ;
    int64_t ithunk = 0 ;

    if (op_is_unop)
    {
        // apply a unary operator: scalar is ignored
        ASSERT_OP_OK ( op, "unop for GB_apply", GB0) ;
        if (!op_is_positional)
        {
            // A must also be compatible with op->xtype
            if (!GB_Type_compatible (A->type, op->xtype))
            { 
                GB_ERROR (GrB_DOMAIN_MISMATCH,
                    "Incompatible type for z=%s(x):\n"
                    "input A of type [%s]\n"
                    "cannot be typecast to x input of type [%s]",
                    op->name, A->type->name, op->xtype->name) ;
            }
        }
    }
    else if (op_is_binop)
    {
        // apply a binary operator, with one input bound to a scalar
        ASSERT_OP_OK (op, "binop for GB_apply", GB0) ;
        ASSERT_SCALAR_OK (scalar, "scalar for GB_apply", GB0) ;
        if (!op_is_positional)
        {
            bool op_is_first  = opcode == GB_FIRST_binop_code ;
            bool op_is_second = opcode == GB_SECOND_binop_code ;
            bool op_is_pair   = opcode == GB_PAIR_binop_code ;
            if (binop_bind1st)
            {
                // C = op (scalar,A)
                // A must be compatible with op->ytype
                if (!(op_is_first || op_is_pair ||
                      GB_Type_compatible (A->type, op->ytype)))
                { 
                    GB_ERROR (GrB_DOMAIN_MISMATCH,
                        "Incompatible type for z=%s(x,y):\n"
                        "input A of type [%s]\n"
                        "cannot be typecast to y input of type [%s]",
                        op->name, A->type->name, op->ytype->name) ;
                }
                // scalar must be compatible with op->xtype
                if (!(op_is_second || op_is_pair ||
                      GB_Type_compatible (scalar->type, op->xtype)))
                { 
                    GB_ERROR (GrB_DOMAIN_MISMATCH,
                        "Incompatible type for z=%s(x,y):\n"
                        "input scalar of type [%s]\n"
                        "cannot be typecast to x input of type [%s]",
                        op->name, scalar->type->name, op->xtype->name) ;
                }
            }
            else
            {
                // C = op (A,scalar)
                // A must be compatible with op->xtype
                if (!(op_is_first || op_is_pair ||
                      GB_Type_compatible (A->type, op->xtype)))
                { 
                    GB_ERROR (GrB_DOMAIN_MISMATCH,
                        "Incompatible type for z=%s(x,y):\n"
                        "input A of type [%s]\n"
                        "cannot be typecast to x input of type [%s]",
                        op->name, A->type->name, op->xtype->name) ;
                }
                // scalar must be compatible with op->ytype
                if (!(op_is_second || op_is_pair
                      || GB_Type_compatible (scalar->type, op->ytype)))
                { 
                    GB_ERROR (GrB_DOMAIN_MISMATCH,
                        "Incompatible type for z=%s(x,y):\n"
                        "input scalar of type [%s]\n"
                        "cannot be typecast to y input of type [%s]",
                        op->name, scalar->type->name, op->ytype->name) ;
                }
            }
        }
    }
    else // op_is_idxunop
    {
        // apply an idxunop operator, with a thunk scalar
        ASSERT_OP_OK (op, "idxunop for GB_apply", GB0) ;
        ASSERT_SCALAR_OK (scalar, "thunk for GB_apply", GB0) ;
        // A must be compatible with op->xtype
        if (!GB_Type_compatible (A->type, op->xtype))
        { 
            GB_ERROR (GrB_DOMAIN_MISMATCH,
                "Incompatible type for z=%s(x,i,j,thunk):\n"
                "input A of type [%s]\n"
                "cannot be typecast to x input of type [%s]",
                op->name, A->type->name, op->xtype->name) ;
        }
        // scalar must be compatible with op->ytype
        if (!GB_Type_compatible (scalar->type, op->ytype))
        { 
            GB_ERROR (GrB_DOMAIN_MISMATCH,
                "Incompatible type for z=%s(x,i,j,thunk):\n"
                "input scalar of type [%s]\n"
                "cannot be typecast to thunk input of type [%s]",
                op->name, scalar->type->name, op->ytype->name) ;
        }
    }

    // check domains and dimensions for C<M> = accum (C,T)
    GrB_Info info ;
    GB_OK (GB_compatible (C->type, C, M, Mask_struct, accum, T_type, Werk)) ;

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

    // quick return if an empty mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp, Mask_struct) ;

    // delete any lingering zombies and assemble any pending tuples
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;      // A can be jumbled
    GB_MATRIX_WAIT (scalar) ;

    if (!op_is_unop && GB_nnz ((GrB_Matrix) scalar) != 1)
    { 
        // the scalar entry must be present
        GB_ERROR (GrB_EMPTY_OBJECT, "%s", "Scalar must contain an entry") ;
    }

    //--------------------------------------------------------------------------
    // rename binop and idxunop operators
    //--------------------------------------------------------------------------

    GB_binop_rename (&op, binop_bind1st) ;

    opcode = op->opcode ;
    op_is_unop = GB_IS_UNARYOP_CODE (opcode) ;
    op_is_binop = GB_IS_BINARYOP_CODE (opcode) ;
    op_is_idxunop = GB_IS_INDEXUNARYOP_CODE (opcode) ;
    op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;

    // all VALUE* index_unary ops have been renamed to their corresponding
    // binary ops.  Only positional and user-defined idxunops remain.
    ASSERT (GB_IMPLIES (op_is_idxunop,
        op_is_positional || opcode == GB_USER_idxunop_code)) ;

    //--------------------------------------------------------------------------
    // get the int64 value of the thunk for positional operators
    //--------------------------------------------------------------------------

    if (op_is_idxunop && op_is_positional)
    { 
        // ithunk = (int64) scalar
        GB_cast_scalar (&ithunk, GB_INT64_code, scalar->x, scalar->type->code,
            scalar->type->size) ;
        // wrap ithunk in the new scalar
        scalar = GB_Scalar_wrap (&Thunk_header, GrB_INT64, &ithunk) ;
    }

    //--------------------------------------------------------------------------
    // T = op(A) or op(A')
    //--------------------------------------------------------------------------

    bool T_is_csc = C->is_csc ;
    if (T_is_csc != A->is_csc)
    { 
        // Flip the sense of A_transpose
        A_transpose = !A_transpose ;
    }

    if (!T_is_csc && op_is_positional)
    {
        // positional ops must be flipped, with i and j swapped
        if (op_is_unop)
        { 
            op = (GB_Operator) GB_positional_unop_ijflip ((GrB_UnaryOp) op) ;
        }
        else if (op_is_binop)
        { 
            op = (GB_Operator) GB_positional_binop_ijflip ((GrB_BinaryOp) op) ;
        }
        else // op_is_idxunop
        { 
            // also revise ithunk as needed (TRIL, TRIU, DIAG, OFFDIAG only)
            op = (GB_Operator) GB_positional_idxunop_ijflip (&ithunk,
                (GrB_IndexUnaryOp) op) ;
        }
        opcode = op->opcode ;
    }

    // user operator must have i,j flipped
    bool flipij = (!T_is_csc && opcode == GB_USER_idxunop_code) ;

    if (A_transpose)
    { 
        // T = op (A'), typecasting to op->ztype
        GBURBLE ("(transpose-op) ") ;
        GB_CLEAR_STATIC_HEADER (T, &T_header) ;
        info = GB_transpose (T, T_type, T_is_csc, A, op, scalar,
            binop_bind1st, flipij, Werk) ;
        ASSERT (GB_JUMBLED_OK (T)) ;
        // A positional op is applied to C after the transpose is computed,
        // using the T_is_csc format.  The ijflip is handled above.
    }
    else if (M == NULL && accum == NULL && (C == A) && C->type == T_type
        && GB_nnz (C) > 0)
    {
        GBURBLE ("(in-place-op) ") ;
        // C = op (C), operating on the values in-place, with no typecasting
        // of the output of the operator with the matrix C.
        // No work to do if the op is identity.
        if (opcode != GB_IDENTITY_unop_code)
        {
            // the output Cx is aliased with C->x in GB_apply_op.
            GB_iso_code C_code_iso = GB_unop_code_iso (C, op, binop_bind1st) ;
            info = GrB_SUCCESS ;
            if (C_code_iso == GB_NON_ISO && C->iso)
            { 
                // expand C to non-iso; initialize C->x unless the op
                // is positional
                info = GB_convert_any_to_non_iso (C, !op_is_positional) ;
            }
            if (info == GrB_SUCCESS)
            { 
                // C->x = op (C->x) in place
                info = GB_apply_op ((GB_void *) C->x, C->type, C_code_iso,
                    op, scalar, binop_bind1st, flipij, C, Werk) ;
            }
            if (info == GrB_SUCCESS && C_code_iso != GB_NON_ISO)
            { 
                // compact the iso values of C
                C->iso = true ;
                info = GB_convert_any_to_iso (C, NULL) ;
            }
        }
        return (info) ;
    }
    else
    { 
        // T = op (A), pattern is a shallow copy of A, type is op->ztype.
        GBURBLE ("(shallow-op) ") ;
        GB_CLEAR_STATIC_HEADER (T, &T_header) ;
        info = GB_shallow_op (T, T_is_csc, op, scalar, binop_bind1st, flipij,
            A, Werk) ;
    }

    if (info != GrB_SUCCESS)
    { 
        GB_Matrix_free (&T) ;
        return (info) ;
    }

    ASSERT (T->is_csc == C->is_csc) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T): accumulate the results into C via the M
    //--------------------------------------------------------------------------

    return (GB_accum_mask (C, M, NULL, accum, &T, C_replace, Mask_comp,
        Mask_struct, Werk)) ;
}

