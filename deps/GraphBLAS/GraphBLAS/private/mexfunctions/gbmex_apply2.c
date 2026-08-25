//------------------------------------------------------------------------------
// gbmex_apply2: apply idxunop or binary op to a matrix, with scalar binding
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_apply2 is an interface to GrB_Matrix_apply_BinaryOp1st_Scalar,
// GrB_Matrix_apply_BinaryOp2nd_Scalar, and GrB_Matrix_apply_IndexOp_Scalar.
// One of the inputs A or B are non-empty scalars.  This method implements
// GrB.apply2 and GhB.apply2.

// Usage for GrB and GhB (omitting desc argument):

// C = GrB.apply2 (op, A, B)                    % C = op(A,B)
// C = GrB.apply2 (Cin, op, A, B)               % C = Cin ; C = op(A,B)
// C = GrB.apply2 (Cin, accum, op, A, B)        % C = Cin ; C += op(A,B)
// C = GrB.apply2 (Cin, M, op, A, B)            % C = Cin ; C<M> = op(A,B)
// C = GrB.apply2 (Cin, M, accum, op, A, B)     % C = Cin ; C<M> += op(A,B)

// Usage for GhB only:

// GhB.apply2 (C, op, A, B)                     % C = op(A,B)
// GhB.apply2 (C, accum, op, A, B)              % C += op(A,B)
// GhB.apply2 (C, M, op, A, B)                  % C<M> = op(A,B)
// GhB.apply2 (C, M, accum, op, A, B)           % C<M> += op(A,B)

// Either A or B (or both) must be a non-empty scalar (1-by-1, with 1 entry).
// If both A and B are non-empty scalars, then A is treated as the input
// 'matrix' and B is treated as the scalar.

#include "gb_interface.h"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"

#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&M_to_free) ;  \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Matrix_free (&B_to_free) ;  \
    GrB_Scalar_free (&Thunk) ;      \
    GrB_Descriptor_free (&desc) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = GrB.apply2 (Cin, M, accum, op, A, B, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Type atype, btype, ctype = NULL ;
    GrB_Matrix *C_opaque = NULL, C = NULL, M = NULL, A = NULL, B = NULL,
        M_to_free = NULL, A_to_free = NULL, B_to_free = NULL ;
    GrB_Scalar Thunk = NULL ;
    GrB_Descriptor desc = NULL ;

    GBMX_USAGE (nargin >= 4 && nargin <= 8 && nargout <= 2, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    bool inplace = ghb && (nargout == 0) ;
    double *kind_output = NULL ;
    if (!inplace)
    { 
        if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;
        pargout [1] = mxCreateDoubleScalar (0) ;
        kind_output = (double *) mxGetData (pargout [1]) ;
    }
    else
    { 
        /* for tracking test coverage */ ;
    }

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [6] ;
    mxArray *Cell [2] ;
    char String [2][LEN+2] ;
    int nmatrices, nstrings, ncells ;
    struct gb_descriptor_struct gbdesc ;
    gbmx_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String,
        &nstrings, Cell, &ncells, &gbdesc) ;

    CHECK_ERROR (nmatrices < 2 || nmatrices > 4 || nstrings < 1 || ncells > 0,
        USAGE) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor (&desc, &gbdesc, arena, err)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (nmatrices == 2)
    { 
        CHECK_ERROR (inplace, "invalid in-place usage") ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&B, &B_to_free, &(Matrix [1]), arena, err)) ;
    }
    else if (nmatrices == 3)
    { 
        OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [1]), arena, err)) ;
        OK (gb_get_matrix (&B, &B_to_free, &(Matrix [2]), arena, err)) ;
    }
    else // if (nmatrices == 4)
    { 
        OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&M, &M_to_free, &(Matrix [1]), arena, err)) ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [2]), arena, err)) ;
        OK (gb_get_matrix (&B, &B_to_free, &(Matrix [3]), arena, err)) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GxB_Matrix_type (&btype, B)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // determine which input is the scalar and which is the matrix
    //--------------------------------------------------------------------------

    uint64_t anrows, ancols, bnrows, bncols, anvals, bnvals ;

    // get the size of A and B
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nvals (&anvals, A)) ;
    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;
    OK (GrB_Matrix_nvals (&bnvals, B)) ;

    GrB_Scalar scalar = NULL ;
    bool binop_bind1st ;
    bool A_is_scalar = (anrows == 1 && ancols == 1 && anvals == 1) ;
    bool B_is_scalar = (bnrows == 1 && bncols == 1 && bnvals == 1) ;

    if (B_is_scalar)
    { 
        // A is the matrix and B is the scalar
        binop_bind1st = false ;
        scalar = (GrB_Scalar) B ;   // NOTE: this is not allowed by the spec
    }
    else if (A_is_scalar)
    { 
        // A is the scalar and B is the matrix
        binop_bind1st = true ;
        scalar = (GrB_Scalar) A ;   // NOTE: this is not allowed by the spec
    }
    else
    {
        ERROR ("either A or B must be a non-empty scalar", GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // make sure the scalar has one entry
    //--------------------------------------------------------------------------

    // extract the int64 value of the scalar
    int64_t ithunk = 0 ;
    OK (GrB_Scalar_extractElement_INT64 (&ithunk, scalar)) ;

    //--------------------------------------------------------------------------
    // get the operators, and revise ithunk for idxunops
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL, op2 = NULL ;
    GrB_IndexUnaryOp idxunop = NULL ;

    if (nstrings == 1)
    { 
        OK (gb_string_to_binop_or_idxunop (&op2, &idxunop, &ithunk,
            &(String [0][0]), atype, btype, err)) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        OK (gb_string_to_binop (&accum, String [0], ctype, ctype, err)) ;
        OK (gb_string_to_binop_or_idxunop (&op2, &idxunop, &ithunk,
            &(String [1][0]), atype, btype, err)) ;
    }

    // create an int64 scalar from ithunk
    OK (GxB_Scalar_new_arena (&Thunk, GrB_INT64, arena, arena)) ;
    OK (GrB_Scalar_setElement_INT64 (Thunk, ithunk)) ;

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
        ASSERT (!inplace) ;

        // get the descriptor to determine if the input matrix is transposed
        uint64_t cnrows, cncols ;
        if (binop_bind1st)
        { 
            // A is the scalar and B is the matrix
            int in1 ;
            OK (GrB_Descriptor_get_INT32 (desc, &in1, GrB_INP1)) ;
            bool B_transpose = (in1 == GrB_TRAN) ;
            // determine the size of C
            cnrows = (B_transpose) ? bncols : bnrows ;
            cncols = (B_transpose) ? bnrows : bncols ;
        }
        else
        { 
            // A is the matrix and B is the scalar
            int in0 ;
            OK (GrB_Descriptor_get_INT32 (desc, &in0, GrB_INP0)) ;
            bool A_transpose = (in0 == GrB_TRAN) ;
            // determine the size of C
            cnrows = (A_transpose) ? ancols : anrows ;
            cncols = (A_transpose) ? anrows : ancols ;
        }

        // use the ztype of the op as the type of C
        if (op2 != NULL)
        { 
            OK (gb_binaryop_ztype (&ctype, op2, err)) ;
        }
        else
        { 
            int code = 0 ;
            OK (GrB_IndexUnaryOp_get_INT32 (idxunop, &code,
                GrB_OUTP_TYPE_CODE)) ;
            ctype = gb_code_to_type (code) ;
        }

        // create the matrix C and set its format and sparsity
        OK (gb_get_format (cnrows, cncols, A, B, &(gbdesc.fmt), err)) ;
        OK (gb_get_sparsity (A, B, &(gbdesc.sparsity), err)) ;
        OK (gb_new (&C, ctype, cnrows, cncols, gbdesc.fmt, gbdesc.sparsity,
            arena, err)) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += op (A,B) where one input is a scalar
    //--------------------------------------------------------------------------

    if (idxunop != NULL)
    { 
        OK1 (C, GrB_Matrix_apply_IndexOp_Scalar (C, M, accum, idxunop,
            A, Thunk, desc)) ;
    }
    else if (binop_bind1st)
    { 
        OK1 (C, GrB_Matrix_apply_BinaryOp1st_Scalar (C, M, accum, op2,
            scalar, B, desc)) ;
    }
    else
    { 
        OK1 (C, GrB_Matrix_apply_BinaryOp2nd_Scalar (C, M, accum, op2,
            A, scalar, desc)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    if (!inplace)
    { 
        OK (gb_export (C_opaque, &C, gbdesc.kind, ghb, err)) ;
        (*kind_output) = (double) gbdesc.kind ;
    }
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

