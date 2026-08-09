//------------------------------------------------------------------------------
// gbmex_eunion: sparse matrix union
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_eunion is an interface to GxB_Matrix_eWiseUnion, for GrB.eunion
// and GhB.eunion.

// Usage for GrB and GhB (omitting desc argument):

// C = GrB.eunion (op, A, alpha, B, beta)                 C = op(A,alpha,B,beta)
// C = GrB.eunion (Cin, op, A, alpha, B, beta)            C = Cin ; C = op(...)
// C = GrB.eunion (Cin, accum, op, A, alpha, B, beta)     C = Cin ; C += op(...)
// C = GrB.eunion (Cin, M, op, A, alpha, B, beta)         C = Cin ; C<M> =op(..)
// C = GrB.eunion (Cin, M, accum, op, A, alpha, B, beta)  C = Cin ; C<M>+=op(..)

// Usage for GhB only:

// GhB.eunion (C, op, A, alpha, B, beta)                  C = op(A,alpha,B,beta)
// GhB.eunion (C, accum, op, A, alpha, B, beta)           C += op(...)
// GhB.eunion (C, M, op, A, alpha, B, beta)               C<M> = op(...)
// GhB.eunion (C, M, accum, op, A, alpha, B, beta)        C<M> += op(...)

#include "gb_interface.h"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"

#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                       \
    GrB_Matrix_free (&M_to_free) ;      \
    GrB_Matrix_free (&A_to_free) ;      \
    GrB_Matrix_free (&alpha_to_free) ;  \
    GrB_Matrix_free (&B_to_free) ;      \
    GrB_Matrix_free (&beta_to_free) ;   \
    GrB_Descriptor_free (&desc) ;

#undef  FREE_ALL
#define FREE_ALL                        \
    FREE_WORK ;                         \
    GrB_Matrix_free (&C) ;

#define USAGE \
"usage: C = GrB.eunion (Cin, M, accum, binop, A, alpha, B, beta, desc)"

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
        alpha = NULL, beta = NULL, M_to_free = NULL,
        A_to_free = NULL, B_to_free = NULL, alpha_to_free = NULL,
        beta_to_free = NULL ;
    GrB_Descriptor desc = NULL ;

    GBMX_USAGE (nargin >= 4 && nargin <= 10 && nargout <= 2, USAGE) ;
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

    CHECK_ERROR (nmatrices < 4 || nstrings < 1 || ncells > 0, USAGE) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor (&desc, &gbdesc, arena, err)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (nmatrices == 4)
    { 
        CHECK_ERROR (inplace, "invalid in-place usage") ;
        OK (gb_get_matrix (&A    , &A_to_free    , &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&alpha, &alpha_to_free, &(Matrix [1]), arena, err)) ;
        OK (gb_get_matrix (&B    , &B_to_free    , &(Matrix [2]), arena, err)) ;
        OK (gb_get_matrix (&beta , &beta_to_free , &(Matrix [3]), arena, err)) ;
    }
    else if (nmatrices == 5)
    { 
        OK (gb_get_deep   (&C    , inplace,        &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&A    , &A_to_free    , &(Matrix [1]), arena, err)) ;
        OK (gb_get_matrix (&alpha, &alpha_to_free, &(Matrix [2]), arena, err)) ;
        OK (gb_get_matrix (&B    , &B_to_free    , &(Matrix [3]), arena, err)) ;
        OK (gb_get_matrix (&beta , &beta_to_free , &(Matrix [4]), arena, err)) ;
    }
    else // if (nmatrices == 6)
    { 
        OK (gb_get_deep   (&C    , inplace,        &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&M    , &M_to_free    , &(Matrix [1]), arena, err)) ;
        OK (gb_get_matrix (&A    , &A_to_free    , &(Matrix [2]), arena, err)) ;
        OK (gb_get_matrix (&alpha, &alpha_to_free, &(Matrix [3]), arena, err)) ;
        OK (gb_get_matrix (&B    , &B_to_free    , &(Matrix [4]), arena, err)) ;
        OK (gb_get_matrix (&beta , &beta_to_free , &(Matrix [5]), arena, err)) ;
    }

    uint64_t n ;
    OK (GrB_Matrix_nrows (&n, alpha)) ;
    CHECK_ERROR (n != 1, "alpha must be a scalar") ;
    OK (GrB_Matrix_ncols (&n, alpha)) ;
    CHECK_ERROR (n != 1, "alpha must be a scalar") ;
    OK (GrB_Matrix_nrows (&n, beta)) ;
    CHECK_ERROR (n != 1, "beta must be a scalar") ;
    OK (GrB_Matrix_ncols (&n, beta)) ;
    CHECK_ERROR (n != 1, "beta must be a scalar") ;

    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GxB_Matrix_type (&btype, B)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL, op = NULL ;

    if (nstrings == 1)
    { 
        OK (gb_string_to_binop (&op, String [0], atype, btype, err)) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        OK (gb_string_to_binop (&accum, String [0], ctype, ctype, err)) ;
        OK (gb_string_to_binop (&op   , String [1], atype, btype, err)) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
        // get the descriptor contents to determine if A is transposed
        int in0 ;
        OK (GrB_Descriptor_get_INT32 (desc, &in0, GrB_INP0)) ;
        bool A_transpose = (in0 == GrB_TRAN) ;

        // get the size of A
        uint64_t anrows, ancols ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;

        // determine the size of C
        uint64_t cnrows = (A_transpose) ? ancols : anrows ;
        uint64_t cncols = (A_transpose) ? anrows : ancols ;

        // use the ztype of the op as the type of C
        OK (gb_binaryop_ztype (&ctype, op, err)) ;

        // create the matrix C and set its format and sparsity
        OK (gb_get_format (cnrows, cncols, A, B, &(gbdesc.fmt), err)) ;
        OK (gb_get_sparsity (A, B, &(gbdesc.sparsity), err)) ;
        OK (gb_new (&C, ctype, cnrows, cncols, gbdesc.fmt, gbdesc.sparsity,
            arena, err)) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += A+B
    //--------------------------------------------------------------------------

    OK1 (C, GxB_Matrix_eWiseUnion (C, M, accum, op,
        A, (GrB_Scalar) alpha, B, (GrB_Scalar) beta, desc)) ;

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

