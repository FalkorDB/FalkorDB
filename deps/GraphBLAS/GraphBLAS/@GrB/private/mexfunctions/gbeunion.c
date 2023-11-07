//------------------------------------------------------------------------------
// gbeunion: sparse matrix union
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbeunion is an interface to GxB_Matrix_eWiseUnion.

// Usage:

// C = gbeunion (binop, A, alpha, B, beta)
// C = gbeunion (binop, A, alpha, B, beta, desc)
// C = gbeunion (Cin, accum, binop, A, alpha, B, beta, desc)
// C = gbeunion (Cin, M, binop, A, alpha, B, beta, desc)
// C = gbeunion (Cin, M, accum, binop, A, alpha, B, beta, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A, B, and the descriptor).

#include "gb_interface.h"

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
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin >= 3 && nargin <= 9 && nargout <= 2, USAGE) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    mxArray *Matrix [6], *String [2], *Cell [2] ;
    base_enum_t base ;
    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    int nmatrices, nstrings, ncells, sparsity ;
    GrB_Descriptor desc ;
    gb_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String, &nstrings,
        Cell, &ncells, &desc, &base, &kind, &fmt, &sparsity) ;

    CHECK_ERROR (nmatrices < 4 || nstrings < 1 || ncells > 0, USAGE) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    GrB_Type atype, btype, ctype = NULL ;
    GrB_Matrix C = NULL, M = NULL, A, B, alpha, beta ;

    if (nmatrices == 4)
    { 
        A = gb_get_shallow (Matrix [0]) ;
        alpha = gb_get_shallow (Matrix [1]) ;
        B = gb_get_shallow (Matrix [2]) ;
        beta = gb_get_shallow (Matrix [3]) ;
    }
    else if (nmatrices == 5)
    { 
        C = gb_get_deep    (Matrix [0]) ;
        A = gb_get_shallow (Matrix [1]) ;
        alpha = gb_get_shallow (Matrix [2]) ;
        B = gb_get_shallow (Matrix [3]) ;
        beta = gb_get_shallow (Matrix [4]) ;
    }
    else // if (nmatrices == 6)
    { 
        C = gb_get_deep    (Matrix [0]) ;
        M = gb_get_shallow (Matrix [1]) ;
        A = gb_get_shallow (Matrix [2]) ;
        alpha = gb_get_shallow (Matrix [3]) ;
        B = gb_get_shallow (Matrix [4]) ;
        beta = gb_get_shallow (Matrix [5]) ;
    }

    GrB_Index n ;
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
        op    = gb_mxstring_to_binop (String [0], atype, btype) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        accum = gb_mxstring_to_binop (String [0], ctype, ctype) ;
        op    = gb_mxstring_to_binop (String [1], atype, btype) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
        // get the descriptor contents to determine if A is transposed
        GrB_Desc_Value in0 ;
        OK (GxB_Desc_get (desc, GrB_INP0, &in0)) ;
        bool A_transpose = (in0 == GrB_TRAN) ;

        // get the size of A
        GrB_Index anrows, ancols ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;

        // determine the size of C
        GrB_Index cnrows = (A_transpose) ? ancols : anrows ;
        GrB_Index cncols = (A_transpose) ? anrows : ancols ;

        // use the ztype of the op as the type of C
        OK (GxB_BinaryOp_ztype (&ctype, op)) ;

        // create the matrix C and set its format and sparsity
        fmt = gb_get_format (cnrows, cncols, A, B, fmt) ;
        sparsity = gb_get_sparsity (A, B, sparsity) ;
        C = gb_new (ctype, cnrows, cncols, fmt, sparsity) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += A+B
    //--------------------------------------------------------------------------

    OK1 (C, GxB_Matrix_eWiseUnion (C, M, accum, op,
        A, (GrB_Scalar) alpha, B, (GrB_Scalar) beta, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&M)) ;
    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&alpha)) ;
    OK (GrB_Matrix_free (&B)) ;
    OK (GrB_Matrix_free (&beta)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    GB_WRAPUP ;
}

