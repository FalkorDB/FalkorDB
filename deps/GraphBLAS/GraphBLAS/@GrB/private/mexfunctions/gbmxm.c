//------------------------------------------------------------------------------
// gbmxm: sparse matrix-matrix multiplication
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmxm is an interface to GrB_mxm.

// Usage:

// C = gbmxm (semiring, A, B)
// C = gbmxm (semiring, A, B, desc)
// C = gbmxm (Cin, accum, semiring, A, B, desc)
// C = gbmxm (Cin, M, semiring, A, B, desc)
// C = gbmxm (Cin, M, accum, semiring, A, B, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A, B, and the descriptor).

#include "gb_interface.h"

#define USAGE "usage: C = GrB.mxm (Cin, M, accum, semiring, A, B, desc)"

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

    gb_usage (nargin >= 3 && nargin <= 7 && nargout <= 2, USAGE) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    mxArray *Matrix [6], *String [2], *Cell [2] ;
    base_enum_t base ;
    kind_enum_t kind ;
    int fmt ;
    int nmatrices, nstrings, ncells, sparsity ;
    GrB_Descriptor desc ;
    gb_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String, &nstrings,
        Cell, &ncells, &desc, &base, &kind, &fmt, &sparsity) ;

    CHECK_ERROR (nmatrices < 2 || nmatrices > 4 || nstrings < 1 || ncells > 0,
        USAGE) ;

    // ensure the descriptor is present, and set GxB_SORT to true
    if (desc == NULL)
    { 
        OK (GrB_Descriptor_new (&desc)) ;
    }
    OK (GrB_Descriptor_set_INT32 (desc, true, GxB_SORT)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    GrB_Type atype, btype, ctype = NULL ;
    GrB_Matrix C = NULL, M = NULL, A, B ;

    if (nmatrices == 2)
    { 
        A = gb_get_shallow (Matrix [0]) ;
        B = gb_get_shallow (Matrix [1]) ;
    }
    else if (nmatrices == 3)
    { 
        C = gb_get_deep    (Matrix [0]) ;
        A = gb_get_shallow (Matrix [1]) ;
        B = gb_get_shallow (Matrix [2]) ;
    }
    else // if (nmatrices == 4)
    { 
        C = gb_get_deep    (Matrix [0]) ;
        M = gb_get_shallow (Matrix [1]) ;
        A = gb_get_shallow (Matrix [2]) ;
        B = gb_get_shallow (Matrix [3]) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GxB_Matrix_type (&btype, B)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;
    GrB_Semiring semiring ;

    if (nstrings == 1)
    { 
        semiring = gb_mxstring_to_semiring (String [0], atype, btype) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        accum    = gb_mxstring_to_binop    (String [0], ctype, ctype) ;
        semiring = gb_mxstring_to_semiring (String [1], atype, btype) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
        // get the descriptor contents to determine if A and B are transposed
        int in0, in1 ;
        OK (GrB_Descriptor_get_INT32 (desc, &in0, GrB_INP0)) ;
        OK (GrB_Descriptor_get_INT32 (desc, &in1, GrB_INP1)) ;
        bool A_transpose = (in0 == GrB_TRAN) ;
        bool B_transpose = (in1 == GrB_TRAN) ;

        // get the size of A and B
        uint64_t anrows, ancols, bnrows, bncols ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;
        OK (GrB_Matrix_nrows (&bnrows, B)) ;
        OK (GrB_Matrix_ncols (&bncols, B)) ;

        // determine the size of C
        uint64_t cnrows = (A_transpose) ? ancols : anrows ;
        uint64_t cncols = (B_transpose) ? bnrows : bncols ;

        // use the semiring's ztype as the type of C
        int code ;
        OK (GrB_Semiring_get_INT32 (semiring, &code, GrB_OUTP_TYPE_CODE)) ;
        ctype = gb_code_to_type (code) ;

        // create the matrix C and set its format and sparsity
        fmt = gb_get_format (cnrows, cncols, A, B, fmt) ;
        sparsity = gb_get_sparsity (A, B, sparsity) ;
        C = gb_new (ctype, cnrows, cncols, fmt, sparsity) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += A*B
    //--------------------------------------------------------------------------

    OK1 (C, GrB_mxm (C, M, accum, semiring, A, B, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&M)) ;
    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&B)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    gb_wrapup ( ) ;
}

