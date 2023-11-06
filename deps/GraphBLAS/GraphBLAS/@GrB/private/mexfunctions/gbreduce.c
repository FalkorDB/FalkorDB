//------------------------------------------------------------------------------
// gbreduce: reduce a sparse matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbreduce is an interface to GrB_Matrix_reduce_Monoid_Scalar.

// Usage:

//  cout = gbreduce (op, A)
//  cout = gbreduce (op, A, desc)
//  cout = gbreduce (cin, accum, op, A, desc)

// If cin is not present then it is implicitly a 1-by-1 matrix with no entries.

#include "gb_interface.h"

#define USAGE "usage: C = GrB.reduce (cin, accum, op, A, desc)"

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

    gb_usage (nargin >= 2 && nargin <= 5 && nargout <= 2, USAGE) ;

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

    CHECK_ERROR (nmatrices < 1 || nmatrices > 2 || nstrings < 1 || ncells > 0,
        USAGE) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    GrB_Type atype, ctype = NULL ;
    GrB_Matrix C = NULL, A ;

    if (nmatrices == 1)
    { 
        A = gb_get_shallow (Matrix [0]) ;
    }
    else // if (nmatrices == 2)
    { 
        C = gb_get_deep    (Matrix [0]) ;
        A = gb_get_shallow (Matrix [1]) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;
    GrB_Monoid monoid ;

    if (nstrings == 1)
    { 
        monoid = gb_mxstring_to_monoid (String [0], atype) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        accum  = gb_mxstring_to_binop  (String [0], ctype, ctype) ;
        monoid = gb_mxstring_to_monoid (String [1], atype) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
        // use the ztype of the monoid as the type of C
        GrB_BinaryOp binop ;
        OK (GxB_Monoid_operator (&binop, monoid)) ;
        OK (GxB_BinaryOp_ztype (&ctype, binop)) ;

        fmt = gb_get_format (1, 1, A, NULL, fmt) ;
        sparsity = gb_get_sparsity (A, NULL, sparsity) ;
        C = gb_new (ctype, 1, 1, fmt, sparsity) ;
    }

    //--------------------------------------------------------------------------
    // ensure C is 1-by-1
    //--------------------------------------------------------------------------

    GrB_Index cnrows, cncols ;
    OK (GrB_Matrix_nrows (&cnrows, C)) ;
    OK (GrB_Matrix_ncols (&cncols, C)) ;
    if (cnrows != 1 || cncols != 1)
    { 
        ERROR ("cin must be a scalar") ;
    }

    //--------------------------------------------------------------------------
    // compute C += reduce(A)
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_reduce_Monoid_Scalar ((GrB_Scalar) C, accum, monoid, A, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    GB_WRAPUP ;
}

