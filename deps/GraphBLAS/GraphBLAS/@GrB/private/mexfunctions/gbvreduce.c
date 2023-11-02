//------------------------------------------------------------------------------
// gbvreduce: reduce a matrix to a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbvreduce is an interface to GrB_Matrix_reduce.

// Usage:

//  C = gbvreduce (op, A)
//  C = gbvreduce (op, A, desc)
//  C = gbvreduce (Cin, M, op, A, desc)
//  C = gbvreduce (Cin, accum, op, A, desc)
//  C = gbvreduce (Cin, M, accum, op, A, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A and the descriptor).

#include "gb_interface.h"

#define USAGE "usage: C = GrB.vreduce (Cin, M, accum, op, A, desc)"

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

    gb_usage (nargin >= 2 && nargin <= 6 && nargout <= 2, USAGE) ;

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

    CHECK_ERROR (nmatrices < 1 || nmatrices > 3 || nstrings < 1 || ncells > 0,
        USAGE) ;

    // ensure the descriptor is present, and set GxB_SORT to true
    if (desc == NULL)
    { 
        OK (GrB_Descriptor_new (&desc)) ;
    }
    OK (GxB_Desc_set (desc, GxB_SORT, true)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    GrB_Type atype, ctype = NULL ;
    GrB_Matrix C = NULL, M = NULL, A ;

    if (nmatrices == 1)
    { 
        A = gb_get_shallow (Matrix [0]) ;
    }
    else if (nmatrices == 2)
    { 
        C = gb_get_deep    (Matrix [0]) ;
        A = gb_get_shallow (Matrix [1]) ;
    }
    else // if (nmatrices == 3)
    { 
        C = gb_get_deep    (Matrix [0]) ;
        M = gb_get_shallow (Matrix [1]) ;
        A = gb_get_shallow (Matrix [2]) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    if (C != NULL)
    { 
        CHECK_ERROR (C->h != NULL, "Cin cannot be hypersparse") ;
        CHECK_ERROR (!(C->is_csc), "Cin must be stored by column") ;
        CHECK_ERROR (!GB_VECTOR_OK (C), "Cin must be a column vector") ;
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
        // get the descriptor contents to determine if A is transposed
        GrB_Desc_Value in0 ;
        OK (GxB_Desc_get (desc, GrB_INP0, &in0)) ;
        bool A_transpose = (in0 == GrB_TRAN) ;

        // get the size of A
        GrB_Index anrows, ancols ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;

        // determine the size of the vector C
        GrB_Index cnrows = (A_transpose) ? ancols : anrows ;

        // use the ztype of the monoid as the type of C
        GrB_BinaryOp binop ;
        OK (GxB_Monoid_operator (&binop, monoid)) ;
        OK (GxB_BinaryOp_ztype (&ctype, binop)) ;

        // create the matrix C and set its format and sparsity
        fmt = gb_get_format (cnrows, 1, A, NULL, fmt) ;
        sparsity = gb_get_sparsity (A, NULL, sparsity) ;
        C = gb_new (ctype, cnrows, 1, fmt, sparsity) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += reduce(A)
    //--------------------------------------------------------------------------

    OK1 (C, GrB_Matrix_reduce_Monoid ((GrB_Vector) C, (GrB_Vector) M,
        accum, monoid, A, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&M)) ;
    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    GB_WRAPUP ;
}

