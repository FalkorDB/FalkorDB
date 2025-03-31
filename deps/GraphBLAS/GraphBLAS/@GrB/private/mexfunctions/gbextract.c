//------------------------------------------------------------------------------
// gbextract: extract entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbextract is an interface to GrB_Matrix_extract and
// GrB_Matrix_extract_[TYPE], computing the GraphBLAS expression:

//      C<#M,replace> = accum (C, A (I,J)) or
//      C<#M,replace> = accum (C, AT (I,J))

// Usage:

//      C = gbextract (Cin, M, accum, A, I, J, desc)

// A is required.  See GrB.m for more details.
// If accum or M is used, then Cin must appear.

#include "gb_interface.h"

#define USAGE "usage: C = GrB.extract (Cin, M, accum, A, I, J, desc)"

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

    gb_usage (nargin >= 1 && nargin <= 7 && nargout <= 2, USAGE) ;

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

    CHECK_ERROR (nmatrices < 1 || nmatrices > 3 || nstrings > 1, USAGE) ;

    //--------------------------------------------------------------------------
    // create the descriptor, if not present
    //--------------------------------------------------------------------------

    if (desc == NULL)
    { 
        OK (GrB_Descriptor_new (&desc)) ;
    }

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
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operator
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;

    if (nstrings == 1)
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        accum  = gb_mxstring_to_binop (String [0], ctype, ctype) ;
    }

    //--------------------------------------------------------------------------
    // get the size of A
    //--------------------------------------------------------------------------

    int in0 ;
    OK (GrB_Descriptor_get_INT32 (desc, &in0, GrB_INP0)) ;
    uint64_t anrows, ancols ;
    bool A_transpose = (in0 == GrB_TRAN) ;
    if (A_transpose)
    { 
        // T = AT (I,J) is to be extracted where AT = A'
        OK (GrB_Matrix_nrows (&ancols, A)) ;
        OK (GrB_Matrix_ncols (&anrows, A)) ;
    }
    else
    { 
        // T = A (I,J) is to be extracted
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;
    }

    //--------------------------------------------------------------------------
    // get I and J
    //--------------------------------------------------------------------------

    GrB_Vector I = NULL, J = NULL ;
    uint64_t cnrows = anrows, cncols = ancols ;
    int icells = 0, jcells = 0 ;
    int base_offset = (base == BASE_0_INT) ? 0 : 1 ;

    if (anrows == 1 && ncells == 1)
    { 
        // only J is present
        J = gb_mxcell_to_list (Cell [0], base_offset, ancols, &cncols, NULL) ;
        jcells = mxGetNumberOfElements (Cell [0]) ;
    }
    else if (ncells == 1)
    { 
        // only I is present
        I = gb_mxcell_to_list (Cell [0], base_offset, anrows, &cnrows, NULL) ;
        icells = mxGetNumberOfElements (Cell [0]) ;
    }
    else if (ncells == 2)
    { 
        // both I and J are present
        I = gb_mxcell_to_list (Cell [0], base_offset, anrows, &cnrows, NULL) ;
        J = gb_mxcell_to_list (Cell [1], base_offset, ancols, &cncols, NULL) ;
        icells = mxGetNumberOfElements (Cell [0]) ;
        jcells = mxGetNumberOfElements (Cell [1]) ;
    }

    if (icells > 1)
    { 
        // I is a 3-element vector containing a stride
        OK (GrB_Descriptor_set_INT32 (desc, GxB_IS_STRIDE, GxB_ROWINDEX_LIST)) ;
    }

    if (jcells > 1)
    { 
        // J is a 3-element vector containing a stride
        OK (GrB_Descriptor_set_INT32 (desc, GxB_IS_STRIDE, GxB_COLINDEX_LIST)) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    if (C == NULL)
    { 
        // Cin is not present: determine its size, same type as A.
        // T = A(I,J) or AT(I,J) will be extracted; accum must be null.
        // create the matrix C and set its format and sparsity
        fmt = gb_get_format (cnrows, cncols, A, NULL, fmt) ;
        sparsity = gb_get_sparsity (A, NULL, sparsity) ;
        ctype = atype ;
        C = gb_new (ctype, cnrows, cncols, fmt, sparsity) ;
    }

    //--------------------------------------------------------------------------
    // C<M> += A(I,J) or AT(I,J)
    //--------------------------------------------------------------------------

    OK1 (C, GxB_Matrix_extract_Vector (C, M, accum, A, I, J, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies and workspace
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&M)) ;
    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Vector_free (&I)) ;
    OK (GrB_Vector_free (&J)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    gb_wrapup ( ) ;
}

