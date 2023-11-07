//------------------------------------------------------------------------------
// gbargsort: sort a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:

// [C,P] = gbargsort (A, dim, direction)

// where dim = 1 to sort the columns of A, dim = 2 to the rows of A.
// direction is 'ascend' or 'descend'.

#include "gb_interface.h"

#define USAGE "usage: [C,P] = gbargsort (A, dim, direction)"

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

    gb_usage (nargin == 3 && (nargout == 2 || nargout == 1), USAGE) ;

    //--------------------------------------------------------------------------
    // find the arguments and determine the sort direction
    //--------------------------------------------------------------------------

    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    int dim = (int) mxGetScalar (pargin [1]) ;
    CHECK_ERROR (dim < 0 || dim > 2, "invalid dim") ;

    #define LEN 256
    char direction [LEN+2] ;
    gb_mxstring_to_string (direction, LEN, pargin [2], "direction") ;

    GrB_Type type ;
    OK (GxB_Matrix_type (&type, A)) ;

    GrB_BinaryOp op ;
    if (MATCH (direction, "ascend"))
    { 
        // ascending sort
        if      (type == GrB_BOOL  ) op = GrB_LT_BOOL   ;
        else if (type == GrB_INT8  ) op = GrB_LT_INT8   ;
        else if (type == GrB_INT16 ) op = GrB_LT_INT16  ;
        else if (type == GrB_INT32 ) op = GrB_LT_INT32  ;
        else if (type == GrB_INT64 ) op = GrB_LT_INT64  ;
        else if (type == GrB_UINT8 ) op = GrB_LT_UINT8  ;
        else if (type == GrB_UINT16) op = GrB_LT_UINT16 ;
        else if (type == GrB_UINT32) op = GrB_LT_UINT32 ;
        else if (type == GrB_UINT64) op = GrB_LT_UINT64 ;
        else if (type == GrB_FP32  ) op = GrB_LT_FP32   ;
        else if (type == GrB_FP64  ) op = GrB_LT_FP64   ;
        else ERROR ("unsupported type") ;
    }
    else if (MATCH (direction, "descend"))
    { 
        // descending sort
        if      (type == GrB_BOOL  ) op = GrB_GT_BOOL   ;
        else if (type == GrB_INT8  ) op = GrB_GT_INT8   ;
        else if (type == GrB_INT16 ) op = GrB_GT_INT16  ;
        else if (type == GrB_INT32 ) op = GrB_GT_INT32  ;
        else if (type == GrB_INT64 ) op = GrB_GT_INT64  ;
        else if (type == GrB_UINT8 ) op = GrB_GT_UINT8  ;
        else if (type == GrB_UINT16) op = GrB_GT_UINT16 ;
        else if (type == GrB_UINT32) op = GrB_GT_UINT32 ;
        else if (type == GrB_UINT64) op = GrB_GT_UINT64 ;
        else if (type == GrB_FP32  ) op = GrB_GT_FP32   ;
        else if (type == GrB_FP64  ) op = GrB_GT_FP64   ;
        else ERROR ("unsupported type") ;
    }
    else
    { 
        ERROR2 ("unrecognized direction: %s\n", direction) ;
    }

    GrB_Descriptor desc ;
    if (dim == 1)
    { 
        // sort the columns of A
        desc = GrB_DESC_T0 ;
    }
    else // dim == 2
    { 
        // sort the rows of A
        desc = NULL ;
    }

    //--------------------------------------------------------------------------
    // create the outputs C and P
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, P = NULL ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    OK (GrB_Matrix_new (&C, type, nrows, ncols)) ;
    if (nargout > 1)
    { 
        OK (GrB_Matrix_new (&P, GrB_INT64, nrows, ncols)) ;
    }

    //--------------------------------------------------------------------------
    // sort the matrix
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_sort (C, P, op, A, desc)) ;

    //--------------------------------------------------------------------------
    // add 1 to the entries in P, to convert to 1-based indexing
    //--------------------------------------------------------------------------

    if (P != NULL)
    { 
        OK (GrB_Matrix_apply_BinaryOp2nd_INT64 (P, NULL, NULL, GrB_PLUS_INT64,
            P, (int64_t) 1, NULL)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    pargout [0] = gb_export (&C, KIND_GRB) ;
    if (nargout > 1)
    { 
        pargout [1] = gb_export (&P, KIND_GRB) ;
    }
    GB_WRAPUP ;
}

