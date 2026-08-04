//------------------------------------------------------------------------------
// gbmex_argsort: sort a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:

// [C,P] = gbmex_argsort (A, dim, direction)

// where dim = 1 to sort the columns of A, dim = 2 to the rows of A.
// direction is 'ascend' or 'descend'.

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                       \
    GrB_Matrix_free (&A_to_free) ;

#undef  FREE_ALL
#define FREE_ALL                        \
    FREE_WORK ;                         \
    GrB_Matrix_free (&C) ;              \
    GrB_Matrix_free (&P) ;

#define USAGE "usage: [C,P] = gbmex_argsort (ghb, A, dim, direction)"

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

    GrB_Matrix *C_opaque = NULL, *P_opaque = NULL,
        A = NULL, A_to_free = NULL, C = NULL, P = NULL ;

    GBMX_USAGE (nargin == 4 && nargout <= 2, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    if (ghb)
    { 
        pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;
        if (nargout > 1)
        { 
            pargout [1] = gbmx_export_ghb_mxstruct (&P_opaque) ;
        }
    }

    //--------------------------------------------------------------------------
    // find the arguments and determine the sort direction
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [1]) ;

    int dim = (int) mxGetScalar (pargin [2]) ;
    CHECK_ERROR (dim < 0 || dim > 2, "invalid dim") ;

    char direction [LEN+2] ;
    gbmx_mxstring_to_string (direction, LEN, pargin [3], "direction") ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;

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
        else ERROR ("unsupported type", GrB_DOMAIN_MISMATCH) ;
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
        else ERROR ("unsupported type", GrB_DOMAIN_MISMATCH) ;
    }
    else
    { 
        ERROR2 ("unrecognized direction: %s", direction, GrB_INVALID_VALUE) ;
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

    uint64_t nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    OK (GxB_Matrix_new_arena (&C, type, nrows, ncols, arena, arena)) ;
    if (nargout > 1)
    { 
        OK (GxB_Matrix_new_arena (&P, GrB_INT64, nrows, ncols, arena, arena)) ;
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
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    OK (gb_export (C_opaque, &C, KIND_GRB, ghb, err)) ;
    if (nargout > 1)
    { 
        OK (gb_export (P_opaque, &P, KIND_GRB, ghb, err)) ;
    }
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
        if (nargout > 1)
        { 
            pargout [1] = gbmx_export_grb_mxstruct (&P) ;
        }
    }

    gb_wrapup ( ) ;
}

