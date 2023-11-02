//------------------------------------------------------------------------------
// gbextractvalues: extract all entries from a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// X = gbextractvalues (A)

#include "gb_interface.h"

#define USAGE "usage: X = GrB.extractvalues (A)"

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

    gb_usage (nargin == 1 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    GrB_Index nvals ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    GrB_Type xtype ;
    OK (GxB_Matrix_type (&xtype, A)) ;
    GrB_Index s = MAX (nvals, 1) ;

    //--------------------------------------------------------------------------
    // extract the tuples
    //--------------------------------------------------------------------------

    if (xtype == GrB_BOOL)
    { 
        bool *X = mxMalloc (s * sizeof (bool)) ;
        OK (GrB_Matrix_extractTuples_BOOL (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_BOOL) ;
    }
    else if (xtype == GrB_INT8)
    { 
        int8_t *X = mxMalloc (s * sizeof (int8_t)) ;
        OK (GrB_Matrix_extractTuples_INT8 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_INT8) ;
    }
    else if (xtype == GrB_INT16)
    { 
        int16_t *X = mxMalloc (s * sizeof (int16_t)) ;
        OK (GrB_Matrix_extractTuples_INT16 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_INT16) ;
    }
    else if (xtype == GrB_INT32)
    { 
        int32_t *X = mxMalloc (s * sizeof (int32_t)) ;
        OK (GrB_Matrix_extractTuples_INT32 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_INT32) ;
    }
    else if (xtype == GrB_INT64)
    { 
        int64_t *X = mxMalloc (s * sizeof (int64_t)) ;
        OK (GrB_Matrix_extractTuples_INT64 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_INT64) ;
    }
    else if (xtype == GrB_UINT8)
    { 
        uint8_t *X = mxMalloc (s * sizeof (uint8_t)) ;
        OK (GrB_Matrix_extractTuples_UINT8 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_UINT8) ;
    }
    else if (xtype == GrB_UINT16)
    { 
        uint16_t *X = mxMalloc (s * sizeof (uint16_t)) ;
        OK (GrB_Matrix_extractTuples_UINT16 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_UINT16) ;
    }
    else if (xtype == GrB_UINT32)
    { 
        uint32_t *X = mxMalloc (s * sizeof (uint32_t)) ;
        OK (GrB_Matrix_extractTuples_UINT32 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_UINT32) ;
    }
    else if (xtype == GrB_UINT64)
    { 
        uint64_t *X = mxMalloc (s * sizeof (uint64_t)) ;
        OK (GrB_Matrix_extractTuples_UINT64 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_UINT64) ;
    }

    else if (xtype == GrB_FP32)
    { 
        float *X = mxMalloc (s * sizeof (float)) ;
        OK (GrB_Matrix_extractTuples_FP32 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_FP32) ;
    }
    else if (xtype == GrB_FP64)
    { 
        double *X = mxMalloc (s * sizeof (double)) ;
        OK (GrB_Matrix_extractTuples_FP64 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GrB_FP64) ;
    }
    else if (xtype == GxB_FC32)
    { 
        GxB_FC32_t *X = mxMalloc (s * sizeof (GxB_FC32_t)) ;
        OK (GxB_Matrix_extractTuples_FC32 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GxB_FC32) ;
    }
    else if (xtype == GxB_FC64)
    { 
        GxB_FC64_t *X = mxMalloc (s * sizeof (GxB_FC64_t)) ;
        OK (GxB_Matrix_extractTuples_FC64 (NULL, NULL, X, &nvals, A)) ;
        pargout [0] = gb_export_to_mxfull ((void **) &X, nvals, 1, GxB_FC64) ;
    }
    else
    {
        ERROR ("unsupported type") ;
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    GB_WRAPUP ;
}

