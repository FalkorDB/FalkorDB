//------------------------------------------------------------------------------
// gbextracttuples: extract all entries from a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// [I J X] = GrB.extracttuples (A)
// [I J X] = GrB.extracttuples (A, desc)

// The desciptor is optional.  If present, it must be a struct.

// desc.base = 'zero-based':    I and J are returned as 0-based integer indices
// desc.base = 'one-based int': I and J are returned as 1-based integer indices
// desc.base = 'one-based':     I and J are returned as 1-based double indices
// desc.base = 'one-based double' one-based double unless max(size(A)) >
//                              flintmax, in which case 'one-based int' is used.
// desc.base = 'default':       'one-based int'

#include "gb_interface.h"

#define USAGE "usage: [I,J,X] = GrB.extracttuples (A, desc)"

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

    gb_usage (nargin >= 1 && nargin <= 2 && nargout <= 3, USAGE) ;

    //--------------------------------------------------------------------------
    // get the optional descriptor
    //--------------------------------------------------------------------------

    base_enum_t base = BASE_DEFAULT ;
    kind_enum_t kind = KIND_FULL ;              // ignored
    int fmt = GxB_NO_FORMAT ;                   // ignored
    int sparsity = 0 ;                          // ignored
    GrB_Descriptor desc = NULL ;
    if (nargin > 1)
    { 
        desc = gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt,
            &sparsity, &base) ;
    }
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // get the matrix; disable burble for scalars
    //--------------------------------------------------------------------------

    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    uint64_t nrows, ncols, nvals ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    int burble ;
    bool disable_burble = (nrows <= 1 && ncols <= 1) ;
    if (disable_burble)
    { 
        OK (GrB_Global_get_INT32 (GrB_GLOBAL, &burble, GxB_BURBLE)) ;
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, false, GxB_BURBLE)) ;
    }
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    GrB_Type xtype ;
    OK (GxB_Matrix_type (&xtype, A)) ;

    //--------------------------------------------------------------------------
    // determine what to extract
    //--------------------------------------------------------------------------

    bool extract_I = true ;
    bool extract_J = (nargout > 1) ;
    bool extract_X = (nargout > 2) ;

    //--------------------------------------------------------------------------
    // create empty GrB_Vectors for I, J, and X
    //--------------------------------------------------------------------------

    GrB_Vector I = NULL, J = NULL, X = NULL ;
    if (extract_I) OK (GrB_Vector_new (&I, GrB_UINT64, 0)) ;
    if (extract_J) OK (GrB_Vector_new (&J, GrB_UINT64, 0)) ;
    if (extract_X) OK (GrB_Vector_new (&X, xtype, 0)) ;

    //--------------------------------------------------------------------------
    // extract the tuples from A into I, J, and X
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_extractTuples_Vector (I, J, X, A, NULL)) ;

    //--------------------------------------------------------------------------
    // determine if 1 must be added to the indices
    //--------------------------------------------------------------------------

    int base_offset = (base == BASE_0_INT) ? 0 : 1 ;

    //--------------------------------------------------------------------------
    // return I to MATLAB
    //--------------------------------------------------------------------------

    void *x = NULL ;
    uint64_t size = 0 ;
    int ignore = 0 ;
    GrB_Type type = NULL ;
    GrB_Vector T = NULL ;

    if (extract_I)
    { 
        if (base == BASE_1_DOUBLE && nrows <= FLINTMAX)
        { 
            // I = (double) (I + 1)
            OK (GrB_Vector_new (&T, GrB_FP64, nvals)) ;
            OK (GrB_Vector_apply_BinaryOp2nd_FP64 (T, NULL, NULL,
                GrB_PLUS_FP64, I, base_offset, NULL)) ;
            OK (GrB_Vector_free (&I)) ;
            I = T ;
        }
        else if (base_offset != 0)
        { 
            // I = I+1, as a int64 or int32 vector
            OK (GrB_Vector_apply_BinaryOp2nd_UINT64 (I, NULL, NULL,
                GrB_PLUS_UINT64, I, 1, NULL)) ;
        }
        OK (GxB_Vector_unload (I, &x, &type, &nvals, &size, &ignore, NULL)) ;
        if (type == GrB_UINT32) type = GrB_INT32 ;
        if (type == GrB_UINT64) type = GrB_INT64 ;
        pargout [0] = gb_export_to_mxfull (&x, nvals, 1, type) ;
        OK (GrB_Vector_free (&I)) ;
    }

    //--------------------------------------------------------------------------
    // return J to MATLAB
    //--------------------------------------------------------------------------

    if (extract_J)
    { 
        if (base == BASE_1_DOUBLE && ncols <= FLINTMAX)
        { 
            // J = (double) (J + 1)
            OK (GrB_Vector_new (&T, GrB_FP64, nvals)) ;
            OK (GrB_Vector_apply_BinaryOp2nd_FP64 (T, NULL, NULL,
                GrB_PLUS_FP64, J, base_offset, NULL)) ;
            OK (GrB_Vector_free (&J)) ;
            J = T ;
        }
        else if (base_offset != 0)
        { 
            // J = J+1, as a int64 or int32 vector
            OK (GrB_Vector_apply_BinaryOp2nd_UINT64 (J, NULL, NULL,
                GrB_PLUS_UINT64, J, 1, NULL)) ;
        }
        OK (GxB_Vector_unload (J, &x, &type, &nvals, &size, &ignore, NULL)) ;
        if (type == GrB_UINT32) type = GrB_INT32 ;
        if (type == GrB_UINT64) type = GrB_INT64 ;
        pargout [1] = gb_export_to_mxfull (&x, nvals, 1, type) ;
        OK (GrB_Vector_free (&J)) ;
    }

    //--------------------------------------------------------------------------
    // return X to MATLAB
    //--------------------------------------------------------------------------

    if (extract_X)
    { 
        OK (GxB_Vector_unload (X, &x, &xtype, &nvals, &size, &ignore, NULL)) ;
        pargout [2] = gb_export_to_mxfull (&x, nvals, 1, xtype) ;
        OK (GrB_Vector_free (&X)) ;
    }

    //--------------------------------------------------------------------------
    // restore burble and return result
    //--------------------------------------------------------------------------

    if (disable_burble)
    { 
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, burble, GxB_BURBLE)) ;
    }
    GrB_Matrix_free (&A) ;
    gb_wrapup ( ) ;
}

