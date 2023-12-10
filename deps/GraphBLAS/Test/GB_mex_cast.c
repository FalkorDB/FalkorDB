//------------------------------------------------------------------------------
// GB_mex_cast: cast a built-in array using C-style casting rules
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage: C = GB_mex_cast (X, type) casts the dense array X to given type using
// C-style typecasting rules instead of built-in rules.

#include "GB_mex.h"

#define USAGE "C = GB_mex_cast (X, type, cover)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;

    // do not get coverage counts unless the 3rd arg is present
    bool do_cover = (nargin == 3) ;
    bool malloc_debug = GB_mx_get_global (do_cover) ;

    // check inputs
    if (nargout > 2 || nargin < 1 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    if (mxIsSparse (pargin [0]))
    {
        mexErrMsgTxt ("X must be dense") ;
    }

    // get X
    GB_void *X ;
    int64_t nrows, ncols ;
    GrB_Type xtype ;
    GB_mx_mxArray_to_array (pargin [0], &X, &nrows, &ncols, &xtype) ;
    if (xtype == NULL)
    {
        mexErrMsgTxt ("X must be numeric") ;
    }

    // get the type for C, default is same as X
    GrB_Type ctype = GB_mx_string_to_Type (PARGIN (1), xtype) ;
    if (ctype == NULL)
    {
        mexErrMsgTxt ("C must be numeric") ;
    }

    // create C
    pargout [0] = GB_mx_create_full (nrows, ncols, ctype) ;
    if (ctype == Complex) ctype = GxB_FC64 ;
    if (xtype == Complex) xtype = GxB_FC64 ;
    GB_void *C = mxGetData (pargout [0]) ;

    // cast the data from X to C
    int64_t cnz = nrows*ncols ;
    if (C == NULL && cnz > 0) mexErrMsgTxt ("C is NULL!\n") ;
    if (ctype == xtype)
    {
        memcpy (C, X, cnz * xtype->size) ;
    }
    else
    {
        // create a shallow cnz-by-1 matrix T to wrap the array X
        T = NULL ;
        void *Tx = X ;
        GrB_Index nrows = cnz, ncols = 1, Tx_size = cnz * xtype->size ;
        GxB_Matrix_import_FullC (&T, xtype, nrows, ncols, &Tx, Tx_size, false, NULL) ;
        // GB_cast_array (C, ctype->code, X, xtype->code, NULL, cnz, 1) ;
        GB_cast_array (C, ctype->code, T, 1) ;
        bool iso ;
        GxB_Matrix_export_FullC (&T, &xtype, &nrows, &ncols, &Tx, &Tx_size, &iso, NULL) ;
    }

    GB_mx_put_global (do_cover) ;
}

