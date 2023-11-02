//------------------------------------------------------------------------------
// gbbuild: build a GraphBLAS matrix or a built-in sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// A = gbbuild (I, J, X)
// A = gbbuild (I, J, X, desc)
// A = gbbuild (I, J, X, m, desc)
// A = gbbuild (I, J, X, m, n, desc)
// A = gbbuild (I, J, X, m, n, dup, desc) ;
// A = gbbuild (I, J, X, m, n, dup, type, desc) ;

// X and either I or J may be a scalars, in which case they are effectively
// expanded so that they all have the same length.  X is only implicitly
// expanded if A is built as an iso matrix.

// m and n default to the largest index in I and J, respectively.

// dup is a string that defaults to 'plus.xtype' where xtype is the type of X.
// If dup is given by without a type,  type of dup defaults to the type of X.

// If dup is the empty string '' then any duplicates result in an error.
// If dup is the string 'ignore' then duplicates are ignored.

// type is a string that defines is the type of A, which defaults to the type
// of X.

// If X is a scalar, and dup is '1st', '2nd', 'any', 'min', 'max',
// 'pair' (same as 'oneb'),
// 'or', 'and', 'bitor', or 'bitand', then GxB_Matrix_build_Scalar is used and
// A is built as an iso matrix.  X is not explicitly expanded. This is
// much faster than when using the default dup.

// The descriptor is optional; if present, it must be the last input parameter.
// desc.kind is the only part used from the descriptor, and it defaults to
// desc.kind = 'GrB'.

#include "gb_interface.h"

#define USAGE "usage: A = GrB.build (I, J, X, m, n, dup, type, desc)"

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

    gb_usage (nargin >= 3 && nargin <= 8 && nargout <= 2, USAGE) ;

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    base_enum_t base ;
    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    int sparsity ;
    GrB_Descriptor desc = NULL ;
    desc = gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt,
        &sparsity, &base) ;

    // if present, remove the descriptor from consideration
    if (desc != NULL) nargin-- ;

    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // get I and J
    //--------------------------------------------------------------------------

    GrB_Index ni, nj ;
    bool I_allocated, J_allocated ;
    int64_t Imax = -1, Jmax = -1 ;

    GrB_Index *I = (GrB_Index *) gb_mxarray_to_list (pargin [0], base,
        &I_allocated, (int64_t *) &ni, &Imax) ;

    GrB_Index *J = (GrB_Index *) gb_mxarray_to_list (pargin [1], base,
        &J_allocated, (int64_t *) &nj, &Jmax) ;

    //--------------------------------------------------------------------------
    // get X
    //--------------------------------------------------------------------------

    const mxArray *Xm = pargin [2] ;
    GrB_Type xtype = gb_mxarray_type (Xm) ;
    GrB_Index nx = mxGetNumberOfElements (Xm) ;

    //--------------------------------------------------------------------------
    // check the sizes of I, J, and X, and the type of X
    //--------------------------------------------------------------------------

    GrB_Index nvals = MAX (ni, nj) ;
    nvals = MAX (nvals, nx) ;

    if (!(ni == 1 || ni == nvals) ||
        !(nj == 1 || nj == nvals) ||
        !(nx == 1 || nx == nvals))
    { 
        ERROR ("I, J, and X must have the same length") ;
    }

    CHECK_ERROR (!(mxIsNumeric (Xm) || mxIsLogical (Xm)),
        "X must be a numeric or logical array") ;
    CHECK_ERROR (mxIsSparse (Xm), "X cannot be sparse") ;

    //--------------------------------------------------------------------------
    // expand any scalars in I and J (but not X)
    //--------------------------------------------------------------------------

    if (ni == 1 && ni < nvals)
    { 
        GrB_Index *I2 = (GrB_Index *) mxMalloc (nvals * sizeof (GrB_Index)) ;
        GB_helper8 ((GB_void *) I2, (GB_void *) I, nvals, sizeof (GrB_Index)) ;
        if (I_allocated) gb_mxfree ((void **) (&I)) ;
        I_allocated = true ;
        I = I2 ;
    }

    if (nj == 1 && nj < nvals)
    { 
        GrB_Index *J2 = (GrB_Index *) mxMalloc (nvals * sizeof (GrB_Index)) ;
        GB_helper8 ((GB_void *) J2, (GB_void *) J, nvals, sizeof (GrB_Index)) ;
        if (J_allocated) gb_mxfree ((void **) (&J)) ;
        J_allocated = true ;
        J = J2 ;
    }

    //--------------------------------------------------------------------------
    // get m and n if present
    //--------------------------------------------------------------------------

    GrB_Index nrows = 0, ncols = 0 ;

    if (nargin < 4)
    {
        // nrows = max entry in I + 1
        if (Imax > -1)
        { 
            // Imax already computed
            nrows = Imax ;
        }
        else
        { 
            // nrows = max entry in I+1
            bool ok = GB_helper4 (I, ni, &nrows) ;
            CHECK_ERROR (!ok, "out of memory") ;
        }
    }
    else
    { 
        // m is provided on input
        nrows = gb_mxget_uint64_scalar (pargin [3], "m") ;
    }

    if (nargin < 5)
    {
        if (Jmax > -1)
        { 
            // Jmax already computed
            ncols = Jmax ;
        }
        else
        { 
            // ncols = max entry in J+1
            bool ok = GB_helper4 (J, nj, &ncols) ;
            CHECK_ERROR (!ok, "out of memory") ;
        }
    }
    else
    { 
        // n is provided on input
        ncols = gb_mxget_uint64_scalar (pargin [4], "n") ;
    }

    //--------------------------------------------------------------------------
    // get the dup operator
    //--------------------------------------------------------------------------

    // default_dup: if dup does not appear as a parameter
    bool default_dup = (nargin < 6) ;
    GrB_BinaryOp dup = GxB_IGNORE_DUP ;
    if (!default_dup)
    { 
        dup = gb_mxstring_to_binop (pargin [5], xtype, xtype) ;
    }

    // if dup defaults to plus.xtype, below, or GrB_LOR for boolean

    bool nice_iso_dup ;
    if (default_dup)
    {
        // dup will be GrB_LOR which is nice for an iso build.  For all other
        // types, the dup is plus, which is not nice.
        nice_iso_dup = (xtype == GrB_BOOL) ;
    }
    else if (dup == NULL || dup == GxB_IGNORE_DUP)
    {
        // if X is a scalar and dup is '' (NULL) or 'ignore' (GxB_IGNORE_DUP),
        // then dup is a nice iso dup.
        nice_iso_dup = true ;
    }
    else
    {
        // parse dup to see if it will build an iso matrix if X is a scalar
        #define LEN 256
        char sdup [LEN+2] ;
        gb_mxstring_to_string (sdup, LEN, pargin [5], "dup") ;
        int32_t position [2] ;
        gb_find_dot (position, sdup) ;
        if (position [0] >= 0) sdup [position [0]] = '\0' ;
        nice_iso_dup =
            MATCH (sdup, "1st") || MATCH (sdup, "first" ) ||
            MATCH (sdup, "2nd") || MATCH (sdup, "second") ||
            MATCH (sdup, "any") ||
            MATCH (sdup, "min") || MATCH (sdup, "max"   ) ||
            MATCH (sdup, "||" ) || MATCH (sdup, "|"     ) ||
            MATCH (sdup, "&&" ) || MATCH (sdup, "&"     ) ||
            MATCH (sdup, "or" ) || MATCH (sdup, "bitor" ) ||
            MATCH (sdup, "and") || MATCH (sdup, "bitand") ||
            MATCH (sdup, "lor") || MATCH (sdup, "land"  ) ;
    }

    //--------------------------------------------------------------------------
    // get the output matrix type
    //--------------------------------------------------------------------------

    GrB_Type type = NULL ;
    if (nargin > 6)
    { 
        type = gb_mxstring_to_type (pargin [6]) ;
        CHECK_ERROR (type == NULL, "unknown type") ;
    }
    else
    { 
        type = xtype ;
    }

    //--------------------------------------------------------------------------
    // build the matrix
    //--------------------------------------------------------------------------

    fmt = gb_get_format (nrows, ncols, NULL, NULL, fmt) ;
    sparsity = gb_get_sparsity (NULL, NULL, sparsity) ;
    GrB_Matrix A = gb_new (type, nrows, ncols, fmt, sparsity) ;

    void *X2 = NULL ;
    bool X_is_scalar = (nx == 1 && nx < nvals) ;
    bool iso_build = X_is_scalar && nice_iso_dup ;

    // mxGetData is used instead of the MATLAB-recommended mxGetDoubles, etc,
    // because mxGetData works best for Octave, and it works fine for MATLAB
    // since GraphBLAS requires R2018a with the interleaved complex data type.

    if (iso_build)
    {
        // build an iso matrix, with no dup operator (dup is GxB_IGNORE_DUP)
        GrB_Scalar x_scalar = (GrB_Scalar) gb_get_shallow (Xm) ;
        OK1 (A, GxB_Matrix_build_Scalar (A, I, J, x_scalar, nvals)) ;
        OK (GrB_Scalar_free (&x_scalar)) ;
    }
    else if (xtype == GrB_BOOL)
    { 
        bool empty = 0 ;
        bool *X = (nvals == 0) ? &empty : ((bool *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_LOR ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (bool)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof (bool)) ;
            X = (bool *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_BOOL (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_INT8)
    { 
        int8_t empty = 0 ;
        int8_t *X = (nvals == 0) ? &empty : ((int8_t *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_PLUS_INT8 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (int8_t)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof (int8_t)) ;
            X = (int8_t *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_INT8 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_INT16)
    { 
        int16_t empty = 0 ;
        int16_t *X = (nvals == 0) ? &empty : ((int16_t *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_PLUS_INT16 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (int16_t)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof (int16_t));
            X = (int16_t *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_INT16 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_INT32)
    { 
        int32_t empty = 0 ;
        int32_t *X = (nvals == 0) ? &empty : ((int32_t *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_PLUS_INT32 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (int32_t)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof (int32_t));
            X = (int32_t *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_INT32 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_INT64)
    { 
        int64_t empty = 0 ;
        int64_t *X = (nvals == 0) ? &empty : ((int64_t *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_PLUS_INT64 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (int64_t)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof (int64_t));
            X = (int64_t *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_INT64 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_UINT8)
    { 
        uint8_t empty = 0 ;
        uint8_t *X = (nvals == 0) ? &empty : ((uint8_t *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_PLUS_UINT8 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (uint8_t)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof (uint8_t));
            X = (uint8_t *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_UINT8 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_UINT16)
    { 
        uint16_t empty = 0 ;
        uint16_t *X = (nvals == 0) ? &empty : ((uint16_t *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_PLUS_UINT16 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (uint16_t)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof(uint16_t));
            X = (uint16_t *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_UINT16 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_UINT32)
    { 
        uint32_t empty = 0 ;
        uint32_t *X = (nvals == 0) ? &empty : ((uint32_t *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_PLUS_UINT32 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (uint32_t)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof(uint32_t));
            X = (uint32_t *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_UINT32 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_UINT64)
    { 
        uint64_t empty = 0 ;
        uint64_t *X = (nvals == 0) ? &empty : ((uint64_t *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_PLUS_UINT64 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (uint64_t)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof(uint64_t));
            X = (uint64_t *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_UINT64 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_FP32)
    { 
        float empty = 0 ;
        float *X = (nvals == 0) ? &empty : ((float *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_PLUS_FP32 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (float)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof (float)) ;
            X = (float *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_FP32 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_FP64)
    { 
        double empty = 0 ;
        double *X = (nvals == 0) ? &empty : ((double *) mxGetData (Xm)) ;
        if (default_dup) dup = GrB_PLUS_FP64 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (double)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals, sizeof (double)) ;
            X = (double *) X2 ;
        }
        OK1 (A, GrB_Matrix_build_FP64 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GxB_FC32)
    { 
        GxB_FC32_t empty = GxB_CMPLXF (0,0) ;
        GxB_FC32_t *X = &empty ;
        if (nvals > 0) X = (GxB_FC32_t *) mxGetData (Xm) ;
        if (default_dup) dup = GxB_PLUS_FC32 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (GxB_FC32_t)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals,
                sizeof (GxB_FC32_t)) ;
            X = (GxB_FC32_t *) X2 ;
        }
        OK1 (A, GxB_Matrix_build_FC32 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GxB_FC64)
    { 
        GxB_FC64_t empty = GxB_CMPLX (0,0) ;
        GxB_FC64_t *X = &empty ;
        if (nvals > 0) X = (GxB_FC64_t *) mxGetData (Xm) ;
        if (default_dup) dup = GxB_PLUS_FC64 ;
        if (X_is_scalar)
        { 
            X2 = mxMalloc (nvals * sizeof (GxB_FC64_t)) ;
            GB_helper8 ((GB_void *) X2, (GB_void *) X, nvals,
                sizeof (GxB_FC64_t)) ;
            X = (GxB_FC64_t *) X2 ;
        }
        OK1 (A, GxB_Matrix_build_FC64 (A, I, J, X, nvals, dup)) ;
    }
    else
    {
        ERROR ("unsupported type") ;
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    if (X2 != NULL ) gb_mxfree ((void **) (&X2)) ;
    if (I_allocated) gb_mxfree ((void **) (&I)) ;
    if (J_allocated) gb_mxfree ((void **) (&J)) ;

    //--------------------------------------------------------------------------
    // export the output matrix A
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&A, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    GB_WRAPUP ;
}

