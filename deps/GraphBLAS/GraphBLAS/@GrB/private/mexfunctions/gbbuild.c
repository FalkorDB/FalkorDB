//------------------------------------------------------------------------------
// gbbuild: build a GraphBLAS matrix or a built-in sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

//------------------------------------------------------------------------------
// gb_get_scalar: x = find (V, 'first')
//------------------------------------------------------------------------------

static GrB_Scalar gb_get_scalar (GrB_Vector V, GrB_Type type)
{
    // get the first entry from a vector V
    GrB_Scalar x ;
    GrB_Vector T ;
    OK (GrB_Scalar_new (&x, type)) ;
    OK (GrB_Vector_new (&T, type, 0)) ;
    OK (GxB_Vector_extractTuples_Vector (NULL, T, V, NULL)) ;
    OK (GrB_Vector_extractElement_Scalar (x, T, 0)) ;
    OK (GrB_Vector_free (&T)) ;
    return (x) ;
}

//------------------------------------------------------------------------------
// gb_expand: V (1:nvals) = V (1)
//------------------------------------------------------------------------------

static void gb_expand (GrB_Vector *V, GrB_Type type, uint64_t nvals)
{
    // get the single entry from the input vector V, and then free it
    GrB_Scalar x = gb_get_scalar (*V, type) ;
    OK (GrB_Vector_free (V)) ;
    // expand the scalar back into V, expanding V to length nvals
    OK (GrB_Vector_new (V, type, nvals)) ;
    OK (GxB_Vector_assign_Scalar_Vector (*V, NULL, NULL, x, NULL, NULL)) ;
    OK (GrB_Scalar_free (&x)) ;
}

//------------------------------------------------------------------------------
// gbbuild mexFunction
//------------------------------------------------------------------------------

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
    int fmt ;
    int sparsity ;
    GrB_Descriptor desc = NULL ;
    desc = gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt,
        &sparsity, &base) ;

    // if present, remove the descriptor from consideration
    if (desc != NULL) nargin-- ;

    OK (GrB_Descriptor_free (&desc)) ;

    int base_offset = (base == BASE_0_INT) ? 0 : 1 ;

    //--------------------------------------------------------------------------
    // get I, J, and X and their properties
    //--------------------------------------------------------------------------

    GrB_Vector I = gb_mxarray_to_list (pargin [0], base_offset) ;
    GrB_Vector J = gb_mxarray_to_list (pargin [1], base_offset) ;
    GrB_Vector X = gb_mxarray_to_list (pargin [2], 0) ;

    uint64_t ni, nj, nx ;
    OK (GrB_Vector_nvals (&ni, I)) ;
    OK (GrB_Vector_nvals (&nj, J)) ;
    OK (GrB_Vector_nvals (&nx, X)) ;

    GrB_Type xtype ;
    OK (GxB_Vector_type (&xtype, X)) ;

    uint64_t Imax = UINT64_MAX, Jmax = UINT64_MAX ;

    //--------------------------------------------------------------------------
    // check the sizes of I, J, and X, and the type of X
    //--------------------------------------------------------------------------

    uint64_t nvals = MAX (ni, nj) ;
    nvals = MAX (nvals, nx) ;

    if (!(ni == 1 || ni == nvals) ||
        !(nj == 1 || nj == nvals) ||
        !(nx == 1 || nx == nvals))
    { 
        ERROR ("I, J, and X must have the same # of entries") ;
    }

    //--------------------------------------------------------------------------
    // expand any scalars in I and J (but not yet X)
    //--------------------------------------------------------------------------

    GrB_Monoid max = GrB_MAX_MONOID_UINT64 ;

    if (ni == 1 && ni < nvals)
    { 
        if (Imax == UINT64_MAX)
        {
            OK (GrB_Vector_reduce_UINT64 (&Imax, NULL, max, I, NULL)) ;
        }
        gb_expand (&I, (Imax < UINT32_MAX) ? GrB_UINT32 : GrB_UINT64, nvals) ;
    }

    if (nj == 1 && nj < nvals)
    { 
        if (Jmax == UINT64_MAX)
        {
            OK (GrB_Vector_reduce_UINT64 (&Jmax, NULL, max, J, NULL)) ;
        }
        gb_expand (&J, (Jmax < UINT32_MAX) ? GrB_UINT32 : GrB_UINT64, nvals) ;
    }

    //--------------------------------------------------------------------------
    // get m and n if present
    //--------------------------------------------------------------------------

    uint64_t nrows = 0, ncols = 0 ;

    if (nargin < 4)
    {
        // nrows = max entry in I + 1
        if (Imax == UINT64_MAX)
        {
            OK (GrB_Vector_reduce_UINT64 (&Imax, NULL, max, I, NULL)) ;
        }
        nrows = Imax + 1 ;
    }
    else
    { 
        // m is provided on input
        nrows = gb_mxget_uint64_scalar (pargin [3], "m") ;
    }

    if (nargin < 5)
    {
        // ncols = max entry in J + 1
        if (Jmax == UINT64_MAX)
        {
            OK (GrB_Vector_reduce_UINT64 (&Jmax, NULL, max, J, NULL)) ;
        }
        ncols = Jmax + 1 ;
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

    bool nice_iso_dup = false ;
    if (default_dup)
    {
        // dup defaults to plus.xtype or GrB_LOR for boolean
        if (xtype == GrB_BOOL)
        { 
            // dup is GrB_LOR which is nice for an iso build.  For all other
            // types, the dup is plus, which is not nice.
            dup = GrB_LOR ;
            nice_iso_dup = true ;
        }
        else if (xtype == GrB_INT8)
        { 
            dup = GrB_PLUS_INT8 ;
        }
        else if (xtype == GrB_INT16)
        { 
            dup = GrB_PLUS_INT16 ;
        }
        else if (xtype == GrB_INT32)
        { 
            dup = GrB_PLUS_INT32 ;
        }
        else if (xtype == GrB_INT64)
        { 
            dup = GrB_PLUS_INT64 ;
        }
        else if (xtype == GrB_UINT8)
        { 
            dup = GrB_PLUS_UINT8 ;
        }
        else if (xtype == GrB_UINT16)
        { 
            dup = GrB_PLUS_UINT16 ;
        }
        else if (xtype == GrB_UINT32)
        { 
            dup = GrB_PLUS_UINT32 ;
        }
        else if (xtype == GrB_UINT64)
        { 
            dup = GrB_PLUS_UINT64 ;
        }
        else if (xtype == GrB_FP32)
        { 
            dup = GrB_PLUS_FP32 ;
        }
        else if (xtype == GrB_FP64)
        { 
            dup = GrB_PLUS_FP64 ;
        }
        else if (xtype == GxB_FC32)
        { 
            dup = GxB_PLUS_FC32 ;
        }
        else if (xtype == GxB_FC64)
        { 
            dup = GxB_PLUS_FC64 ;
        }
        else
        {
            ERROR ("unsupported type") ;
        }
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

    if (nvals > 0)
    {
        bool X_is_scalar = (nx == 1 && nx < nvals) ;
        bool iso_build = X_is_scalar && nice_iso_dup ;
        if (iso_build)
        {
            // build an iso matrix, with no dup operator
            GrB_Scalar x = gb_get_scalar (X, xtype) ;
            OK1 (A, GxB_Matrix_build_Scalar_Vector (A, I, J, x, NULL)) ;
            OK (GrB_Scalar_free (&x)) ;
        }
        else
        {
            // build a standard matrix from the three vectors I,J,X
            if (X_is_scalar)
            {
                // expand X from a scalar to a vector of length nvals
                gb_expand (&X, xtype, nvals) ;
            }
            OK1 (A, GxB_Matrix_build_Vector (A, I, J, X, dup, NULL)) ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    OK (GrB_Vector_free (&I)) ;
    OK (GrB_Vector_free (&J)) ;
    OK (GrB_Vector_free (&X)) ;

    //--------------------------------------------------------------------------
    // export the output matrix A
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&A, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    gb_wrapup ( ) ;
}

