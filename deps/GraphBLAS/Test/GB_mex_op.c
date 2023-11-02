//------------------------------------------------------------------------------
// GB_mex_op: apply a built-in GraphBLAS operator to built-in arrays
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// Z = GB_mex_op (op, X, Y)
// Z = GB_mex_op (op, X)

// Apply a built-in GraphBLAS operator or a user-defined Complex operator to
// one or two arrays X and Y of any built-in logical or numeric type.  X and Y
// are first typecasted into the x and y operand types of the op.  The output Z
// has the same type as the z type of the op.

#include "GB_mex.h"

#define USAGE "Z = GB_mex_op (opname, X, Y, cover)"

#define FREE_ALL                        \
{                                       \
    GB_mx_put_global (do_cover) ;       \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GB_void *X = NULL, *Y = NULL, *Z = NULL ;
    GrB_Type X_type = NULL, Y_type = NULL ;
    int64_t nrows = 0, ncols = 0, nx = 0, ny = 0, nrows2 = 0, ncols2 = 0 ;
    size_t Y_size = 1 ;

    bool do_cover = (nargin == 4) ;
    bool malloc_debug = GB_mx_get_global (do_cover) ;

    // if Y is char and cover present, treat as if nargin == 2
    if (do_cover)
    {
        if (mxIsChar (pargin [2]))
        {
            nargin = 2 ;
        }
    }

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (nargout > 1 || nargin < 2 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    //--------------------------------------------------------------------------
    // get op; default type is the same type as X
    //--------------------------------------------------------------------------

    GrB_UnaryOp  op1 = NULL ;
    GrB_BinaryOp op2 = NULL ;
    GrB_Type op_ztype = NULL, op_xtype, op_ytype ;
    size_t   op_zsize, op_xsize, op_ysize ;
    GrB_Type xtype = GB_mx_Type (pargin [1]) ;

    // check for complex case
    bool XisComplex = mxIsComplex (pargin [1]) ;
    bool YisComplex = (nargin > 2) ? mxIsComplex (pargin [2]) : false ;
    bool user_complex = (Complex != GxB_FC64) && (XisComplex || YisComplex) ;

    if (nargin > 2)
    {
        // get a binary op
        if (!GB_mx_mxArray_to_BinaryOp (&op2, pargin [0], "GB_mex_op",
            xtype, user_complex) || op2 == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("binary op missing") ;
        }
        op_ztype = op2->ztype ; op_zsize = op_ztype->size ;
        op_xtype = op2->xtype ; op_xsize = op_xtype->size ;
        op_ytype = op2->ytype ; op_ysize = op_ytype->size ;
        ASSERT_BINARYOP_OK (op2, "binary op", GB0) ;
        if (GB_OP_IS_POSITIONAL (op2))
        { 
            FREE_ALL ;
            mexErrMsgTxt ("binary positional op not supported") ;
        }
    }
    else
    {
        // get a unary op
        if (!GB_mx_mxArray_to_UnaryOp (&op1, pargin [0], "GB_mex_op",
            xtype, user_complex) || op1 == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("unary op missing") ;
        }
        op_ztype = op1->ztype ; op_zsize = op_ztype->size ;
        op_xtype = op1->xtype ; op_xsize = op_xtype->size ;
        op_ytype = NULL       ; op_ysize = 1 ;
        ASSERT_UNARYOP_OK (op1, "unary op", GB0) ;
        if (GB_OP_IS_POSITIONAL (op1))
        { 
            FREE_ALL ;
            mexErrMsgTxt ("unary positional op not supported") ;
        }
    }

    ASSERT_TYPE_OK (op_ztype, "Z type", GB0) ;

    //--------------------------------------------------------------------------
    // get X
    //--------------------------------------------------------------------------

    GB_mx_mxArray_to_array (pargin [1], &X, &nrows, &ncols, &X_type) ;
    nx = nrows * ncols ;
    if (X_type == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("X must be numeric") ;
    }
    ASSERT_TYPE_OK (X_type, "X type", GB0) ;
    size_t X_size = X_type->size ;

    if (!GB_Type_compatible (op_xtype, X_type))
    {
        FREE_ALL ;
        mexErrMsgTxt ("op xtype not compatible with X") ;
    }

    //--------------------------------------------------------------------------
    // get Y
    //--------------------------------------------------------------------------

    if (nargin > 2)
    {
        GB_mx_mxArray_to_array (pargin [2], &Y, &nrows2, &ncols2, &Y_type) ;
        ny = nrows2 * ncols2 ;
        if (nrows2 != nrows || ncols2 != ncols)
        {
            FREE_ALL ;
            mexErrMsgTxt ("X and Y must be the same size") ;
        }
        if (Y_type == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("Y must be numeric") ;
        }
        ASSERT_TYPE_OK (Y_type, "Y type", GB0) ;
        Y_size = Y_type->size ;

        if (!GB_Type_compatible (op_ytype, Y_type))
        {
            FREE_ALL ;
            mexErrMsgTxt ("op ytype not compatible with Y") ;
        }
    }

    //--------------------------------------------------------------------------
    // create Z of the same type as op_ztype
    //--------------------------------------------------------------------------

    pargout [0] = GB_mx_create_full (nrows, ncols, op_ztype) ;
    Z = mxGetData (pargout [0]) ;

    //--------------------------------------------------------------------------
    // get scalar workspace
    //--------------------------------------------------------------------------

    char xwork [GB_VLA (op_xsize)] ;
    char ywork [GB_VLA (op_ysize)] ;

    GB_cast_function cast_X = GB_cast_factory (op_xtype->code, X_type->code) ;

    //--------------------------------------------------------------------------
    // do the op
    //--------------------------------------------------------------------------

    if (nargin > 2)
    {
        // Z = f (X,Y)
        GxB_binary_function f_binary = op2->binop_function ;

        GB_cast_function cast_Y = GB_cast_factory (op_ytype->code,Y_type->code);
        for (int64_t k = 0 ; k < nx ; k++)
        {
            cast_X (xwork, X +(k*X_size), X_size) ;
            cast_Y (ywork, Y +(k*Y_size), Y_size) ;
            // printf ("x: ")   ; GB_code_check (op_xtype->code,xwork,3,NULL) ;
            // printf ("\ny: ") ; GB_code_check (op_ytype->code,ywork,3,NULL) ;
            f_binary (Z +(k*op_zsize), xwork, ywork) ;
            // printf ("\nz: ") ; GB_code_check (op_ztype->code,
            //                    Z +(k*op_zsize), 3, NULL) ; printf ("\n") ;
        }

    }
    else
    {
        // Z = f (X)
        GxB_unary_function f_unary = op1->unop_function ;
        for (int64_t k = 0 ; k < nx ; k++)
        {
            cast_X (xwork, X +(k*X_size), X_size) ;
            f_unary (Z +(k*op_zsize), xwork) ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return
    //--------------------------------------------------------------------------

    FREE_ALL ;
}

