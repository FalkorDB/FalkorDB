//------------------------------------------------------------------------------
// GB_mx_mxArray_to_UnaryOp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Convert a built-in string or struct to a built-in GraphBLAS UnaryOp.  The
// mxArray is either a struct containing two terms: opname (an operator name),
// and an optional built-in string optype (a string, 'logical', 'double', etc).
// If not present, the default type is used (provided on input).
//
// That is:
// op = 'identity' ;    % the GrB_IDENTITY_*, type is default_optype.
//
// op.opname = 'minv' ; op.type = 'int8' ; % the GrB_MINV_INT8 operator.

#include "GB_mex.h"

bool GB_mx_mxArray_to_UnaryOp           // true if successful
(
    GrB_UnaryOp *op_handle,             // the unary op
    const mxArray *op_builtin,          // built-in version of op
    const char *name,                   // name of the argument
    const GrB_Type default_optype,      // default operator type
    const bool user_complex             // if true, use user-defined Complex
)
{
    (*op_handle) = NULL ;
    const mxArray *opname_mx = NULL, *optype_mx = NULL ;

    if (op_builtin == NULL || mxIsEmpty (op_builtin))
    {
        // op is not present; defaults will be used
        ;
    }
    else if (mxIsStruct (op_builtin))
    {
        // look for op.opname
        int fieldnumber = mxGetFieldNumber (op_builtin, "opname") ;
        if (fieldnumber >= 0)
        {
            opname_mx = mxGetFieldByNumber (op_builtin, 0, fieldnumber) ;
        }
        // look for op.type
        fieldnumber = mxGetFieldNumber (op_builtin, "optype") ;
        if (fieldnumber >= 0)
        {
            optype_mx = mxGetFieldByNumber (op_builtin, 0, fieldnumber) ;
        }
    }
    else if (mxIsChar (op_builtin))
    {
        // op is a string
        opname_mx = op_builtin ;
    }
    else
    {
        mexWarnMsgIdAndTxt ("GB:warn",
            "unrecognized unary op: must be string or struct") ;
        return (false) ;
    }

    // find the corresponding built-in GraphBLAS operator
    GrB_UnaryOp op ;
    if (!GB_mx_string_to_UnaryOp (&op, default_optype,
        opname_mx, optype_mx, user_complex))
    {
        mexWarnMsgIdAndTxt ("GB:warn", "unary op failed") ;
        return (false) ;
    }

    // return the op
    ASSERT_UNARYOP_OK (op, name, GB0) ;
    (*op_handle) = op ;
    return (true) ;
}

