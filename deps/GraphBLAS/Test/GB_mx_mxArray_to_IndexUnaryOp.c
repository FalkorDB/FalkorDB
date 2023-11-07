//------------------------------------------------------------------------------
// GB_mx_mxArray_to_IndexUnaryOp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Convert a built-in string or struct to a built-in GraphBLAS IndexUnaryOp.
// The mxArray is either a struct containing two terms: opname (an operator
// name), and an optional built-in string optype (a string, 'logical',
// 'double', etc).  If not present, the default type is used (provided on
// input).
//
// op = 'tril' ;    % GrB_TRIL operator, no type needed
//
// op.opname = 'rowindex' ; op.type = 'int32' ; % the GrB_ROWINDEX_INT32
// operator.

#include "GB_mex.h"

bool GB_mx_mxArray_to_IndexUnaryOp      // true if successful, false otherwise
(
    GrB_IndexUnaryOp *op_handle,        // the binary op
    const mxArray *op_builtin,          // built-in version of op
    const char *name,                   // name of the argument
    const GrB_Type default_optype       // default operator type
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
        // look for op.optype
        fieldnumber = mxGetFieldNumber (op_builtin, "optype") ;
        if (fieldnumber >= 0)
        {
            optype_mx = mxGetFieldByNumber (op_builtin, 0, fieldnumber) ;
        }
    }
    else
    {
        // op must be a string.  default type will be used
        opname_mx = op_builtin ;
    }

    // find the corresponding built-in GraphBLAS operator
    GrB_IndexUnaryOp op ;
    if (!GB_mx_string_to_IndexUnaryOp (&op, default_optype, opname_mx,
        optype_mx))
    {
        return (false) ;
    }

    // return the op
    ASSERT_INDEXUNARYOP_OK (op, name, GB0) ;
    (*op_handle) = op ;
    return (true) ;
}

