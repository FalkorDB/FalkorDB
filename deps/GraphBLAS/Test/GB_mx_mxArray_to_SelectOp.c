//------------------------------------------------------------------------------
// GB_mx_mxArray_to_SelectOp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Convert a built-in string to a built-in GraphBLAS SelectOp.

#include "GB_mex.h"

bool GB_mx_mxArray_to_SelectOp          // true if successful
(
    GxB_SelectOp *handle,               // returns GraphBLAS version of op
    const mxArray *op_builtin,          // built-in version of op
    const char *name                    // name of the argument
)
{
    (*handle) = NULL ;
    const mxArray *opname_mx = NULL ;

    if (op_builtin == NULL || mxIsEmpty (op_builtin))
    {
        mexWarnMsgIdAndTxt ("GB:warn", "select op missing") ;
        return (false) ;
    }
    else if (mxIsChar (op_builtin))
    {
        // op is a string
        opname_mx = op_builtin ;
    }
    else
    {
        mexWarnMsgIdAndTxt ("GB:warn", "select op must be string") ;
        return (false) ;
    }

    // find the corresponding built-in GraphBLAS operator
    GxB_SelectOp op ;

    // get the string
    #define LEN 256
    char opname [LEN+2] ;
    int len = GB_mx_mxArray_to_string (opname, LEN, opname_mx) ;
    if (len < 0)
    {
        return (false) ;
    }

    if      (MATCH (opname, "tril"     )) { op = GxB_TRIL ; }
    else if (MATCH (opname, "triu"     )) { op = GxB_TRIU ; }
    else if (MATCH (opname, "diag"     )) { op = GxB_DIAG ; }
    else if (MATCH (opname, "offdiag"  )) { op = GxB_OFFDIAG ; }

    else if (MATCH (opname, "nonzero"  )) { op = GxB_NONZERO ; }
    else if (MATCH (opname, "eq_zero"  )) { op = GxB_EQ_ZERO ; }
    else if (MATCH (opname, "gt_zero"  )) { op = GxB_GT_ZERO ; }
    else if (MATCH (opname, "ge_zero"  )) { op = GxB_GE_ZERO ; }
    else if (MATCH (opname, "lt_zero"  )) { op = GxB_LT_ZERO ; }
    else if (MATCH (opname, "le_zero"  )) { op = GxB_LE_ZERO ; }

    else if (MATCH (opname, "ne_thunk" )) { op = GxB_NE_THUNK ; }
    else if (MATCH (opname, "eq_thunk" )) { op = GxB_EQ_THUNK ; }
    else if (MATCH (opname, "gt_thunk" )) { op = GxB_GT_THUNK ; }
    else if (MATCH (opname, "ge_thunk" )) { op = GxB_GE_THUNK ; }
    else if (MATCH (opname, "lt_thunk" )) { op = GxB_LT_THUNK ; }
    else if (MATCH (opname, "le_thunk" )) { op = GxB_LE_THUNK ; }

    else if (MATCH (opname, "isnan"    )) { op = NULL ; }

    else
    {
        mexWarnMsgIdAndTxt ("GB:warn", "unknown select op") ;
        return (false) ;
    }

    // return the op
    if (op != NULL)
    {
        ASSERT_SELECTOP_OK (op, name, GB0) ;
    }
    (*handle) = op ;
    return (true) ;
}

