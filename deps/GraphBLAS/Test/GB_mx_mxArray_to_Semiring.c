//------------------------------------------------------------------------------
// GB_mx_mxArray_to_Semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Get a built-in semiring struct and convert it into a GraphBLAS semiring.
//
// The semiring struct must contain the following strings:
//
//      multiply    a string with the name of the 'multiply' binary operator.
//      add         a string with the name of the 'add' binary operator.
//                  The operator must be commutative.
//      type        the type of x and y for the multiply operator.
//                  ('logical', 'int8', ... 'double complex').  optional.

#include "GB_mex.h"

bool GB_mx_mxArray_to_Semiring          // true if successful
(
    GrB_Semiring *handle,               // the semiring
    const mxArray *semiring_builtin,    // built-in version of semiring
    const char *name,                   // name of the argument
    const GrB_Type default_optype,      // default operator type
    const bool user_complex             // if true, use user-defined Complex op
)
{

    (*handle) = NULL ;
    const mxArray *multiply_mx = NULL, *type_mx = NULL, *add_mx = NULL ;

    if (semiring_builtin == NULL || mxIsEmpty (semiring_builtin))
    {
        // semiring is not present; defaults will be used
        ;
    }
    else if (mxIsStruct (semiring_builtin))
    {
        // look for semiring.multiply
        int fieldnumber = mxGetFieldNumber (semiring_builtin, "multiply") ;
        if (fieldnumber >= 0)
        {
            multiply_mx = mxGetFieldByNumber (semiring_builtin, 0, fieldnumber);
        }
        // look for semiring.class
        fieldnumber = mxGetFieldNumber (semiring_builtin, "class") ;
        if (fieldnumber >= 0)
        {
            type_mx = mxGetFieldByNumber (semiring_builtin, 0, fieldnumber) ;
        }
        // look for semiring.add
        fieldnumber = mxGetFieldNumber (semiring_builtin, "add") ;
        if (fieldnumber >= 0)
        {
            add_mx = mxGetFieldByNumber (semiring_builtin, 0, fieldnumber) ;
        }
    }
    else
    {
        mexWarnMsgIdAndTxt ("GB:warn","invalid semiring") ;
        return (false) ;
    }

    // find the corresponding GraphBLAS multiply operator
    GrB_BinaryOp multiply = NULL ;
    if (!GB_mx_string_to_BinaryOp (&multiply, default_optype,
        multiply_mx, type_mx, user_complex) || multiply == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn","mult missing or failed") ;
        return (false) ;
    }

    ASSERT_BINARYOP_OK (multiply, "semiring multiply", GB0) ;

    // find the corresponding GraphBLAS add operator
    GrB_BinaryOp add = NULL ;
    if (!GB_mx_mxArray_to_BinaryOp (&add, add_mx, "add", multiply->ztype,
        user_complex) || add == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn", "add missing or failed") ;
        return (false) ;
    }
    ASSERT_BINARYOP_OK (add, "semiring add", GB0) ;

    // create the monoid with the add operator and its identity value
    GrB_Monoid monoid = GB_mx_BinaryOp_to_Monoid (add) ;
    ASSERT_MONOID_OK (monoid, "semiring monoid", GB0) ;

    if (monoid == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn", "monoid missing or failed") ;
        return (false) ;
    }

    // create the semiring
    GrB_Semiring semiring = GB_mx_semiring (monoid, multiply) ;
    if (semiring == NULL)
    {
        return (false) ;
    }

    ASSERT_SEMIRING_OK (semiring, "semiring", GB0) ;

    (*handle) = semiring ;
    return (true) ;
}

