//------------------------------------------------------------------------------
// GB_mx_mxArray_to_Vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

GrB_Vector GB_mx_mxArray_to_Vector     // returns GraphBLAS version of V
(
    const mxArray *V_builtin,           // built-in version of V
    const char *name,                   // name of the argument
    const bool deep_copy,               // if true, return a deep copy
    const bool empty    // if false, 0-by-0 matrices are returned as NULL.
                        // if true, a 0-by-0 matrix is returned.
)
{

    GrB_Matrix V = GB_mx_mxArray_to_Matrix (V_builtin, name, deep_copy, empty) ;
    if (V != NULL && !GB_VECTOR_OK (V))
    {
        mexWarnMsgIdAndTxt ("GB:warn", "must be a column vector") ;
        GrB_Matrix_free_(&V) ;
        return (NULL) ;
    }
    return ((GrB_Vector) V) ;
}

