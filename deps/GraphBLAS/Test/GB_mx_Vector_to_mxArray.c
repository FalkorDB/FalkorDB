//------------------------------------------------------------------------------
// GB_mx_Vector_to_mxArray
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Convert a GraphBLAS sparse vector to a built-in struct containing a
// sparse vector and a string.  The GraphBLAS vector is destroyed.

#include "GB_mex.h"

mxArray *GB_mx_Vector_to_mxArray   // returns the built-in mxArray
(
    GrB_Vector *handle,             // handle of GraphBLAS matrix to convert
    const char *name,               // name for error reporting
    const bool create_struct        // if true, then return a struct
)
{
    return (GB_mx_object_to_mxArray ((GrB_Matrix *)
        handle, name, create_struct)) ;
}

