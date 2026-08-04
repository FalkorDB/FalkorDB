//------------------------------------------------------------------------------
// gbmx_mxarray_is_scalar: check if mxArray is non-sparse numeric scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

bool gbmx_mxarray_is_scalar   // true if built-in array is a scalar
(
    const mxArray *S
)
{ 
    return (S != NULL && mxIsScalar (S) && mxIsNumeric (S) && !mxIsSparse (S)) ;
}

