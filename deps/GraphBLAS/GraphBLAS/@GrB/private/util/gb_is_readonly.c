//------------------------------------------------------------------------------
// gb_is_readonly: determine if A has any readonly components
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

bool gb_is_readonly             // true if A has any readonly components
(
    GrB_Matrix A                // GrB_matrix to query
)
{
    if (A == NULL) return (false) ;
    int readonly ;
    OK (GrB_Matrix_get_INT32 (A, &readonly, GxB_IS_READONLY)) ;
    return ((bool) readonly) ;
}

