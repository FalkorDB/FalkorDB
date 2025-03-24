//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_free: free an index_binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_IndexBinaryOp_free     // free a user-created index binary operator
(
    GxB_IndexBinaryOp *op           // handle of index binary operator to free
)
{ 
    return (GB_Op_free ((GB_Operator *) op)) ;
}

