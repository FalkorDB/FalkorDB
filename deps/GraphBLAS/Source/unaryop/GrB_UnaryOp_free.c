//------------------------------------------------------------------------------
// GrB_UnaryOp_free: free a unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_UnaryOp_free           // free a user-created unary operator
(
    GrB_UnaryOp *op                 // handle of unary operator to free
)
{ 
    return (GB_Op_free ((GB_Operator *) op)) ;
}

