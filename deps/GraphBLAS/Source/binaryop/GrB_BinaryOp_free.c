//------------------------------------------------------------------------------
// GrB_BinaryOp_free: free a binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_BinaryOp_free          // free a user-created binary operator
(
    GrB_BinaryOp *op                // handle of binary operator to free
)
{ 
    return (GB_Op_free ((GB_Operator *) op)) ;
}

