//------------------------------------------------------------------------------
// gb_binaryop_ztype: get the GrB_Type of the z output of a binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Type gb_binaryop_ztype
(
    GrB_BinaryOp op
)
{ 
    int code = 0 ;
    OK (GrB_BinaryOp_get_INT32 (op, &code, GrB_OUTP_TYPE_CODE)) ;
    return (gb_code_to_type (code)) ;
}

