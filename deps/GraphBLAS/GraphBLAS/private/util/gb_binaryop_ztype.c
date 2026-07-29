//------------------------------------------------------------------------------
// gb_binaryop_ztype: get the GrB_Type of the z output of a binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GrB_Info gb_binaryop_ztype
(
    // output
    GrB_Type *ztype,    // the GrB_Type of the output of a binary op
    // input
    GrB_BinaryOp op,
    char err [ERRLEN]
)
{ 
    int code = 0 ;
    OK (GrB_BinaryOp_get_INT32 (op, &code, GrB_OUTP_TYPE_CODE)) ;
    (*ztype) = gb_code_to_type (code) ;
    return (GrB_SUCCESS) ;
}

