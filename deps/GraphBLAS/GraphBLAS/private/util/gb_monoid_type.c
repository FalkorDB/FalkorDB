//------------------------------------------------------------------------------
// gb_monoid_type: get the GrB_Type of a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GrB_Info gb_monoid_type
(
    // output:
    GrB_Type *type,
    // input:
    GrB_Monoid op,
    char err [ERRLEN]
)
{ 
    int code = 0 ;
    OK (GrB_Monoid_get_INT32 (op, &code, GrB_OUTP_TYPE_CODE)) ;
    (*type) = gb_code_to_type (code) ;
    return (GrB_SUCCESS) ;
}

