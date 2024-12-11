//------------------------------------------------------------------------------
// GB_code_type: convert a type code to a GrB_Type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GrB_assign, GxB_subassign, and GrB_setElement operations all accept
// scalar inputs.  The scalar code is converted to an appropriate GrB_Type
// here.  For user-defined types, the scalar is required to have the same type
// as the matrix being operated on.  This cannot be checked; results are
// undefined if the user passes in a void * pointer to a different user-defined
// type.

#include "GB.h"

GrB_Type GB_code_type           // return the GrB_Type corresponding to the code
(
    const GB_Type_code code,    // type code to convert
    const GrB_Type type         // user type if code is user-defined
)
{

    ASSERT (code <= GB_UDT_code) ;
    switch (code)
    {
        case GB_BOOL_code   : return (GrB_BOOL)   ;
        case GB_INT8_code   : return (GrB_INT8)   ;
        case GB_INT16_code  : return (GrB_INT16)  ;
        case GB_INT32_code  : return (GrB_INT32)  ;
        case GB_INT64_code  : return (GrB_INT64)  ;
        case GB_UINT8_code  : return (GrB_UINT8)  ;
        case GB_UINT16_code : return (GrB_UINT16) ;
        case GB_UINT32_code : return (GrB_UINT32) ;
        case GB_UINT64_code : return (GrB_UINT64) ;
        case GB_FP32_code   : return (GrB_FP32)   ;
        case GB_FP64_code   : return (GrB_FP64)   ;
        case GB_FC32_code   : return (GxB_FC32)   ;
        case GB_FC64_code   : return (GxB_FC64)   ;
        case GB_UDT_code    : 
        default             : return (type) ;
    }
}

