//------------------------------------------------------------------------------
// GB_unop_identity: return an identity unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_unop.h"
#include "GB_unused.h"

GB_Operator GB_unop_identity    // return IDENTITY operator, or NULL on error
(
    GrB_Type type,              // operator type
    GrB_UnaryOp op              // header for IDENTITY_UDT operator
)
{
    if (type == NULL) return (NULL) ;
    switch (type->code)
    {
        case GB_BOOL_code    : return ((GB_Operator) GrB_IDENTITY_BOOL  ) ;
        case GB_INT8_code    : return ((GB_Operator) GrB_IDENTITY_INT8  ) ;
        case GB_INT16_code   : return ((GB_Operator) GrB_IDENTITY_INT16 ) ;
        case GB_INT32_code   : return ((GB_Operator) GrB_IDENTITY_INT32 ) ;
        case GB_INT64_code   : return ((GB_Operator) GrB_IDENTITY_INT64 ) ;
        case GB_UINT8_code   : return ((GB_Operator) GrB_IDENTITY_UINT8 ) ;
        case GB_UINT16_code  : return ((GB_Operator) GrB_IDENTITY_UINT16) ;
        case GB_UINT32_code  : return ((GB_Operator) GrB_IDENTITY_UINT32) ;
        case GB_UINT64_code  : return ((GB_Operator) GrB_IDENTITY_UINT64) ;
        case GB_FP32_code    : return ((GB_Operator) GrB_IDENTITY_FP32  ) ;
        case GB_FP64_code    : return ((GB_Operator) GrB_IDENTITY_FP64  ) ;
        case GB_FC32_code    : return ((GB_Operator) GxB_IDENTITY_FC32  ) ;
        case GB_FC64_code    : return ((GB_Operator) GxB_IDENTITY_FC64  ) ;
        default              : 
        {
            // construct the IDENTITY_UDT operator.  It will have a NULL
            // function pointer so it cannot be used in a generic kernel.  It
            // will have a nonzero hash, and will thus not be treated as a a
            // built-in operator in the JIT kernels.  The name of the operator
            // is the name of its type.
            if (op == NULL) return (NULL) ;
            // op = &op_header has been provided by the caller
            op->header_size = 0 ;
            GrB_Info info = GB_unop_new (op,
                NULL,           // op->unop_function is NULL for IDENTITY_UDT
                type, type,     // type is user-defined
                type->name,     // name is same as the type
                NULL,           // no op->defn
                GB_IDENTITY_unop_code) ;    // using a built-in opcode
            ASSERT (info == GrB_SUCCESS) ;
            return ((GB_Operator) op) ;
        }
    }
}

