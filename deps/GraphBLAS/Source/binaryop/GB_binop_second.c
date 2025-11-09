//------------------------------------------------------------------------------
// GB_binop_second: return a SECOND binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "binaryop/GB_binop.h"

GrB_BinaryOp GB_binop_second    // return SECOND operator, or NULL on error
(
    GrB_Type type,              // operator type
    GrB_BinaryOp op             // header for SECOND_UDT operator
)
{ 

    if (type == NULL) return (NULL) ;

    switch (type->code)
    {
        case GB_BOOL_code   : return (GrB_SECOND_BOOL  ) ;
        case GB_INT8_code   : return (GrB_SECOND_INT8  ) ;
        case GB_INT16_code  : return (GrB_SECOND_INT16 ) ;
        case GB_INT32_code  : return (GrB_SECOND_INT32 ) ;
        case GB_INT64_code  : return (GrB_SECOND_INT64 ) ;
        case GB_UINT8_code  : return (GrB_SECOND_UINT8 ) ;
        case GB_UINT16_code : return (GrB_SECOND_UINT16) ;
        case GB_UINT32_code : return (GrB_SECOND_UINT32) ;
        case GB_UINT64_code : return (GrB_SECOND_UINT64) ;
        case GB_FP32_code   : return (GrB_SECOND_FP32  ) ;
        case GB_FP64_code   : return (GrB_SECOND_FP64  ) ;
        case GB_FC32_code   : return (GxB_SECOND_FC32  ) ;
        case GB_FC64_code   : return (GxB_SECOND_FC64  ) ;
        default : 
        {
            // Create a SECOND_UDT binary operator.  The function pointer for
            // the SECOND_UDT op is NULL; it is not needed since A and B are
            // disjoint for GB_wait, or the operator will not be used in a
            // generic kernel.  The function defn is also NULL.  In the JIT,
            // the SECOND multiply operator is a simple assignment so there's
            // no need for a function definition (but this assignment will not
            // be used at all anyway).  This binary op will not be treated as a
            // builtin operator, however, since its data type is not builtin.
            // Its hash, op->hash, will be nonzero.  The name of SECOND_UDT is
            // the same as the name of the type.  This is a sufficiently unique
            // name for the JIT, because the opcode differs from any user-
            // defined op, and no other mechanism exists for creating a
            // SECOND_UDT opertor except for this method.
            if (op == NULL) return (NULL) ;
            // op = &op_header has been provided by the caller
            op->header_size = 0 ;
            #ifdef GB_DEBUG
            GrB_Info info =
            #endif
            GB_binop_new (op, NULL, // op->binop_function NULL for 2nd
                type, type, type,   // type is user-defined
                type->name,         // same name as type
                NULL,               // no op->defn for the SECOND_UDT operator
                GB_SECOND_binop_code) ; // using a built-in opcode
            ASSERT (info == GrB_SUCCESS) ;
            if (GB_Global_burble_get ( ))
            { 
                #ifdef GB_DEBUG
                info =
                #endif
                GB_BinaryOp_check (op, "2nd_UDT", GxB_COMPLETE, NULL) ;
                ASSERT (info == GrB_SUCCESS) ;
            }
            ASSERT_BINARYOP_OK (op, "2nd_UDT", GB0) ;
            return (op) ;
        }
    }
}

