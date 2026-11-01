//------------------------------------------------------------------------------
// GB_binop_first: return a FIRST operator of the given type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "binaryop/GB_binop.h"

GrB_BinaryOp GB_binop_first
(
    GrB_Type type,
    GrB_BinaryOp op             // empty op header for user-defined case
)
{

    if (type == NULL) return (NULL) ;

    switch (type->code)
    {
        case GB_BOOL_code   : return (GrB_FIRST_BOOL  ) ;
        case GB_INT8_code   : return (GrB_FIRST_INT8  ) ;
        case GB_INT16_code  : return (GrB_FIRST_INT16 ) ;
        case GB_INT32_code  : return (GrB_FIRST_INT32 ) ;
        case GB_INT64_code  : return (GrB_FIRST_INT64 ) ;
        case GB_UINT8_code  : return (GrB_FIRST_UINT8 ) ;
        case GB_UINT16_code : return (GrB_FIRST_UINT16) ;
        case GB_UINT32_code : return (GrB_FIRST_UINT32) ;
        case GB_UINT64_code : return (GrB_FIRST_UINT64) ;
        case GB_FP32_code   : return (GrB_FIRST_FP32  ) ;
        case GB_FP64_code   : return (GrB_FIRST_FP64  ) ;
        case GB_FC32_code   : return (GxB_FIRST_FC32  ) ;
        case GB_FC64_code   : return (GxB_FIRST_FC64  ) ;
        default : 
            // Create a FIRST_UDT binary operator.  The function pointer for
            // the FIRST_UDT op is NULL; it is not needed by FIRST.  The
            // function defn is also NULL.  In the JIT, the FIRST multiply
            // operator is a simple assignment so there's no need for a
            // function definition.  This binary op will not be treated as a
            // builtin operator, however, since its data type is not builtin.
            // Its hash, op->hash, will be nonzero.  The name of FIRST_UDT is
            // just "1st".  It is not possible for the user to create a
            // jitifyable operator with the name "1st", because of the leading
            // "1" character in its name.
            op->header_mem = 0 ;        // static header for binary op
            #ifdef GB_DEBUG
            GrB_Info info =
            #endif
            GB_binop_new (op, NULL,     // op->binop_func. NULL for FIRST_UDT
                type, type, type,       // type is user-defined
                "1st",                  // simple name for this operator
                NULL,                   // no op->defn for FIRST_UDT
                GB_FIRST_binop_code,    // using a built-in opcode
                0) ;                    // arena not used (static header)
            ASSERT (info == GrB_SUCCESS) ;
            return (op) ;
    }
}

