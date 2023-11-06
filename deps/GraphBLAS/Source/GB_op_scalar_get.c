//------------------------------------------------------------------------------
// GB_op_scalar_get: get a field in an op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_op_scalar_get
(
    GB_Operator op,
    GrB_Scalar value,
    GrB_Field field,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    int i ;
    GrB_Info info = GB_op_enum_get (op, &i, field) ;
    if (info == GrB_SUCCESS)
    { 
        info = GB_setElement ((GrB_Matrix) value, NULL, &i, 0, 0,
            GB_INT32_code, Werk) ;
    }
    return (info) ;
}

