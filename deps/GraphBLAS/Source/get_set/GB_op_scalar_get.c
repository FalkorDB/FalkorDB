//------------------------------------------------------------------------------
// GB_op_scalar_get: get a field in an op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

GrB_Info GB_op_scalar_get
(
    GB_Operator op,
    GrB_Scalar scalar,
    int field,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    if (field == GxB_THETA)
    {
        // get theta from an index binary op, or a binary op created from one
        if (op->theta == NULL)
        {
            // this op does not have a theta scalar
            return (GrB_INVALID_VALUE) ;
        }
        else if (scalar->type != op->theta_type)
        { 
            // scalar type must match the theta type
            return (GrB_DOMAIN_MISMATCH) ;
        }
        return (GB_setElement ((GrB_Matrix) scalar, NULL, op->theta, 0, 0,
            op->theta_type->code, Werk)) ;
    }
    else
    {
        // get an integer (enum) scalar from any op
        int i ;
        GrB_Info info = GB_op_enum_get (op, &i, field) ;
        if (info == GrB_SUCCESS)
        { 
            info = GB_setElement ((GrB_Matrix) scalar, NULL, &i, 0, 0,
                GB_INT32_code, Werk) ;
        }
        return (info) ;
    }
}

