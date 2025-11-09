//------------------------------------------------------------------------------
// GB_op_enum_get: get a field in an op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

GrB_Info GB_op_enum_get
(
    GB_Operator op,
    int32_t * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    GrB_Type type = NULL ;
    (*value) = -1 ;

    switch ((int) field)
    {
        case GrB_INP0_TYPE_CODE : type = op->xtype ; break ;
        case GrB_INP1_TYPE_CODE : type = op->ytype ; break ;
        case GrB_OUTP_TYPE_CODE : type = op->ztype ; break ;
        case GxB_THETA_TYPE_CODE : type = op->theta_type ; break ;
        default : ;
            return (GrB_INVALID_VALUE) ;
    }

    if (type == NULL)
    { 
        // operator does not depend on this input
        return (GrB_NO_VALUE) ;
    }

    (*value) = (int32_t) GB_type_code_get (type->code) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

