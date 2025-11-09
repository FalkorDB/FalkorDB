//------------------------------------------------------------------------------
// GB_monoid_get: get a field in a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

GrB_Info GB_monoid_get
(
    GrB_Monoid monoid,
    GrB_Scalar scalar,
    int field,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {
        case GrB_INP0_TYPE_CODE : 
        case GrB_INP1_TYPE_CODE : 
        case GrB_OUTP_TYPE_CODE : 

            return (GB_op_scalar_get ((GB_Operator) monoid->op, scalar, field,
                Werk)) ;

        case GxB_MONOID_IDENTITY : 

            if (scalar->type != monoid->op->ztype)
            { 
                // scalar type must match the monoid type
                return (GrB_DOMAIN_MISMATCH) ;
            }
            return (GB_setElement ((GrB_Matrix) scalar, NULL,
                monoid->identity, 0, 0, monoid->op->ztype->code, Werk)) ;

        case GxB_MONOID_TERMINAL : 

            if (scalar->type != monoid->op->ztype)
            { 
                // scalar type must match the monoid type
                return (GrB_DOMAIN_MISMATCH) ;
            }
            if (monoid->terminal == NULL)
            { 
                // monoid is not terminal: clear the output scalar.
                // This is not an error
                return (GB_clear ((GrB_Matrix) scalar, Werk)) ;
            }
            else
            { 
                // monoid is terminal: return the terminal scalar.
                return (GB_setElement ((GrB_Matrix) scalar, NULL,
                    monoid->terminal, 0, 0, monoid->op->ztype->code,
                    Werk)) ;
            }

        default : 
            return (GrB_INVALID_VALUE) ;
    }
}

