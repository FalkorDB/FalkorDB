//------------------------------------------------------------------------------
// GB_op_string_get: get a field in an op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

GrB_Info GB_op_string_get
(
    GB_Operator op,
    char * value,
    int field
)
{

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    (*value) = '\0' ;
    GrB_Type type = NULL ;
    const char *name ;

    switch ((int) field)
    {

        case GrB_NAME : 

            name = GB_op_name_get (op) ;
            if (name != NULL)
            {
                strcpy (value, name) ;
            }
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GxB_JIT_C_NAME : 

            strcpy (value, op->name) ;
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GxB_JIT_C_DEFINITION : 

            if (op->defn != NULL)
            { 
                strcpy (value, op->defn) ;
            }
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GrB_INP0_TYPE_STRING : type = op->xtype ; break ;
        case GrB_INP1_TYPE_STRING : type = op->ytype ; break ;
        case GrB_OUTP_TYPE_STRING : type = op->ztype ; break ;
        case GxB_THETA_TYPE_STRING : type = op->theta_type ; break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    if (type == NULL)
    { 
        // operator does not depend on this input
        return (GrB_NO_VALUE) ;
    }

    name = GB_type_name_get (type) ;
    if (name != NULL)
    {
        strcpy (value, name) ;
    }
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

