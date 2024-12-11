//------------------------------------------------------------------------------
// GB_op_size_get: get the size of a string in an op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

GrB_Info GB_op_size_get
(
    GB_Operator op,
    size_t * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    const char *s ;

    switch ((int) field)
    {

        case GxB_JIT_C_DEFINITION : 
            (*value) = (op->defn == NULL) ? 1 : (strlen (op->defn) + 1) ;
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GxB_JIT_C_NAME : 
            (*value) = strlen (op->name) + 1 ;
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GrB_NAME : 
            s = GB_op_name_get (op) ;
            (*value) = (s == NULL) ? 1 : (strlen (s) + 1) ;
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GrB_INP0_TYPE_STRING : 
            s = GB_type_name_get (op->xtype) ;
            break ; ;

        case GrB_INP1_TYPE_STRING : 
            s = GB_type_name_get (op->ytype) ;
            break ;

        case GrB_OUTP_TYPE_STRING : 
            s = GB_type_name_get (op->ztype) ;
            break ;

        case GxB_THETA_TYPE_STRING : 
            s = GB_type_name_get (op->theta_type) ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    (*value) = (s == NULL) ? 1 : (strlen (s) + 1) ;

    #pragma omp flush
    return ((s == NULL) ? GrB_NO_VALUE : GrB_SUCCESS) ;
}

