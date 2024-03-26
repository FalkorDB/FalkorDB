//------------------------------------------------------------------------------
// GrB_Monoid_get_*: get a field in a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Monoid_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_get_Scalar
(
    GrB_Monoid monoid,
    GrB_Scalar value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Monoid_get_Scalar (monoid, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_MONOID_OK (monoid, "monoid to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_monoid_get (monoid, value, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_get_String
(
    GrB_Monoid monoid,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Monoid_get_String (monoid, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MONOID_OK (monoid, "monoid to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    (*value) = '\0' ;
    const char *name ;

    switch ((int) field)
    {
        case GrB_NAME : 

            // get the name of the monoid
            name = GB_monoid_name_get (monoid) ;
            if (name != NULL)
            {
                strcpy (value, name) ;
            }
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GrB_INP0_TYPE_STRING : 
        case GrB_INP1_TYPE_STRING : 
        case GrB_OUTP_TYPE_STRING : 
            return (GB_op_string_get ((GB_Operator) (monoid->op),
                value, field)) ;

        default : ;
            return (GrB_INVALID_VALUE) ;
    }
}

//------------------------------------------------------------------------------
// GrB_Monoid_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_get_INT32
(
    GrB_Monoid monoid,
    int32_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Monoid_get_INT32 (monoid, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MONOID_OK (monoid, "monoid to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_enum_get ((GB_Operator) (monoid->op), value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_get_SIZE
(
    GrB_Monoid monoid,
    size_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Monoid_get_SIZE (monoid, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MONOID_OK (monoid, "monoid to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    const char *name ;

    switch ((int) field)
    {

        case GrB_NAME : 

            // get the length of the monoid user_name, or built-in name
            name = GB_monoid_name_get (monoid) ;
            break ;

        case GrB_INP0_TYPE_STRING : 
        case GrB_INP1_TYPE_STRING : 
        case GrB_OUTP_TYPE_STRING : 
            name = GB_type_name_get (monoid->op->ztype) ;
            break ;

        case GxB_MONOID_OPERATOR : 
            (*value) = sizeof (GrB_BinaryOp) ;
            return (GrB_SUCCESS) ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    (*value) = (name == NULL) ? 1 : (strlen (name) + 1) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_get_VOID
(
    GrB_Monoid monoid,
    void * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Monoid_get_VOID (monoid, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MONOID_OK (monoid, "monoid to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {

        case GxB_MONOID_OPERATOR : 
            memcpy (value, &(monoid->op), sizeof (GrB_BinaryOp)) ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

