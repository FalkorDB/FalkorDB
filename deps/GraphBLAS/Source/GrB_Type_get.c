//------------------------------------------------------------------------------
// GrB_Type_get_*: get a field in a type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Type_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Type_get_Scalar
(
    GrB_Type type,
    GrB_Scalar value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Type_get_Scalar (type, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_TYPE_OK (type, "type for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    int32_t i ;
    uint64_t u ;

    switch ((int) field)
    {
        case GrB_EL_TYPE_CODE : 
            i = (int32_t) GB_type_code_get (type->code) ;
            return (GB_setElement ((GrB_Matrix) value, NULL, &i, 0, 0,
                GB_INT32_code, Werk)) ;
            break ;

        case GrB_SIZE : 
            u = (uint64_t) type->size ;
            return (GB_setElement ((GrB_Matrix) value, NULL, &u, 0, 0,
                GB_UINT64_code, Werk)) ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }
}

//------------------------------------------------------------------------------
// GrB_Type_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_Type_get_String
(
    GrB_Type type,
    char * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Type_get_String (type, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_TYPE_OK (type, "type for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    (*value) = '\0' ;
    const char *name ;

    switch ((int) field)
    {
        case GrB_NAME : 
        case GrB_EL_TYPE_STRING : 

            name = GB_type_name_get (type) ;
            if (name != NULL)
            {
                strcpy (value, name) ;
            }
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GxB_JIT_C_NAME : 

            strcpy (value, type->name) ;
            break ;

        case GxB_JIT_C_DEFINITION : 

            if (type->defn != NULL)
            { 
                strcpy (value, type->defn) ;
            }
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Type_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Type_get_INT32
(
    GrB_Type type,
    int32_t * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Type_get_INT32 (type, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_TYPE_OK (type, "type for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {

        case GrB_EL_TYPE_CODE : 

            (*value) = (int32_t) GB_type_code_get (type->code) ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Type_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Type_get_SIZE
(
    GrB_Type type,
    size_t * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Type_get_SIZE (type, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_TYPE_OK (type, "type for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    const char *s = NULL ;

    switch ((int) field)
    {

        case GrB_SIZE : 
            (*value) = type->size ;
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GrB_NAME : 
        case GrB_EL_TYPE_STRING : 

            s = GB_type_name_get (type) ;
            break ;

        case GxB_JIT_C_NAME : 

            s = type->name ;
            break ;

        case GxB_JIT_C_DEFINITION : 

            s = type->defn ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    (*value) = (s == NULL) ? 1 : (strlen (s) + 1) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Type_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Type_get_VOID
(
    GrB_Type type,
    void * value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

