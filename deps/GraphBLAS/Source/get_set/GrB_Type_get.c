//------------------------------------------------------------------------------
// GrB_Type_get_*: get a field in a type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Type_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Type_get_Scalar
(
    GrB_Type type,
    GrB_Scalar scalar,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL (scalar) ;
    GB_WHERE_1 (scalar, "GrB_Type_get_Scalar (type, scalar, field)") ;

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
            return (GB_setElement ((GrB_Matrix) scalar, NULL, &i, 0, 0,
                GB_INT32_code, Werk)) ;
            break ;

        case GrB_SIZE : 
            u = (uint64_t) type->size ;
            return (GB_setElement ((GrB_Matrix) scalar, NULL, &u, 0, 0,
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
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
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
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
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
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
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
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_TYPE_OK (type, "type for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    if (field == GxB_PRINT_FUNCTION && type->code == GB_UDT_code)
    { 
        void **func = (void **) value ;
        (*func) = type->print_function ;
        return (GrB_SUCCESS) ;
    }

    return (GrB_INVALID_VALUE) ;
}

