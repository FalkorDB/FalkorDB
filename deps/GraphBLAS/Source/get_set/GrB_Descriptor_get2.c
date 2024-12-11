//------------------------------------------------------------------------------
// GrB_Descriptor_get_*: get a field in a descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------

static GrB_Info GB_desc_get
(
    GrB_Descriptor desc,        // descriptor to query; NULL is ok
    int32_t *value,             // return value of the descriptor
    int field                   // parameter to query
)
{

    //--------------------------------------------------------------------------
    // get the parameter
    //--------------------------------------------------------------------------

    switch (field)
    {
        case GrB_OUTP : 

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->out) ;
            break ;

        case GrB_MASK : 

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->mask) ;
            break ;

        case GrB_INP0 : 

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->in0) ;
            break ;

        case GrB_INP1 : 

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->in1) ;
            break ;

        case GxB_AxB_METHOD : 

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->axb) ;
            break ;

        case GxB_SORT : 

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->do_sort);
            break ;

        case GxB_COMPRESSION : 

            (*value) = (int32_t) ((desc == NULL) ?
                GrB_DEFAULT : desc->compression) ; 
            break ;

        case GxB_IMPORT : 

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->import) ;
            if ((*value) != GrB_DEFAULT) (*value) = GxB_SECURE_IMPORT ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_get_Scalar
(
    GrB_Descriptor desc,
    GrB_Scalar value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Descriptor_get_Scalar (desc, value, field)") ;
    GB_RETURN_IF_FAULTY (desc) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    int32_t i ;
    GrB_Info info = GB_desc_get (desc, &i, field) ;
    if (info == GrB_SUCCESS)
    { 
        info = GB_setElement ((GrB_Matrix) value, NULL, &i, 0, 0,   
            GB_INT32_code, Werk) ;
    }
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_get_String
(
    GrB_Descriptor desc,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Descriptor_get_String (desc, value, field)") ;
    GB_RETURN_IF_FAULTY (desc) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the name
    //--------------------------------------------------------------------------

    (*value) = '\0' ;
    const char *dname = GB_desc_name_get (desc) ;
    if (dname != NULL)
    {
        strcpy (value, dname) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_get_INT32
(
    GrB_Descriptor desc,
    int32_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Descriptor_get_INT32 (desc, value, field)") ;
    GB_RETURN_IF_FAULTY (desc) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_desc_get (desc, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_get_SIZE
(
    GrB_Descriptor desc,
    size_t * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Descriptor_get_SIZE (desc, value, field)") ;
    GB_RETURN_IF_FAULTY (desc) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    if (field != GrB_NAME)
    { 
        return (GrB_INVALID_VALUE) ;
    }

    if (desc != NULL && desc->user_name != NULL)
    { 
        (*value) = desc->user_name_size ;
    }
    else
    {
        (*value) = GxB_MAX_NAME_LEN ;
    }
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_get_VOID
(
    GrB_Descriptor desc,
    void * value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

