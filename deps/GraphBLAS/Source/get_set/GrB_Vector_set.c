//------------------------------------------------------------------------------
// GrB_Vector_set_*: set a field in a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Vector_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_set_Scalar
(
    GrB_Vector v,
    GrB_Scalar scalar,
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (v) ;
    GB_RETURN_IF_NULL (scalar) ;
    GB_WHERE2 (v, scalar, "GrB_Vector_set_Scalar (v, scalar, field)") ;

    ASSERT_VECTOR_OK (v, "v to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    double dvalue = 0 ;
    int32_t ivalue = 0 ;

    switch ((int) field)
    {
        case GxB_BITMAP_SWITCH : 
            info = GrB_Scalar_extractElement_FP64 (&dvalue, scalar) ;
            break ;
        default : 
            info = GrB_Scalar_extractElement_INT32 (&ivalue, scalar) ;
            break ;
    }

    if (info != GrB_SUCCESS)
    { 
        return ((info == GrB_NO_VALUE) ? GrB_EMPTY_OBJECT : info) ;
    } 
    return (GB_matvec_set ((GrB_Matrix) v, true, ivalue, dvalue, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_set_String
(
    GrB_Vector v,
    char * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_INVALID (v) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_VECTOR_OK (v, "v to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_matvec_name_set ((GrB_Matrix) v, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_set_INT32
(
    GrB_Vector v,
    int32_t value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (v) ;
    GB_WHERE_1 (v, "GrB_Vector_set_INT32 (v, value, field)") ;

    ASSERT_VECTOR_OK (v, "v to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_matvec_set ((GrB_Matrix) v, true, value, 0, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_set_VOID
(
    GrB_Vector v,
    void * value,
    int field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

