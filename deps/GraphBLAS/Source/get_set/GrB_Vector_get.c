//------------------------------------------------------------------------------
// GrB_Vector_get_*: get a field in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Vector_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_get_Scalar
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
    GB_WHERE2 (v, scalar, "GrB_Vector_get_Scalar (v, scalar, field)") ;

    ASSERT_VECTOR_OK (v, "v to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    int32_t i ;
    info = GB_matvec_enum_get ((GrB_Matrix) v, &i, field) ;
    if (info == GrB_SUCCESS)
    { 
        // field specifies an int32_t: assign it to the scalar
        info = GB_setElement ((GrB_Matrix) scalar, NULL, &i, 0, 0,
            GB_INT32_code, Werk) ;
    }
    else
    {
        double x ;
        switch ((int) field)
        {
            case GxB_BITMAP_SWITCH : 
                x = (double) (v->bitmap_switch) ;
                break ;

            default : 
                return (GrB_INVALID_VALUE) ;
        }
        // field specifies a double: assign it to the scalar
        info = GB_setElement ((GrB_Matrix) scalar, NULL, &x, 0, 0,
            GB_FP64_code, Werk) ;
    }

    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_get_String
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
    ASSERT_VECTOR_OK (v, "v to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_matvec_name_get ((GrB_Matrix) v, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_get_INT32
(
    GrB_Vector v,
    int32_t * value,
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
    ASSERT_VECTOR_OK (v, "v to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_matvec_enum_get ((GrB_Matrix) v, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_get_SIZE
(
    GrB_Vector v,
    size_t * value,
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
    ASSERT_VECTOR_OK (v, "v to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_matvec_name_size_get ((GrB_Matrix) v, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_get_VOID
(
    GrB_Vector v,
    void * value,
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

