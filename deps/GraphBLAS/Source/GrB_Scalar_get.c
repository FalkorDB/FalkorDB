//------------------------------------------------------------------------------
// GrB_Scalar_get_*: get a field in a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Scalar_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_get_Scalar
(
    GrB_Scalar s,
    GrB_Scalar value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Scalar_get_Scalar (s, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (s) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_SCALAR_OK (s, "s to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    int32_t i ;
    GrB_Info info = GB_matvec_enum_get ((GrB_Matrix) s, &i, field) ;
    if (info == GrB_SUCCESS)
    { 
        // field specifies an int32_t: assign it to the scalar
        info = GB_setElement ((GrB_Matrix) value, NULL, &i, 0, 0,
            GB_INT32_code, Werk) ;
    }
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Scalar_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_get_String
(
    GrB_Scalar s,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Scalar_get_String (s, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (s) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SCALAR_OK (s, "s to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_matvec_name_get ((GrB_Matrix) s, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Scalar_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_get_INT32
(
    GrB_Scalar s,
    int32_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Scalar_get_INT32 (s, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (s) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SCALAR_OK (s, "s to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_matvec_enum_get ((GrB_Matrix) s, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Scalar_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_get_SIZE
(
    GrB_Scalar s,
    size_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Scalar_get_SIZE (s, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (s) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SCALAR_OK (s, "s to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_matvec_name_size_get ((GrB_Matrix) s, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Scalar_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_get_VOID
(
    GrB_Scalar s,
    void * value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

