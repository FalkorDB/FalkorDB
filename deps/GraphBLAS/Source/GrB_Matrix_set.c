//------------------------------------------------------------------------------
// GrB_Matrix_set_*: set a field in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Matrix_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_set_Scalar
(
    GrB_Matrix A,
    GrB_Scalar value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_set_Scalar (A, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT_MATRIX_OK (A, "A to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    double dvalue = 0 ;
    int32_t ivalue = 0 ;
    GrB_Info info ;

    switch ((int) field)
    {

        case GxB_HYPER_SWITCH : 
        case GxB_BITMAP_SWITCH : 

            info = GrB_Scalar_extractElement_FP64 (&dvalue, value) ;
            break ;

        default : 

            info = GrB_Scalar_extractElement_INT32 (&ivalue, value) ;
            break ;
    }

    if (info != GrB_SUCCESS)
    { 
        return ((info == GrB_NO_VALUE) ? GrB_EMPTY_OBJECT : info) ;
    } 

    return (GB_matvec_set (A, false, ivalue, dvalue, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_set_String
(
    GrB_Matrix A,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_set_String (A, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MATRIX_OK (A, "A to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_matvec_name_set (A, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_set_INT32
(
    GrB_Matrix A,
    int32_t value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_set_INT32 (A, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT_MATRIX_OK (A, "A to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_matvec_set (A, false, value, 0, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_set_VOID
(
    GrB_Matrix A,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

