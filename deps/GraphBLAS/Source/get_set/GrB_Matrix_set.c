//------------------------------------------------------------------------------
// GrB_Matrix_set_*: set a field in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Matrix_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_set_Scalar
(
    GrB_Matrix A,
    GrB_Scalar scalar,
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL (scalar) ;
    GB_WHERE_2 (A, scalar, "GrB_Matrix_set_Scalar (A, scalar, field)") ;

    ASSERT_MATRIX_OK (A, "GrB: A to set Scalar option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    double dvalue = 0 ;
    int32_t ivalue = 0 ;

    switch ((int) field)
    {

        case GxB_HYPER_SWITCH : 
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

    return (GB_matvec_set (A, false, ivalue, dvalue, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_set_String
(
    GrB_Matrix A,
    char * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_INVALID (A) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MATRIX_OK (A, "GrB: A to set String option", GB0) ;

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
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_WHERE1 (A, "GrB_Matrix_set_INT32 (A, value, field)") ;

    ASSERT_MATRIX_OK (A, "GrB: A to set int32 option", GB0) ;

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
    int field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

