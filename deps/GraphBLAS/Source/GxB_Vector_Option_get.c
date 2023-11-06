//------------------------------------------------------------------------------
// GxB_Vector_Option_get: get an option in a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------

// GxB_Vector_Option_get is a single va_arg-based method for any vector option,
// of any type.  The following functions are non-va_arg-based methods
// (useful for compilers and interfaces that do not support va_arg):
//
//  GxB_Vector_Option_get_INT32         int32_t scalars
//  GxB_Vector_Option_get_FP64          double scalars

//------------------------------------------------------------------------------
// GxB_Vector_Option_get_INT32: get vector options (int32_t scalars)
//------------------------------------------------------------------------------

GrB_Info GxB_Vector_Option_get_INT32    // gets the current option of a vector
(
    GrB_Vector v,                   // vector to query
    GxB_Option_Field field,         // option to query
    int32_t *value                  // return value of the vector option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_Option_get_INT32 (v, field, &value)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    ASSERT_VECTOR_OK (v, "v to get option", GB0) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the option
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_SPARSITY_CONTROL : 

            (*value) = v->sparsity_control ;
            break ;

        case GxB_SPARSITY_STATUS : 

            (*value) = GB_sparsity ((GrB_Matrix) v) ;
            break ;

        case GxB_FORMAT : 

            // a GrB_Vector is always stored by-column
            (*value) = (int32_t) GxB_BY_COL ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;

    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Vector_Option_get_FP64: get vector options (double scalars)
//------------------------------------------------------------------------------

GrB_Info GxB_Vector_Option_get_FP64      // gets the current option of a vector
(
    GrB_Vector v,                   // vector to query
    GxB_Option_Field field,         // option to query
    double *value                   // return value of the vector option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_Option_get_FP64 (v, field, &value)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    ASSERT_VECTOR_OK (v, "v to get option", GB0) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the option
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_BITMAP_SWITCH : 

            (*value) = (double) v->bitmap_switch ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;

    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Vector_Option_get: based on va_arg
//------------------------------------------------------------------------------

GrB_Info GxB_Vector_Option_get      // gets the current option of a vector
(
    GrB_Vector v,                   // vector to query
    GxB_Option_Field field,         // option to query
    ...                             // return value of the vector option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_Option_get (v, field, &value)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    ASSERT_VECTOR_OK (v, "v to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the option
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        case GxB_BITMAP_SWITCH : 

            {
                va_start (ap, field) ;
                double *bitmap_switch = va_arg (ap, double *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (bitmap_switch) ;
                (*bitmap_switch) = (double) v->bitmap_switch ;
            }
            break ;

        case GxB_SPARSITY_CONTROL : 

            {
                va_start (ap, field) ;
                int *sparsity_control = va_arg (ap, int *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (sparsity_control) ;
                (*sparsity_control) = v->sparsity_control ;
            }
            break ;

        case GxB_SPARSITY_STATUS : 

            {
                va_start (ap, field) ;
                int *sparsity = va_arg (ap, int *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (sparsity) ;
                (*sparsity) = GB_sparsity ((GrB_Matrix) v) ;
            }
            break ;

        case GxB_FORMAT : 

            {
                // a GrB_Vector is always stored by-column
                va_start (ap, field) ;
                GxB_Format_Value *format = va_arg (ap, GxB_Format_Value *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (format) ;
                (*format) = GxB_BY_COL ;
            }
            break ;

        case GxB_IS_HYPER : // historical; use GxB_SPARSITY_STATUS instead

            {
                // a GrB_Vector is never hypersparse
                va_start (ap, field) ;
                bool *v_is_hyper = va_arg (ap, bool *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (v_is_hyper) ;
                (*v_is_hyper) = false ;
            }
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;

    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

