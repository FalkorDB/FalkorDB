//------------------------------------------------------------------------------
// GxB_Matrix_Option_get: get an option in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_Matrix_Option_get is a single va_arg-based method for any matrix option,
// of any type.  The following functions are non-va_arg-based methods
// (useful for compilers and interfaces that do not support va_arg):
//
//  GxB_Matrix_Option_get_INT32         int32_t scalars
//  GxB_Matrix_Option_get_FP64          double scalars

#include "GB.h"

//------------------------------------------------------------------------------
// GxB_Matrix_Option_get_INT32: get matrix options (int32_t scalars)
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_Option_get_INT32    // gets the current option of a matrix
(
    GrB_Matrix A,                   // matrix to query
    GxB_Option_Field field,         // option to query
    int32_t *value                  // return value of the matrix option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_Option_get_INT32 (A, field, &value)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT_MATRIX_OK (A, "A to get option", GB0) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the option
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_SPARSITY_CONTROL : 

            (*value) = A->sparsity_control ;
            break ;

        case GxB_SPARSITY_STATUS : 

            (*value) = GB_sparsity (A) ;
            break ;

        case GxB_FORMAT : 

            (*value) = (int32_t) ((A->is_csc) ? GxB_BY_COL : GxB_BY_ROW) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Matrix_Option_get_FP64: get matrix options (double scalars)
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_Option_get_FP64     // gets the current option of a matrix
(
    GrB_Matrix A,                   // matrix to query
    GxB_Option_Field field,         // option to query
    double *value                   // return value of the matrix option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_Option_get_FP64 (A, field, &value)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT_MATRIX_OK (A, "A to get option", GB0) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the option
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_HYPER_SWITCH : 

            (*value) = (double) A->hyper_switch ;
            break ;

        case GxB_BITMAP_SWITCH : 

            (*value) = (double) A->bitmap_switch ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Matrix_Option_get: based on va_arg
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_Option_get      // gets the current option of a matrix
(
    GrB_Matrix A,                   // matrix to query
    GxB_Option_Field field,         // option to query
    ...                             // return value of the matrix option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_Option_get (A, field, &value)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT_MATRIX_OK (A, "A to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the option
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        case GxB_HYPER_SWITCH : 

            {
                va_start (ap, field) ;
                double *hyper_switch = va_arg (ap, double *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (hyper_switch) ;
                (*hyper_switch) = (double) A->hyper_switch ;
            }
            break ;

        case GxB_BITMAP_SWITCH : 

            {
                va_start (ap, field) ;
                double *bitmap_switch = va_arg (ap, double *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (bitmap_switch) ;
                (*bitmap_switch) = (double) A->bitmap_switch ;
            }
            break ;

        case GxB_SPARSITY_CONTROL : 

            {
                va_start (ap, field) ;
                int *sparsity_control = va_arg (ap, int *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (sparsity_control) ;
                (*sparsity_control) = A->sparsity_control ;
            }
            break ;

        case GxB_SPARSITY_STATUS : 

            {
                va_start (ap, field) ;
                int *sparsity = va_arg (ap, int *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (sparsity) ;
                (*sparsity) = GB_sparsity (A) ;
            }
            break ;

        case GxB_FORMAT : 

            {
                va_start (ap, field) ;
                GxB_Format_Value *format = va_arg (ap, GxB_Format_Value *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (format) ;
                (*format) = (A->is_csc) ? GxB_BY_COL : GxB_BY_ROW ;
            }
            break ;

        case GxB_IS_HYPER : // historical; use GxB_SPARSITY_STATUS instead

            {
                va_start (ap, field) ;
                bool *A_is_hyper = va_arg (ap, bool *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (A_is_hyper) ;
                (*A_is_hyper) = (GB_sparsity (A) == GxB_HYPERSPARSE) ;
            }
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

