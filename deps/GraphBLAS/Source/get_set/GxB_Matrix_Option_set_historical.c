//------------------------------------------------------------------------------
// GxB_Matrix_Option_set: set an option in a matrix: historical methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// GxB_Matrix_Option_set_INT32: set matrix options (int32_t scalars)
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_Option_set_INT32    // set an option in a matrix
(
    GrB_Matrix A,                   // matrix to modify
    int field,                      // option to change
    int32_t value                   // value to change it to
)
{ 
    return (GrB_Matrix_set_INT32 (A, value, field)) ;
}

//------------------------------------------------------------------------------
// GxB_Matrix_Option_set_FP64: set matrix options (double scalars)
//------------------------------------------------------------------------------

#define GB_FREE_ALL GrB_Scalar_free (&scalar) ;

GrB_Info GxB_Matrix_Option_set_FP64     // set an option in a matrix
(
    GrB_Matrix A,                   // matrix to modify
    int field,                      // option to change
    double value                    // value to change it to
)
{ 
    GrB_Info info ;
    GrB_Scalar scalar = NULL ;
    GB_OK (GrB_Scalar_new (&scalar, GrB_FP64)) ;
    GB_OK (GrB_Scalar_setElement_FP64 (scalar, value)) ;
    GB_OK (GrB_Matrix_set_Scalar (A, scalar, field)) ;
    GB_FREE_ALL
    return (GrB_SUCCESS) ;
}

#undef GB_FREE_ALL

//------------------------------------------------------------------------------
// GxB_Matrix_Option_set: based on va_arg
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_Option_set      // set an option in a matrix
(
    GrB_Matrix A,                   // matrix to modify
    int field,                      // option to change
    ...                             // value to change it to
)
{ 
    va_list ap ;
    switch (field)
    {
        case GxB_HYPER_SWITCH : 
        case GxB_BITMAP_SWITCH : 
        {
            va_start (ap, field) ;
            double value = va_arg (ap, double) ;
            va_end (ap) ;
            return (GxB_Matrix_Option_set_FP64 (A, field, value)) ;
        }

        default : 
        {
            va_start (ap, field) ;
            int value = va_arg (ap, int) ;
            va_end (ap) ;
            return (GxB_Matrix_Option_set_INT32 (A, field, value)) ;
        }
    }
}

