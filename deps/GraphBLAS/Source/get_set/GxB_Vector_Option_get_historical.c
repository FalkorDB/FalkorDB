//------------------------------------------------------------------------------
// GxB_Vector_Option_get: get an option in a vector: historical methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// GxB_Vector_Option_get_INT32: get vector options (int32_t scalars)
//------------------------------------------------------------------------------

GrB_Info GxB_Vector_Option_get_INT32    // gets the current option of a vector
(
    GrB_Vector v,                   // vector to query
    int field,                      // option to query
    int32_t *value                  // return value of the vector option
)
{ 
    return (GrB_Vector_get_INT32 (v, value, field)) ;
}

//------------------------------------------------------------------------------
// GxB_Vector_Option_get_FP64: get vector options (double scalars)
//------------------------------------------------------------------------------

#define GB_FREE_ALL GrB_Scalar_free (&scalar) ;

GrB_Info GxB_Vector_Option_get_FP64      // gets the current option of a vector
(
    GrB_Vector v,                   // vector to query
    int field,                      // option to query
    double *value                   // return value of the vector option
)
{ 
    GrB_Info info ;
    GrB_Scalar scalar = NULL ;
    GB_OK (GrB_Scalar_new (&scalar, GrB_FP64)) ;
    GB_OK (GrB_Vector_get_Scalar (v, scalar, field)) ;
    GB_OK (GrB_Scalar_extractElement_FP64 (value, scalar)) ;
    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

#undef GB_FREE_ALL

//------------------------------------------------------------------------------
// GxB_Vector_Option_get: based on va_arg
//------------------------------------------------------------------------------

GrB_Info GxB_Vector_Option_get      // gets the current option of a vector
(
    GrB_Vector v,                   // vector to query
    int field,                      // option to query
    ...                             // return value of the vector option
)
{ 
    va_list ap ;
    switch (field)
    {
        case GxB_IS_HYPER : 
        {
            va_start (ap, field) ;
            bool *value = va_arg (ap, bool *) ;
            va_end (ap) ;
            (*value) = false ;
            #pragma omp flush
            return (GrB_SUCCESS) ;
        }

        case GxB_BITMAP_SWITCH : 
        case GxB_HYPER_SWITCH : 
        {
            va_start (ap, field) ;
            double *value = va_arg (ap, double *) ;
            va_end (ap) ;
            return (GxB_Vector_Option_get_FP64 (v, field, value)) ;
        }

        default : 
        {
            va_start (ap, field) ;
            int *value = va_arg (ap, int *) ;
            va_end (ap) ;
            return (GxB_Vector_Option_get_INT32 (v, field, value)) ;
        }
    }
}

