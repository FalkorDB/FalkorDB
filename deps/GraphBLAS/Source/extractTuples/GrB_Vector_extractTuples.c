//------------------------------------------------------------------------------
// GrB_Vector_extractTuples: extract all tuples from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extracts all tuples from a column vector, like [I,~,X] = find (v) in MATLAB.
// If any parameter I and/or X is NULL, then that component is not extracted.
// The size of the I and X arrays (those that are not NULL) is given by nvals,
// which must be at least as large as GrB_nvals (&nvals, v).  The values in the
// vector are typecasted to the type of X, as needed.

// If any parameter I and/or X is NULL, that component is not extracted.  For
// example, to extract just the row indices, pass I as non-NULL, and X as NULL.
// This is like [I,~,~] = find (v) in MATLAB.

// If v is iso and X is not NULL, the iso scalar vx [0] is expanded into X.

#include "GB.h"
#include "extractTuples/GB_extractTuples.h"

#define GB_EXTRACT_TUPLES(function_name,ctype,xtype)                        \
GrB_Info function_name      /* extract tuples from a vector */              \
(                                                                           \
    uint64_t *I,            /* array for returning row indices of tuples */ \
    ctype *X,               /* array for returning values of tuples      */ \
    uint64_t *p_nvals,      /* I, X size on input; # tuples on output    */ \
    const GrB_Vector v      /* vector to extract tuples from             */ \
)                                                                           \
{                                                                           \
    GB_WHERE_1 (v, GB_STR(function_name) " (I, X, nvals, v)") ;             \
    GB_RETURN_IF_NULL (v) ;                                                 \
    GB_RETURN_IF_NULL (p_nvals) ;                                           \
    GB_BURBLE_START (GB_STR(function_name)) ;                               \
    ASSERT (GB_VECTOR_OK (v)) ;                                             \
    info = GB_extractTuples (I, false, NULL, false, X, p_nvals, xtype,      \
        (GrB_Matrix) v, Werk) ;                                             \
    GB_BURBLE_END ;                                                         \
    GB_PRAGMA (omp flush)                                                   \
    return (info) ;                                                         \
}

// with 64-bit I array
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_BOOL  , bool       , GrB_BOOL  )
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_INT8  , int8_t     , GrB_INT8  )
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_INT16 , int16_t    , GrB_INT16 )
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_INT32 , int32_t    , GrB_INT32 )
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_INT64 , int64_t    , GrB_INT64 )
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_UINT8 , uint8_t    , GrB_UINT8 )
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_UINT16, uint16_t   , GrB_UINT16)
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_UINT32, uint32_t   , GrB_UINT32)
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_UINT64, uint64_t   , GrB_UINT64)
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_FP32  , float      , GrB_FP32  )
GB_EXTRACT_TUPLES (GrB_Vector_extractTuples_FP64  , double     , GrB_FP64  )
GB_EXTRACT_TUPLES (GxB_Vector_extractTuples_FC32  , GxB_FC32_t , GxB_FC32  )
GB_EXTRACT_TUPLES (GxB_Vector_extractTuples_FC64  , GxB_FC64_t , GxB_FC64  )

//------------------------------------------------------------------------------
// GrB_Vector_extractTuples_UDT: extract from a vector with user-defined type
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_extractTuples_UDT
(
    uint64_t *I,            // array for returning row indices of tuples
    void *X,                // array for returning values of tuples
    uint64_t *p_nvals,      // I, X size on input; # tuples on output
    const GrB_Vector v      // vector to extract tuples from
)
{ 
    GB_WHERE_1 (v, "GrB_Vector_extractTuples_UDT (I, X, nvals, v)") ;
    GB_RETURN_IF_NULL (v) ;
    GB_RETURN_IF_NULL (p_nvals) ;
    GB_BURBLE_START ("GrB_Vector_extractTuples_UDT") ;

    ASSERT (GB_VECTOR_OK (v)) ;
    if (v->type->code != GB_UDT_code)
    { 
        // v must have a user-defined type
        return (GrB_DOMAIN_MISMATCH) ;
    }
    info = GB_extractTuples (I, false, NULL, false, X, p_nvals, v->type,
        (GrB_Matrix) v, Werk) ;
    GB_BURBLE_END ;
    GB_PRAGMA (omp flush)
    return (info) ;
}

