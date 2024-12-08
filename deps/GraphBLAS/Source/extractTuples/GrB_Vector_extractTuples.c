//------------------------------------------------------------------------------
// GrB_Vector_extractTuples: extract all tuples from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extracts all tuples from a column vector, like [I,~,X] = find (v).  If any
// parameter I and/or X is NULL, then that component is not extracted.  The
// size of the I and X arrays (those that are not NULL) is given by nvals,
// which must be at least as large as GrB_nvals (&nvals, v).  The values in the
// vector are typecasted to the type of X, as needed.

// If any parameter I and/or X is NULL, that component is not extracted.  So to
// extract just the row indices, pass I as non-NULL, and X as NULL.  This is
// like [I,~,~] = find (v) in MATLAB.

// If v is iso and X is not NULL, the iso scalar vx [0] is expanded into X.

#include "GB.h"
#include "extractTuples/GB_extractTuples.h"

#define GB_EXTRACT(prefix,type,T)                                             \
GrB_Info GB_EVAL3 (prefix, _Vector_extractTuples_, T)                         \
(                                                                             \
    GrB_Index *I,           /* array for returning row indices of tuples */   \
    type *X,                /* array for returning values of tuples      */   \
    GrB_Index *p_nvals,     /* I, X size on input; # tuples on output    */   \
    const GrB_Vector v      /* vector to extract tuples from             */   \
)                                                                             \
{                                                                             \
    GB_WHERE1 ("GrB_Vector_extractTuples_" GB_STR(T) " (I, X, nvals, v)") ;   \
    GB_BURBLE_START ("GrB_Vector_extractTuples") ;                            \
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;                                         \
    GB_RETURN_IF_NULL (p_nvals) ;                                             \
    ASSERT (GB_VECTOR_OK (v)) ;                                               \
    GrB_Info info = GB_extractTuples (I, NULL, X, p_nvals, prefix ## _ ## T,  \
        (GrB_Matrix) v, Werk) ;                                               \
    GB_BURBLE_END ;                                                           \
    GB_PRAGMA (omp flush)                                                     \
    return (info) ;                                                           \
}

//       prefix, C type of X, X type
GB_EXTRACT (GrB, bool       , BOOL   )
GB_EXTRACT (GrB, int8_t     , INT8   )
GB_EXTRACT (GrB, uint8_t    , UINT8  )
GB_EXTRACT (GrB, int16_t    , INT16  )
GB_EXTRACT (GrB, uint16_t   , UINT16 )
GB_EXTRACT (GrB, int32_t    , INT32  )
GB_EXTRACT (GrB, uint32_t   , UINT32 )
GB_EXTRACT (GrB, int64_t    , INT64  )
GB_EXTRACT (GrB, uint64_t   , UINT64 )
GB_EXTRACT (GrB, float      , FP32   )
GB_EXTRACT (GrB, double     , FP64   )
GB_EXTRACT (GxB, GxB_FC32_t , FC32   )
GB_EXTRACT (GxB, GxB_FC64_t , FC64   )

//------------------------------------------------------------------------------
// GrB_Vector_extractTuples_UDT: extract from a vector with user-defined type
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_extractTuples_UDT
(
    GrB_Index *I,           // array for returning row indices of tuples
    void *X,                // array for returning values of tuples
    GrB_Index *p_nvals,     // I, X size on input; # tuples on output
    const GrB_Vector v      // vector to extract tuples from
)
{ 
    GB_WHERE1 ("GrB_Vector_extractTuples_UDT (I, X, nvals, v)") ;
    GB_BURBLE_START ("GrB_Vector_extractTuples") ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    GB_RETURN_IF_NULL (p_nvals) ;
    ASSERT (GB_VECTOR_OK (v)) ;
    if (v->type->code != GB_UDT_code)
    { 
        // v must have a user-defined type
        return (GrB_DOMAIN_MISMATCH) ;
    }
    GrB_Info info = GB_extractTuples (I, NULL, X, p_nvals, v->type,
        (GrB_Matrix) v, Werk) ;
    GB_BURBLE_END ;
    GB_PRAGMA (omp flush)
    return (info) ;
}

