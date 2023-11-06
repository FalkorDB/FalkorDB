//------------------------------------------------------------------------------
// GrB_Vector_new: create a new vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The new vector is n-by-1, with no entries in it.
// A->p is size 2 and all zero.  Contents A->x and A->i are NULL.
// If this method fails, *v is set to NULL.  Vectors are not hypersparse,
// so format is standard CSC, and A->h is NULL.

#include "GB.h"

GrB_Info GrB_Vector_new     // create a new vector with no entries
(
    GrB_Vector *v,          // handle of vector to create
    GrB_Type type,          // type of vector to create
    GrB_Index n             // dimension is n-by-1
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Vector_new (&v, type, n)") ;
    GB_RETURN_IF_NULL (v) ;
    (*v) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;

    if (n > GB_NMAX)
    { 
        // problem too large
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // create the vector
    //--------------------------------------------------------------------------

    GrB_Info info ;
    int64_t vlen = (int64_t) n ;

    info = GB_new ((GrB_Matrix *) v, // new user header
        type, vlen, 1, GB_Ap_calloc,
        true,  // a GrB_Vector is always held by-column
        GxB_SPARSE, GB_Global_hyper_switch_get ( ), 1) ;
    ASSERT (GB_IMPLIES (info == GrB_SUCCESS, GB_VECTOR_OK (*v))) ;
    return (info) ;
}

