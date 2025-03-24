//------------------------------------------------------------------------------
// GrB_Vector_new: create a new vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The new vector is n-by-1, with no entries in it.
// A->p is size 2 and all zero.  Contents A->x and A->i are NULL.
// If this method fails, *v is set to NULL.  Vectors are not hypersparse,
// so format is standard CSC, and A->h is NULL.

#include "GB.h"

#define GB_FREE_ALL GB_Matrix_free ((GrB_Matrix *) v) ;

GrB_Info GrB_Vector_new     // create a new vector with no entries
(
    GrB_Vector *v,          // handle of vector to create
    GrB_Type type,          // type of vector to create
    uint64_t n              // dimension is n-by-1
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
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

    int64_t vlen = (int64_t) n ;

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new vector
    bool Vp_is_32, Vj_is_32, Vi_is_32 ;
    GB_determine_pji_is_32 (&Vp_is_32, &Vj_is_32, &Vi_is_32,
        GxB_SPARSE, 1, vlen, 1, NULL) ;

    GB_OK (GB_new ((GrB_Matrix *) v, // new user header
        type, vlen, 1, GB_ph_calloc,
        true,  // a GrB_Vector is always held by-column
        GxB_SPARSE, GB_Global_hyper_switch_get ( ), 1,
        Vp_is_32, Vj_is_32, Vi_is_32)) ;

    return (GrB_SUCCESS) ;
}

