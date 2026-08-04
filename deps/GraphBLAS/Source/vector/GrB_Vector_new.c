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

// The vector is allocated in the arenas determined by the current Context.

#include "GB.h"

GrB_Info GrB_Vector_new     // create a new vector with no entries
(
    GrB_Vector *v,          // handle of vector to create
    GrB_Type type,          // type of vector to create
    uint64_t n              // dimension is n-by-1
)
{
    int header_arena = GB_Context_header_arena ( ) ;
    int data_arena = GB_Context_data_arena ( ) ;
    return (GxB_Vector_new_arena (v, type, n, header_arena, data_arena)) ;
}

