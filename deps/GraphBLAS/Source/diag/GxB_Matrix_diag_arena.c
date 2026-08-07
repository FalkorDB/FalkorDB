//------------------------------------------------------------------------------
// GxB_Matrix_diag_arena: construct a diagonal matrix from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Similar to GrB_Matrix_diag (&C, v, k), except that C is constructed
// in the given header and data arenas.  C has the same type as v.

// The arenas are checked in GxB_Matrix_new_arena.

#include "diag/GB_diag.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_diag_arena    // build a diagonal matrix from a vector
(
    GrB_Matrix *C,                  // output matrix
    const GrB_Vector v,             // input vector
    int64_t k,
    const int header_arena,
    const int data_arena
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE_1 (v, "GxB_Matrix_diag_arena (&C, v, k, h, d)") ;
    GB_RETURN_IF_NULL (v) ;
    GB_BURBLE_START ("GxB_Matrix_diag_arena") ;

    //--------------------------------------------------------------------------
    // C = diag (v,k)
    //--------------------------------------------------------------------------

    uint64_t n = v->vlen + GB_IABS (k) ;
    GB_OK (GxB_Matrix_new_arena (C, v->type, n, n, header_arena, data_arena)) ;
    GB_OK (GB_Matrix_diag (*C, (GrB_Matrix) v, k, Werk)) ;

    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

