//------------------------------------------------------------------------------
// GrB_Matrix_diag: construct a diagonal matrix from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Similar to GxB_Matrix_diag (C, v, k, NULL), except that C is constructed
// as a new matrix, like GrB_Matrix_new.  C has the same type and memlane as v.

#include "diag/GB_diag.h"

#define GB_FREE_ALL ;

GrB_Info GrB_Matrix_diag        // construct a diagonal matrix from a vector
(
    GrB_Matrix *C,              // output matrix
    const GrB_Vector v,         // input vector
    int64_t k
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE_1 (v, "GrB_Matrix_diag (&C, v, k)") ;
    GB_RETURN_IF_NULL (v) ;
    GB_BURBLE_START ("GrB_Matrix_diag") ;

    //--------------------------------------------------------------------------
    // C = diag (v,k)
    //--------------------------------------------------------------------------

    int memlane = GB_memlane (v->header_mem) ;
    uint64_t mem = GB_mem (memlane, 0) ;

    uint64_t n = v->vlen + GB_IABS (k) ;
    GB_OK (GB_Matrix_new (C, v->type, n, n, memlane)) ;
    GB_OK (GB_Matrix_diag (*C, (GrB_Matrix) v, k, Werk)) ;

    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

