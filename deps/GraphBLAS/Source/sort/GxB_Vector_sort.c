//------------------------------------------------------------------------------
// GxB_Vector_sort: sort the values in a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "sort/GB_sort.h"

GrB_Info GxB_Vector_sort
(
    // output:
    GrB_Vector w,           // vector of sorted values
    GrB_Vector p,           // vector containing the permutation
    // input
    GrB_BinaryOp op,        // comparator op
    GrB_Vector u,           // vector to sort
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (u) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (p) ;
    GB_WHERE_3 (w, p, u, "GxB_Vector_sort (w, p, op, u, desc)") ;
    GB_BURBLE_START ("GxB_Vector_sort") ;

    ASSERT (GB_VECTOR_OK (u)) ;

    //--------------------------------------------------------------------------
    // sort the vector
    //--------------------------------------------------------------------------

    info = GB_sort ((GrB_Matrix) w, (GrB_Matrix) p, op,
        (GrB_Matrix) u, true, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

