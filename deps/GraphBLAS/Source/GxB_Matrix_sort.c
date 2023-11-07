//------------------------------------------------------------------------------
// GxB_Matrix_sort: sort the values in each vector of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_sort.h"

GrB_Info GxB_Matrix_sort
(
    // output:
    GrB_Matrix C,           // matrix of sorted values
    GrB_Matrix P,           // matrix containing the permutations
    // input
    GrB_BinaryOp op,        // comparator op
    GrB_Matrix A,           // matrix to sort
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_sort (C, P, op, A, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_sort") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, xx0, xx1, xx2, A_transpose, xx3, xx4, xx5) ;

    //--------------------------------------------------------------------------
    // sort the matrix
    //--------------------------------------------------------------------------

    info = GB_sort (C, P, op, A, A_transpose, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

