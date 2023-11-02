//------------------------------------------------------------------------------
// GB_reshape:  reshape a matrix into another matrix, or reshape it in-place
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_RESHAPE_H
#define GB_RESHAPE_H

GrB_Info GB_reshape         // reshape a GrB_Matrix into another GrB_Matrix
(
    // output, if not in-place:
    GrB_Matrix *Chandle,    // output matrix, in place if Chandle == NULL
    // input, or input/output:
    GrB_Matrix A,           // input matrix, or input/output if in-place
    // input:
    bool by_col,            // true if reshape by column, false if by row
    int64_t nrows_new,      // number of rows of C
    int64_t ncols_new,      // number of columns of C
    GB_Werk Werk
) ;

#endif

