//------------------------------------------------------------------------------
// GB_resize.h: definitions for GB_resize
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_RESIZE_H
#define GB_RESIZE_H

GrB_Info GB_resize              // change the size of a matrix
(
    GrB_Matrix A,               // matrix to modify
    const uint64_t nrows_new,   // new number of rows in matrix
    const uint64_t ncols_new,   // new number of columns in matrix
    GB_Werk Werk
) ;

#endif

