//------------------------------------------------------------------------------
// GB_diag: definitions for GxB_Matrix_diag and GxB_Vector_diag
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_DIAG_H
#define GB_DIAG_H
#include "GB.h"

GrB_Info GB_Matrix_diag     // build a diagonal matrix from a vector
(
    GrB_Matrix C,                   // output matrix
    const GrB_Matrix V_input,       // input vector (as an n-by-1 matrix)
    int64_t k,
    GB_Werk Werk
) ;

GrB_Info GB_Vector_diag     // extract a diagonal from a matrix, as a vector
(
    GrB_Matrix V,                   // output vector (as an n-by-1 matrix)
    const GrB_Matrix A,             // input matrix
    int64_t k,
    GB_Werk Werk
) ;

#endif

