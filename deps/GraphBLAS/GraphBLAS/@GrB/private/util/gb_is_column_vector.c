//------------------------------------------------------------------------------
// gb_is_column_vector: determine if A is a column vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

bool gb_is_column_vector        // true if A is a column vector
(
    GrB_Matrix A                // GrB_matrix to query
)
{
    if (A == NULL) return (false) ;
    uint64_t ncols ;
    int sparsity, orientation ;
    OK (GrB_Matrix_get_INT32 (A, &sparsity, GxB_SPARSITY_STATUS)) ;
    OK (GrB_Matrix_get_INT32 (A, &orientation, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    return (sparsity != GxB_HYPERSPARSE && orientation == GrB_COLMAJOR &&
        ncols == 1) ;
}

