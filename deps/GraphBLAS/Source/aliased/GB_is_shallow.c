//------------------------------------------------------------------------------
// GB_is_shallow: determine if a GrB_matrix has any shallow components
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

bool GB_is_shallow              // true if any component of A is shallow
(
    GrB_Matrix A                // matrix to query
)
{

    if (A == NULL)
    { 
        // a NULL pointer is not shallow
        return (false) ;
    }
    else
    { 
        // check if any component of A is shallow
        return (A->p_shallow || A->h_shallow || A->b_shallow ||
                A->i_shallow || A->x_shallow || A->Y_shallow ||
                GB_is_shallow (A->Y)) ;
    }
}

