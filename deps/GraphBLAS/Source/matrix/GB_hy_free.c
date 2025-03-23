//------------------------------------------------------------------------------
// GB_hy_free: free A->h and A->Y
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

void GB_hy_free                 // free A-h and A->Y of a matrix
(
    GrB_Matrix A                // matrix with content to free
)
{ 

    if (A != NULL)
    {
        // free A->h unless it is shallow
        if (!A->h_shallow)
        { 
            GB_FREE_MEMORY (&(A->h), A->h_size) ;
        }
        A->h = NULL ;
        A->h_size = 0 ;
        A->h_shallow = false ;
        GB_hyper_hash_free (A) ;
    }
}

