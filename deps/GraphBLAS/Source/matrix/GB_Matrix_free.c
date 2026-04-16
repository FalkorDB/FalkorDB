//------------------------------------------------------------------------------
// GB_Matrix_free: free a GrB_Matrix or GrB_Vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Free all the content of a matrix.  After GB_Matrix_free (&A), the header A
// is freed and set to NULL if the header of A was originally dynamically
// allocated.  Otherwise, A is not freed.

#include "GB.h"

// prototype is in Source/callbacks/GB_callbacks.h

#if 0
void GB_Matrix_free             // free a matrix
(
    GrB_Matrix *Ahandle         // handle of matrix to free
)
#endif

GB_CALLBACK_MATRIX_FREE_PROTO (GB_Matrix_free)
{
    if (Ahandle != NULL)
    {
        GrB_Matrix A = *Ahandle ;
        if (A != NULL && (A->magic == GB_MAGIC || A->magic == GB_MAGIC2))
        {
            // free all content of A
            GB_FREE_MEMORY (&(A->user_name), A->user_name_mem) ;
            uint64_t header_mem = A->header_mem ;
            uint64_t header_memsize = GB_memsize (header_mem) ;
            GB_phybix_free (A) ;
            if (header_memsize != 0)
            { 
                // free the header of A itself, unless it is static
                A->magic = GB_FREED ;   // to help detect dangling pointers
                A->header_mem = 0 ;     // header is freed
                GB_FREE_MEMORY (Ahandle, header_mem) ;
                (*Ahandle) = NULL ;
            }
        }
    }
}

