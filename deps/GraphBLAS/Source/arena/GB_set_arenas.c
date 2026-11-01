//------------------------------------------------------------------------------
// GB_set_arenas: set the arenas (header and data) of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_set_arenas          // modify all arenas of a matrix
(
    // input/output
    GrB_Matrix *Ahandle,        // handle of matrix to modify
    // input
    const int new_header_arena, // new arena for the header of A
    const int new_data_arena    // new arena for the data content of A
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    if (Ahandle == NULL || *Ahandle == NULL)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    #define GB_FREE_ALL ;
    GB_OK (GB_check_arena (new_header_arena)) ;
    GB_OK (GB_check_arena (new_data_arena)) ;
    #undef  GB_FREE_ALL
    #define GB_FREE_ALL GrB_Matrix_free (Ahandle) ;

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A = (*Ahandle) ;
    uint64_t header_mem = A->header_mem ;
    int old_header_arena = GB_arena (header_mem) ;
    int old_data_arena = A->data_arena ;
    int64_t anvec = A->nvec ;
    int64_t anz = GB_nnz_held (A) ;

    size_t psize = A->p_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t jsize = A->j_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize = A->i_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    int nthreads = GB_Context_nthreads_max ( ) ;

    //--------------------------------------------------------------------------
    // set the arena of the header of A to the new header arena
    //--------------------------------------------------------------------------

    uint64_t n = sizeof (struct GB_Matrix_opaque) ;
    GB_OK (GB_set_arena ((void **) Ahandle, &header_mem, new_header_arena, n, n,
        nthreads)) ;
    A = (*Ahandle) ;
    A->header_mem = header_mem ;

    //--------------------------------------------------------------------------
    // set the arena of A->[phbix] to the new data arena
    //--------------------------------------------------------------------------

    if (!A->p_shallow)
    { 
        n = (anvec+1) * psize ;
        GB_OK (GB_set_arena (&(A->p), &(A->p_mem), new_data_arena, n, n,
            nthreads)) ;
    }

    if (!A->h_shallow)
    { 
        n = anvec * jsize ;
        GB_OK (GB_set_arena (&(A->h), &(A->h_mem), new_data_arena, n, n,
            nthreads)) ;
    }

    if (!A->b_shallow)
    { 
        n = anz * sizeof (int8_t) ;
        GB_OK (GB_set_arena ((void **) &(A->b), &(A->b_mem), new_data_arena,
            n, n, nthreads)) ;
    }

    if (!A->i_shallow)
    { 
        n = anz * isize ;
        GB_OK (GB_set_arena (&(A->i), &(A->i_mem), new_data_arena, n, n,
            nthreads)) ;
    }

    if (!A->x_shallow)
    { 
        n = (A->iso ? 1:anz) * (A->type->size) ;
        GB_OK (GB_set_arena (&(A->x), &(A->x_mem), new_data_arena, n, n,
            nthreads)) ;
    }

    //--------------------------------------------------------------------------
    // set the arenas of the hyperhash to the new data arena
    //--------------------------------------------------------------------------

    if (!A->Y_shallow)
    { 
        GB_OK (GB_set_arenas (&(A->Y), new_data_arena, new_data_arena)) ;
    }

    //--------------------------------------------------------------------------
    // set the arenas of the Pending tuples to the new data arena
    //--------------------------------------------------------------------------

    GB_Pending Pending = A->Pending ;
    if (Pending != NULL)
    { 
        uint64_t P_header_mem = Pending->header_mem ;
        n = sizeof (struct GB_Pending_struct) ;
        GB_OK (GB_set_arena ((void **) &(A->Pending), &P_header_mem,
            new_data_arena, n, n, nthreads)) ;
        Pending = A->Pending ;
        Pending->header_mem = P_header_mem ;
        int64_t nmax = Pending->nmax ;
        n = Pending->n ;

        // the i,j,x arrays contain n tuples but can hold nmax tuples,
        // where nmax >= n
        ASSERT (nmax >= n) ;

        GB_OK (GB_set_arena (&(Pending->i), &(Pending->i_mem), new_data_arena,
            nmax * isize, n * isize, nthreads)) ;

        GB_OK (GB_set_arena (&(Pending->j), &(Pending->j_mem), new_data_arena,
            nmax * jsize, n * jsize, nthreads)) ;

        GB_OK (GB_set_arena ((void **) &(Pending->x), &(Pending->x_mem),
            new_data_arena, nmax * Pending->size, n * Pending->size,
            nthreads)) ;
    }

    //--------------------------------------------------------------------------
    // set the arenas of the user name and error logger (in header arena of A)
    //--------------------------------------------------------------------------

    n = GB_memsize (A->user_name_mem) ;
    GB_OK (GB_set_arena ((void **) &(A->user_name), &(A->user_name_mem),
        new_header_arena, n, n, nthreads)) ;

    n = GB_LOGGER_LEN + 1 ;
    GB_OK (GB_set_arena ((void **) &(A->logger), &(A->logger_mem),
        new_header_arena, n, n, nthreads)) ;

    //--------------------------------------------------------------------------
    // revise the final data arena and return result
    //--------------------------------------------------------------------------

    A->data_arena = new_data_arena ;
    (*Ahandle) = A ;
    return (GrB_SUCCESS) ;
}

