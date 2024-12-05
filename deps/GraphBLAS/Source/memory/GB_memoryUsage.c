//------------------------------------------------------------------------------
// GB_memoryUsage: # of bytes used for a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

void GB_memoryUsage         // count # allocated blocks and their sizes
(
    int64_t *nallocs,       // # of allocated memory blocks
    size_t *mem_deep,       // # of bytes in blocks owned by this matrix
    size_t *mem_shallow,    // # of bytes in blocks owned by another matrix
    const GrB_Matrix A,     // matrix to query
    bool count_hyper_hash   // if true, include A->Y
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (nallocs != NULL) ;
    ASSERT (mem_deep != NULL) ;
    ASSERT (mem_shallow != NULL) ;

    //--------------------------------------------------------------------------
    // count the allocated blocks and their sizes
    //--------------------------------------------------------------------------

    // a matrix contains 0 to 10 dynamically malloc'd blocks, not including
    // A->Y
    (*nallocs) = 0 ;
    (*mem_deep) = 0 ;
    (*mem_shallow) = 0 ;

    if (A == NULL)
    { 
        #pragma omp flush
        return ;
    }

    GB_Pending Pending = A->Pending ;

    if (!A->static_header)
    { 
        (*nallocs)++ ;
        (*mem_deep) += A->header_size ;
    }

    if (A->p != NULL)
    {
        if (A->p_shallow)
        { 
            (*mem_shallow) += A->p_size ;
        }
        else
        { 
            (*nallocs)++ ;
            (*mem_deep) += A->p_size ;
        }
    }

    if (A->h != NULL)
    {
        if (A->h_shallow)
        { 
            (*mem_shallow) += A->h_size ;
        }
        else
        { 
            (*nallocs)++ ;
            (*mem_deep) += A->h_size ;
        }
    }

    if (A->b != NULL)
    {
        if (A->b_shallow)
        { 
            (*mem_shallow) += A->b_size ;
        }
        else
        { 
            (*nallocs)++ ;
            (*mem_deep) += A->b_size ;
        }
    }

    if (A->i != NULL)
    {
        if (A->i_shallow)
        { 
            (*mem_shallow) += A->i_size ;
        }
        else
        { 
            (*nallocs)++ ;
            (*mem_deep) += A->i_size ;
        }
    }

    if (A->x != NULL)
    {
        if (A->x_shallow)
        { 
            (*mem_shallow) += A->x_size ;
        }
        else
        { 
            (*nallocs)++ ;
            (*mem_deep) += A->x_size ;
        }
    }

    if (Pending != NULL)
    { 
        (*nallocs)++ ;
        (*mem_deep) += Pending->header_size ;
    }

    if (Pending != NULL && Pending->i != NULL)
    { 
        (*nallocs)++ ;
        (*mem_deep) += Pending->i_size ;
    }

    if (Pending != NULL && Pending->j != NULL)
    { 
        (*nallocs)++ ;
        (*mem_deep) += Pending->j_size ;
    }

    if (Pending != NULL && Pending->x != NULL)
    { 
        (*nallocs)++ ;
        (*mem_deep) += Pending->x_size ;
    }

    if (count_hyper_hash && A->Y != NULL)
    {
        int64_t Y_nallocs = 0 ;
        size_t Y_mem_deep = 0 ;
        size_t Y_mem_shallow = 0 ;
        GB_memoryUsage (&Y_nallocs, &Y_mem_deep, &Y_mem_shallow, A->Y, false) ;
        if (A->Y_shallow)
        { 
            // all of A->Y is shallow
            (*mem_shallow) += Y_mem_shallow + Y_mem_deep ;
        }
        else
        { 
            // A->Y itself is not shallow, but may contain shallow content
            (*nallocs) += Y_nallocs ;
            (*mem_deep) += Y_mem_deep ;
            (*mem_shallow) += Y_mem_shallow ;
        }
    }

    #pragma omp flush
    return ;
}

