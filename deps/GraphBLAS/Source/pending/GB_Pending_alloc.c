//------------------------------------------------------------------------------
// GB_Pending_alloc: allocate a list of pending tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "pending/GB_Pending.h"

bool GB_Pending_alloc       // create a list of pending tuples
(
    GrB_Matrix C,           // matrix to create C->Pending for
    bool iso,               // if true, do not allocate Pending->x
    GrB_Type type,          // type of pending tuples
    GrB_BinaryOp op,        // operator for assembling pending tuples
    int64_t nmax            // # of pending tuples to hold
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL) ;
    ASSERT (C->Pending == NULL) ;

    int memlane = GB_memlane (C->header_mem) ;
    uint64_t mem = GB_mem (memlane, 0) ;

    //--------------------------------------------------------------------------
    // allocate the Pending header
    //--------------------------------------------------------------------------

    uint64_t header_mem = mem ;
    GB_Pending Pending = GB_MALLOC_MEMORY (1, sizeof (struct GB_Pending_struct),
        &header_mem) ;
    if (Pending == NULL)
    { 
        // out of memory
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // initialize the contents of the Pending tuples
    //--------------------------------------------------------------------------

    nmax = GB_IMAX (nmax, GB_PENDING_INIT) ;
    Pending->header_mem = header_mem ;
    Pending->n = 0 ;                    // no pending tuples yet
    Pending->nmax = nmax ;              // initial size of list
    Pending->sorted = true ;            // keep track if tuples are sorted
    Pending->type = type ;              // type of pending tuples
    Pending->size = type->size ;        // size of pending tuple type
    Pending->op = (iso) ? NULL : op ;   // pending operator (NULL is OK)
    Pending->i_mem = mem ;
    Pending->j_mem = mem ;
    Pending->x_mem = mem ;

    bool is_matrix = (C->vdim > 1) ;
    size_t jsize = (C->j_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize = (C->i_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

    Pending->i = GB_MALLOC_MEMORY (nmax, isize, &(Pending->i_mem)) ;
    Pending->j = NULL ;
    if (is_matrix)
    { 
        Pending->j = GB_MALLOC_MEMORY (nmax, jsize, &(Pending->j_mem)) ;
    }
    Pending->x = NULL ;
    if (!iso)
    { 
        Pending->x = GB_MALLOC_MEMORY (nmax, Pending->size, &(Pending->x_mem)) ;
    }

    if (Pending->i == NULL
        || (!iso && Pending->x == NULL)
        || (is_matrix && Pending->j == NULL))
    { 
        // out of memory
        GB_Pending_free (&Pending) ;
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->Pending = Pending ;
    return (true) ;
}

