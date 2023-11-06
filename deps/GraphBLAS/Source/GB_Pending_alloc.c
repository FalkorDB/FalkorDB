//------------------------------------------------------------------------------
// GB_Pending_alloc: allocate a list of pending tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_Pending.h"

bool GB_Pending_alloc       // create a list of pending tuples
(
    GB_Pending *PHandle,    // output
    bool iso,               // if true, do not allocate Pending->x
    GrB_Type type,          // type of pending tuples
    GrB_BinaryOp op,        // operator for assembling pending tuples
    bool is_matrix,         // true if Pending->j must be allocated
    int64_t nmax            // # of pending tuples to hold
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (PHandle != NULL) ;
    (*PHandle) = NULL ;

    //--------------------------------------------------------------------------
    // allocate the Pending header
    //--------------------------------------------------------------------------

    size_t header_size ;
    GB_Pending Pending = GB_MALLOC (1, struct GB_Pending_struct, &header_size) ;
    if (Pending == NULL)
    { 
        // out of memory
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // initialize the contents of the Pending tuples
    //--------------------------------------------------------------------------

    nmax = GB_IMAX (nmax, GB_PENDING_INIT) ;
    Pending->header_size = header_size ;
    Pending->n = 0 ;                    // no pending tuples yet
    Pending->nmax = nmax ;              // initial size of list
    Pending->sorted = true ;            // keep track if tuples are sorted
    Pending->type = type ;              // type of pending tuples
    Pending->size = type->size ;        // size of pending tuple type
    Pending->op = (iso) ? NULL : op ;   // pending operator (NULL is OK)
    Pending->i_size = 0 ;
    Pending->j_size = 0 ;
    Pending->x_size = 0 ;

    Pending->i = GB_MALLOC (nmax, int64_t, &(Pending->i_size)) ;
    Pending->j = NULL ;
    if (is_matrix)
    { 
        Pending->j = GB_MALLOC (nmax, int64_t, &(Pending->j_size)) ;
    }
    Pending->x = NULL ;
    if (!iso)
    { 
        Pending->x = GB_MALLOC (nmax * Pending->size, GB_void,  // x:OK
            &(Pending->x_size)) ;
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

    (*PHandle) = Pending ;
    return (true) ;
}

