//------------------------------------------------------------------------------
// gb_free: free space in a given arena
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

typedef void (*free_t) (void *) ;

void gb_free (void **p, int arena)
{
    if (p != NULL && *p != NULL)
    { 
        free_t free_f = NULL ;
        GrB_Global_get_VOID (GrB_GLOBAL, &free_f, GxB_ARENA_FREE + arena) ;
        if (free_f != NULL)
        { 
            // free the pointer in the arena and set the pointer to NULL to
            // indicate it has been freed.
            free_f (*p) ;
            (*p) = NULL ;
        }
    }
}

