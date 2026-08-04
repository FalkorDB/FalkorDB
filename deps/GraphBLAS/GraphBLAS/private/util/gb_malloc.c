//------------------------------------------------------------------------------
// gb_malloc: allocate space in a given arena
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

typedef void * (*malloc_t) (size_t) ;

void *gb_malloc (size_t n, int arena)
{ 
    // allocate memory in the arena; at least 8 bytes
    malloc_t malloc_f = NULL ;
    GrB_Global_get_VOID (GrB_GLOBAL, &malloc_f, GxB_ARENA_MALLOC + arena) ;
    void *p = NULL ;
    if (malloc_f != NULL)
    { 
        p = malloc_f (MAX (n, sizeof (uint64_t))) ;
    }
    return (p) ;
}

