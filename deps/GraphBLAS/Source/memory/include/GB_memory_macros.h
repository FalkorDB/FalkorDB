//------------------------------------------------------------------------------
// GB_memory_macros.h: memory allocation macros
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MEMORY_MACROS_H
#define GB_MEMORY_MACROS_H

//------------------------------------------------------------------------------
// memory arenas
//------------------------------------------------------------------------------

// The 8-byte mem (p_mem when refering to an object p) of a malloc'd object
// contains the arena in the high order byte, and the memsize in the lower 7
// bytes.

#define GB_ARENA_RMM 0          /* fixme for CUDA: Rapids will be on arena 1 */

GB_STATIC_INLINE_BOTH int GB_arena (uint64_t mem)
{
    // return the high order byte, containing the arena
    int arena = (mem >> 56) ;
    return (arena) ;
}

GB_STATIC_INLINE_BOTH uint64_t GB_memsize (uint64_t mem)
{
    // return the 7 low order bytes, containing the memsize
    uint64_t memsize = mem & ((uint64_t) 0x00ffffffffffffffL) ;
    return (memsize) ;
}

GB_STATIC_INLINE_BOTH uint64_t GB_mem (int arena, uint64_t memsize)
{
    // combine the arena and memsize into the _mem state
    uint64_t mem = ((uint64_t) arena) << 56 | memsize ;
    return (mem) ;
}

//------------------------------------------------------------------------------
// malloc/calloc/realloc/free: for permanent contents of GraphBLAS objects
//------------------------------------------------------------------------------

#ifdef GB_MEMDUMP

    #define GBMDUMP(...) GBDUMP (__VA_ARGS__)

    #define GB_FREE_MEMORY(p,mem)                                           \
    {                                                                       \
        if (p != NULL && (*(p)) != NULL)                                    \
        {                                                                   \
            uint64_t memsize = GB_memsize (mem) ;                           \
            int arena = GB_arena (mem) ;                                    \
            GBMDUMP ("free    %p %8lu arena:%d (%s, line %d)\n",            \
                (void *) (*p), memsize, arena, __FILE__, __LINE__) ;        \
        }                                                                   \
        GB_free_memory ((void **) p, mem) ;                                 \
    }

    #define GB_MALLOC_MEMORY(n,sizeof_type,mem)                             \
        GB_malloc_memory (n, sizeof_type, mem) ;                            \
        GBMDUMP ("did malloc: (%s, line %d)\n", __FILE__, __LINE__)

    #define GB_CALLOC_MEMORY(n,sizeof_type,mem)                             \
        GB_calloc_memory (n, sizeof_type, mem) ;                            \
        GBMDUMP ("did calloc: (%s, line %d)\n", __FILE__, __LINE__)

    #define GB_REALLOC_MEMORY(p,nnew,sizeof_type,mem,ok)                    \
    {                                                                       \
        p = GB_realloc_memory (nnew, sizeof_type,                           \
            (void *) p, mem, ok) ;                                          \
        GBMDUMP ("did realloc (%s, line %d)\n", __FILE__, __LINE__) ;       \
    }

    #define GB_XALLOC_MEMORY(use_calloc,iso,n,sizeof_type,mem)              \
        GB_xalloc_memory (use_calloc, iso, n, sizeof_type, mem) ;           \
        GBMDUMP ("did xalloc (%s, line %d)\n", __FILE__, __LINE__)

#else

    #define GBMDUMP(...)

    #define GB_FREE_MEMORY(p,mem)                                           \
        GB_free_memory ((void **) p, mem)

    #define GB_MALLOC_MEMORY(n,sizeof_type,mem)                             \
        GB_malloc_memory (n, sizeof_type, mem)

    #define GB_CALLOC_MEMORY(n,sizeof_type,mem)                             \
        GB_calloc_memory (n, sizeof_type, mem)

    #define GB_REALLOC_MEMORY(p,nnew,sizeof_type,mem,ok)                    \
    {                                                                       \
        p = GB_realloc_memory (nnew, sizeof_type, (void *) p, mem, ok) ;    \
    }

    #define GB_XALLOC_MEMORY(use_calloc,iso,n,sizeof_type,mem)              \
        GB_xalloc_memory (use_calloc, iso, n, sizeof_type, mem)

#endif

#endif

