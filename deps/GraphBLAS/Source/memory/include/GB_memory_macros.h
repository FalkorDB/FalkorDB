//------------------------------------------------------------------------------
// GB_memory_macros.h: memory allocation macros
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MEMORY_MACROS_H
#define GB_MEMORY_MACROS_H

//------------------------------------------------------------------------------
// memory lanes
//------------------------------------------------------------------------------

// The 8-byte mem (p_mem when refering to an object p) of a malloc'd object
// contains the memlane in the high order byte, and the memsize in the lower 7
// bytes.

#define GB_MEMLANES 4           /* total # of memlanes */
#define GB_MEMLANE_DEFAULT 0
#define GB_MEMLANE_RMM 0        /* FIXME: Rapids will be on lane 1 */
#define GB_MEMLANE_MATLAB 0     /* FIXME: mxMalloc will be on lane 2 */

GB_STATIC_INLINE int GB_memlane (uint64_t mem)
{
    // return the high order byte, containing the memlane
    int memlane = (mem >> 56) ;
    return (memlane) ;
}

GB_STATIC_INLINE uint64_t GB_memsize (uint64_t mem)
{
    // return the 7 low order bytes, containing the memsize
    uint64_t memsize = mem & ((uint64_t) 0x00ffffffffffffffL) ;
    return (memsize) ;
}

GB_STATIC_INLINE uint64_t GB_mem (int memlane, uint64_t memsize)
{
    // combine the memlane and memsize into the _mem state
    uint64_t mem = ((uint64_t) memlane) << 56 | memsize ;
    return (mem) ;
}

GB_STATIC_INLINE uint64_t GB_memlane_change (int memlane, uint64_t mem)
{
    // change the memlane of an object, keeping the memsize the same,
    // and return the new mem state
    uint64_t memsize = GB_memsize (mem) ;
    return (GB_mem (memlane, memsize)) ;
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
            int memlane = GB_memlane (mem) ;                                \
            GBMDUMP ("free    %p %8lu lane:%d (%s, line %d)\n",             \
                (void *) (*p), memsize, memlane, __FILE__, __LINE__) ;      \
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

