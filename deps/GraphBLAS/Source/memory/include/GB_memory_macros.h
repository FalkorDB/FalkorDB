//------------------------------------------------------------------------------
// GB_memory_macros.h: memory allocation macros
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MEMORY_MACROS_H
#define GB_MEMORY_MACROS_H

//------------------------------------------------------------------------------
// malloc/calloc/realloc/free: for permanent contents of GraphBLAS objects
//------------------------------------------------------------------------------

#ifdef GB_MEMDUMP

    #define GBMDUMP(...) GBDUMP (__VA_ARGS__)

    #define GB_FREE_MEMORY(p,s)                                             \
    {                                                                       \
        if (p != NULL && (*(p)) != NULL)                                    \
        {                                                                   \
            GBMDUMP ("free    %p %8ld: (%s, line %d)\n",                    \
                (void *) (*p), s, __FILE__, __LINE__) ;                     \
        }                                                                   \
        GB_free_memory ((void **) p, s) ;                                   \
    }

    #define GB_MALLOC_MEMORY(n,sizeof_type,s)                               \
        GB_malloc_memory (n, sizeof_type, s) ;                              \
        GBMDUMP ("did malloc: (%s, line %d)\n", __FILE__, __LINE__)

    #define GB_CALLOC_MEMORY(n,sizeof_type,s)                               \
        GB_calloc_memory (n, sizeof_type, s) ;                              \
        GBMDUMP ("did calloc: (%s, line %d)\n", __FILE__, __LINE__)

    #define GB_REALLOC_MEMORY(p,nnew,sizeof_type,s,ok)                      \
    {                                                                       \
        p = GB_realloc_memory (nnew, sizeof_type,                           \
            (void *) p, s, ok) ;                                            \
        GBMDUMP ("did realloc (%s, line %d)\n", __FILE__, __LINE__) ;                   \
    }

    #define GB_XALLOC_MEMORY(use_calloc,iso,n,sizeof_type,s)                \
        GB_xalloc_memory (use_calloc, iso, n, sizeof_type, s) ;             \
        GBMDUMP ("did xalloc (%s, line %d)\n", __FILE__, __LINE__)

#else

    #define GBMDUMP(...)

    #define GB_FREE_MEMORY(p,s)                                             \
        GB_free_memory ((void **) p, s)

    #define GB_MALLOC_MEMORY(n,sizeof_type,s)                               \
        GB_malloc_memory (n, sizeof_type, s)

    #define GB_CALLOC_MEMORY(n,sizeof_type,s)                               \
        GB_calloc_memory (n, sizeof_type, s)

    #define GB_REALLOC_MEMORY(p,nnew,sizeof_type,s,ok)                      \
    {                                                                       \
        p = GB_realloc_memory (nnew, sizeof_type, (void *) p, s, ok) ;      \
    }

    #define GB_XALLOC_MEMORY(use_calloc,iso,n,sizeof_type,s)                \
        GB_xalloc_memory (use_calloc, iso, n, sizeof_type, s)

#endif

#endif

