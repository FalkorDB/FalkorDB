//------------------------------------------------------------------------------
// GB_memory_macros.h: memory allocation macros
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MEMORY_MACROS_H
#define GB_MEMORY_MACROS_H

//------------------------------------------------------------------------------
// malloc/calloc/realloc/free: for permanent contents of GraphBLAS objects
//------------------------------------------------------------------------------

#ifdef GB_MEMDUMP

    #define GB_FREE(p,s)                                            \
    {                                                               \
        if (p != NULL && (*(p)) != NULL)                            \
        {                                                           \
            printf ("free (%s, line %d): %p size %lu\n", /* MEMDUMP */ \
                __FILE__, __LINE__, (*p), s) ;                      \
        }                                                           \
        GB_free_memory ((void **) p, s) ;                           \
    }

    #define GB_CALLOC(n,type,s)                                     \
        (type *) GB_calloc_memory (n, sizeof (type), s) ;           \
        ; printf ("calloc  (%s, line %d): size %lu\n",  /* MEMDUMP */ \
            __FILE__, __LINE__, *(s)) ;

    #define GB_MALLOC(n,type,s)                                     \
        (type *) GB_malloc_memory (n, sizeof (type), s) ;           \
        ; printf ("malloc  (%s, line %d): size %lu\n", /* MEMDUMP */    \
            __FILE__, __LINE__, *(s)) ;

    #define GB_REALLOC(p,nnew,type,s,ok)                            \
        p = (type *) GB_realloc_memory (nnew, sizeof (type),        \
            (void *) p, s, ok) ;                                    \
        ; printf ("realloc (%s, line %d): size %lu\n", /* MEMDUMP */    \
            __FILE__, __LINE__, *(s)) ;

    #define GB_XALLOC(use_calloc,iso,n,type_size,s)                 \
        GB_xalloc_memory (use_calloc, iso, n, type_size, s) ;       \
        ; printf ("xalloc (%s, line %d): size %lu\n", /* MEMDUMP */     \
            __FILE__, __LINE__, *(s)) ;

#else

    #define GB_FREE(p,s)                                            \
        GB_free_memory ((void **) p, s)

    #define GB_CALLOC(n,type,s)                                     \
        (type *) GB_calloc_memory (n, sizeof (type), s)         

    #define GB_MALLOC(n,type,s)                                     \
        (type *) GB_malloc_memory (n, sizeof (type), s)

    #define GB_REALLOC(p,nnew,type,s,ok)                            \
        p = (type *) GB_realloc_memory (nnew, sizeof (type),        \
            (void *) p, s, ok)

    #define GB_XALLOC(use_calloc,iso,n,type_size,s)                 \
        GB_xalloc_memory (use_calloc, iso, n, type_size, s)

#endif

//------------------------------------------------------------------------------
// malloc/calloc/realloc/free: for workspace
//------------------------------------------------------------------------------

// These macros currently do the same thing as the 4 macros above, but that may
// change in the future.  Even if they always do the same thing, it's useful to
// tag the source code for the allocation of workspace differently from the
// allocation of permament space for a GraphBLAS object, such as a GrB_Matrix.

#define GB_CALLOC_WORK(n,type,s) GB_CALLOC(n,type,s)
#define GB_MALLOC_WORK(n,type,s) GB_MALLOC(n,type,s)
#define GB_REALLOC_WORK(p,nnew,type,s,ok) GB_REALLOC(p,nnew,type,s,ok) 
#define GB_FREE_WORK(p,s) GB_FREE(p,s)

#endif

