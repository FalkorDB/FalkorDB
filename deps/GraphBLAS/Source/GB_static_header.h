//------------------------------------------------------------------------------
// GB_static_header.h: macros for allocating static headers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// By default, many internal temporary matrices use statically allocated
// headers to reduce the number of calls to malloc/free.  This works fine for
// matrices on the CPU, but the static headers do not get automatically
// transfered to the GPU.  Only dynamically allocated headers, allocated by
// rmm_wrap_malloc, get transfered.  Set this to 1 to turn off static headers
// (required for CUDA; see GB_static_headers.h).  Leave static headers
// enabled by default by leaving this commented out or setting GBNSTATIC to 0.
// #undef  GBNSTATIC
// #define GBNSTATIC 1

#ifndef GBNSTATIC
    #if defined ( SUITESPARSE_CUDA )
    #define GBNSTATIC 1
    #else
    #define GBNSTATIC 0
    #endif
#endif

#undef GB_CLEAR_STATIC_HEADER

#if GBNSTATIC

    // do not use any static headers
    #define GB_CLEAR_STATIC_HEADER(XX,XX_header_handle)                     \
    {                                                                       \
        size_t XX_size ;                                                    \
        XX = GB_CALLOC (1, struct GB_Matrix_opaque, &XX_size) ;             \
        if (XX == NULL)                                                     \
        {                                                                   \
            GB_FREE_ALL ;                                                   \
            return (GrB_OUT_OF_MEMORY) ;                                    \
        }                                                                   \
        XX->static_header = false ;                                         \
        XX->header_size = XX_size ;                                         \
        XX->magic = GB_MAGIC2 ;                                             \
    }

#else

    // use static headers
    #define GB_CLEAR_STATIC_HEADER(XX,XX_header_handle)                     \
    {                                                                       \
        XX = GB_clear_static_header (XX_header_handle) ;                    \
    }

#endif

#ifndef GB_STATIC_HEADER_H
#define GB_STATIC_HEADER_H

static inline GrB_Matrix GB_clear_static_header // clear a static header
(
    GrB_Matrix C    // static header to clear
)
{
    ASSERT (C != NULL) ;
    memset (C, 0, sizeof (struct GB_Matrix_opaque)) ;
    C->static_header = true ;
    return (C) ;
}

#endif

