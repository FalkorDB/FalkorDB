//------------------------------------------------------------------------------
// GB_callback_proto.h: prototypes for functions for kernel callbacks
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Prototypes for kernel callbacks.  The JIT kernels are passed a struct
// containing pointers to all these functions, so that they do not have to be
// linked against libgraphblas.so when they are compiled.

//------------------------------------------------------------------------------

#ifndef GB_CALLBACK_PROTO_H
#define GB_CALLBACK_PROTO_H

#define GB_CALLBACK_SAXPY3_CUMSUM_PROTO(GX_AxB_saxpy3_cumsum)               \
void GX_AxB_saxpy3_cumsum                                                   \
(                                                                           \
    GrB_Matrix C,               /* finalize C->p */                         \
    GB_saxpy3task_struct *SaxpyTasks, /* list of tasks, and workspace */    \
    int nfine,                  /* number of fine tasks */                  \
    double chunk,               /* chunk size */                            \
    int nthreads,               /* number of threads */                     \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_BITMAP_M_SCATTER_PROTO(GX_bitmap_M_scatter)             \
void GX_bitmap_M_scatter        /* scatter M into the C bitmap */           \
(                                                                           \
    /* input/output: */                                                     \
    GrB_Matrix C,                                                           \
    /* inputs: */                                                           \
    const GrB_Index *I,         /* I index list */                          \
    const int64_t nI,                                                       \
    const int Ikind,                                                        \
    const int64_t Icolon [3],                                               \
    const GrB_Index *J,         /* J index list */                          \
    const int64_t nJ,                                                       \
    const int Jkind,                                                        \
    const int64_t Jcolon [3],                                               \
    const GrB_Matrix M,         /* mask to scatter into the C bitmap */     \
    const bool Mask_struct,     /* true: M structural, false: M valued */   \
    const int assign_kind,      /* row assign, col assign, assign, sub. */  \
    const int operation,        /* +=2, -=2, or %=2 */                      \
    const int64_t *M_ek_slicing,    /* size 3*M_ntasks+1 */                 \
    const int M_ntasks,                                                     \
    const int M_nthreads                                                    \
)

#define GB_CALLBACK_BITMAP_M_SCATTER_WHOLE_PROTO(GX_bitmap_M_scatter_whole) \
void GX_bitmap_M_scatter_whole  /* scatter M into the C bitmap */           \
(                                                                           \
    /* input/output: */                                                     \
    GrB_Matrix C,                                                           \
    /* inputs: */                                                           \
    const GrB_Matrix M,         /* mask to scatter into the C bitmap */     \
    const bool Mask_struct,     /* true: M structural, false: M valued */   \
    const int operation,        /* +=2, -=2, or %=2 */                      \
    const int64_t *M_ek_slicing, /* size 3*M_ntasks+1 */                    \
    const int M_ntasks,                                                     \
    const int M_nthreads                                                    \
)

#define GB_CALLBACK_BIX_ALLOC_PROTO(GX_bix_alloc)                           \
GrB_Info GX_bix_alloc       /* allocate A->b, A->i, and A->x in a matrix */ \
(                                                                           \
    GrB_Matrix A,           /* matrix to allocate space for */              \
    const GrB_Index nzmax,  /* number of entries the matrix can hold; */    \
                            /* ignored if A is iso and full */              \
    const int sparsity,     /* sparse (=hyper/auto) / bitmap / full */      \
    const bool bitmap_calloc,   /* if true, calloc A->b, else use malloc */ \
    const bool numeric,     /* if true, allocate A->x, else A->x is NULL */ \
    const bool A_iso        /* if true, allocate A as iso */                \
)

#define GB_CALLBACK_EK_SLICE_PROTO(GX_ek_slice)                             \
void GX_ek_slice            /* slice a matrix */                            \
(                                                                           \
    /* output: */                                                           \
    int64_t *restrict A_ek_slicing, /* size 3*ntasks+1 */                   \
    /* input: */                                                            \
    GrB_Matrix A,                   /* matrix to slice */                   \
    int ntasks                      /* # of tasks */                        \
)

#define GB_CALLBACK_EK_SLICE_MERGE1_PROTO(GX_ek_slice_merge1)               \
void GX_ek_slice_merge1     /* merge column counts for the matrix C */      \
(                                                                           \
    /* input/output: */                                                     \
    int64_t *restrict Cp,               /* column counts */                 \
    /* input: */                                                            \
    const int64_t *restrict Wfirst,     /* size A_ntasks */                 \
    const int64_t *restrict Wlast,      /* size A_ntasks */                 \
    const int64_t *A_ek_slicing,        /* size 3*A_ntasks+1 */             \
    const int A_ntasks                  /* # of tasks */                    \
)

#define GB_CALLBACK_FREE_MEMORY_PROTO(GX_free_memory)                       \
void GX_free_memory         /* free memory */                               \
(                                                                           \
    /* input/output*/                                                       \
    void **p,               /* pointer to block of memory to free */        \
    /* input */                                                             \
    size_t size_allocated   /* # of bytes actually allocated */             \
)

#define GB_CALLBACK_MALLOC_MEMORY_PROTO(GX_malloc_memory)                   \
void *GX_malloc_memory      /* pointer to allocated block of memory */      \
(                                                                           \
    size_t nitems,          /* number of items to allocate */               \
    size_t size_of_item,    /* sizeof each item */                          \
    /* output */                                                            \
    size_t *size_allocated  /* # of bytes actually allocated */             \
)

#define GB_CALLBACK_MEMSET_PROTO(GX_memset)                                 \
void GX_memset                  /* parallel memset */                       \
(                                                                           \
    void *dest,                 /* destination */                           \
    const int c,                /* value to to set */                       \
    size_t n,                   /* # of bytes to set */                     \
    int nthreads                /* max # of threads to use */               \
)

#define GB_CALLBACK_QSORT_1_PROTO(GX_qsort_1)                               \
void GX_qsort_1    /* sort array A of size 1-by-n */                        \
(                                                                           \
    int64_t *restrict A_0,      /* size n array */                          \
    const int64_t n                                                         \
)

#define GB_CALLBACK_WERK_POP_PROTO(GX_werk_pop)                             \
void *GX_werk_pop     /* free the top block of werkspace memory */          \
(                                                                           \
    /* input/output */                                                      \
    void *p,                    /* werkspace to free */                     \
    size_t *size_allocated,     /* # of bytes actually allocated for p */   \
    /* input */                                                             \
    bool on_stack,              /* true if werkspace is from Werk stack */  \
    size_t nitems,              /* # of items to allocate */                \
    size_t size_of_item,        /* size of each item */                     \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_WERK_PUSH_PROTO(GX_werk_push)                           \
void *GX_werk_push    /* return pointer to newly allocated space */         \
(                                                                           \
    /* output */                                                            \
    size_t *size_allocated,     /* # of bytes actually allocated */         \
    bool *on_stack,             /* true if werkspace is from Werk stack */  \
    /* input */                                                             \
    size_t nitems,              /* # of items to allocate */                \
    size_t size_of_item,        /* size of each item */                     \
    GB_Werk Werk                                                            \
)

#endif

