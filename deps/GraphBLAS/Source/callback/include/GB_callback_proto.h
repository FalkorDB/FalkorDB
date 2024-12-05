//------------------------------------------------------------------------------
// GB_callback_proto.h: prototypes for functions for kernel callbacks
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
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

#define GB_CALLBACK_BITMAP_M_SCATTER_WHOLE_PROTO(GX_bitmap_M_scatter_whole) \
void GX_bitmap_M_scatter_whole  /* scatter M into the C bitmap */           \
(                                                                           \
    /* input/output: */                                                     \
    GrB_Matrix C,                                                           \
    /* inputs: */                                                           \
    const GrB_Matrix M,         /* mask to scatter into the C bitmap */     \
    const bool Mask_struct,     /* true: M structural, false: M valued */   \
    const int operation,        /* +=2, -=2, or =2 */                       \
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

#define GB_CALLBACK_HYPER_HASH_BUILD_PROTO(GX_hyper_hash_build)             \
GrB_Info GX_hyper_hash_build    /* construct the A->Y hyper_hash for A */   \
(                                                                           \
    GrB_Matrix A,                                                           \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_SUBASSIGN_ONE_SLICE_PROTO(GX_subassign_one_slice)       \
GrB_Info GX_subassign_one_slice     /* slice M for subassign_05, 06n, 07 */ \
(                                                                           \
    /* output: */                                                           \
    GB_task_struct **p_TaskList,    /* array of structs */                  \
    size_t *p_TaskList_size,        /* size of TaskList */                  \
    int *p_ntasks,                  /* # of tasks constructed */            \
    int *p_nthreads,                /* # of threads to use */               \
    /* input: */                                                            \
    const GrB_Matrix C,             /* output matrix C */                   \
    const GrB_Index *I,                                                     \
    const int64_t nI,                                                       \
    const int Ikind,                                                        \
    const int64_t Icolon [3],                                               \
    const GrB_Index *J,                                                     \
    const int64_t nJ,                                                       \
    const int Jkind,                                                        \
    const int64_t Jcolon [3],                                               \
    const GrB_Matrix M,             /* matrix to slice */                   \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_ADD_PHASE0_PROTO(GX_add_phase0)                         \
GrB_Info GX_add_phase0          /* find vectors in C for C=A+B or C<M>=A+B*/\
(                                                                           \
    int64_t *p_Cnvec,           /* # of vectors to compute in C */          \
    int64_t *restrict *Ch_handle,        /* Ch: size Cnvec, or NULL */      \
    size_t *Ch_size_handle,              /* size of Ch in bytes */          \
    int64_t *restrict *C_to_M_handle,    /* C_to_M: size Cnvec, or NULL */  \
    size_t *C_to_M_size_handle,          /* size of C_to_M in bytes */      \
    int64_t *restrict *C_to_A_handle,    /* C_to_A: size Cnvec, or NULL */  \
    size_t *C_to_A_size_handle,          /* size of C_to_A in bytes */      \
    int64_t *restrict *C_to_B_handle,    /* C_to_B: size Cnvec, or NULL */  \
    size_t *C_to_B_size_handle,          /* size of C_to_A in bytes */      \
    bool *p_Ch_is_Mh,           /* if true, then Ch == Mh */                \
    int *C_sparsity,            /* sparsity structure of C */               \
    const GrB_Matrix M,         /* optional mask, may be NULL; not compl */ \
    const GrB_Matrix A,         /* first input matrix */                    \
    const GrB_Matrix B,         /* second input matrix */                   \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_EWISE_SLICE_PROTO(GX_ewise_slice)                       \
GrB_Info GX_ewise_slice                                                     \
(                                                                           \
    /* output: */                                                           \
    GB_task_struct **p_TaskList,    /* array of structs */                  \
    size_t *p_TaskList_size,        /* size of TaskList */                  \
    int *p_ntasks,                  /* # of tasks constructed */            \
    int *p_nthreads,                /* # of threads for eWise operation */  \
    /* input: */                                                            \
    const int64_t Cnvec,            /* # of vectors of C */                 \
    const int64_t *restrict Ch,     /* vectors of C, if hypersparse */      \
    const int64_t *restrict C_to_M, /* mapping of C to M */                 \
    const int64_t *restrict C_to_A, /* mapping of C to A */                 \
    const int64_t *restrict C_to_B, /* mapping of C to B */                 \
    bool Ch_is_Mh,                  /* if true, then Ch == Mh; GB_add only*/\
    const GrB_Matrix M,             /* mask matrix to slice (optional) */   \
    const GrB_Matrix A,             /* matrix to slice */                   \
    const GrB_Matrix B,             /* matrix to slice */                   \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_SUBASSIGN_IXJ_SLICE_PROTO(GX_subassign_IxJ_slice)       \
GrB_Info GX_subassign_IxJ_slice                                             \
(                                                                           \
    /* output: */                                                           \
    GB_task_struct **p_TaskList,    /* array of structs */                  \
    size_t *p_TaskList_size,        /* size of TaskList */                  \
    int *p_ntasks,                  /* # of tasks constructed */            \
    int *p_nthreads,                /* # of threads to use */               \
    /* input: */                                                            \
    const int64_t nI,                                                       \
    const int64_t nJ,                                                       \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_PENDING_ENSURE_PROTO(GX_Pending_ensure)                 \
bool GX_Pending_ensure                                                      \
(                                                                           \
    GB_Pending *PHandle,    /* input/output */                              \
    bool iso,               /* if true, do not allocate Pending->x */       \
    GrB_Type type,          /* type of pending tuples */                    \
    GrB_BinaryOp op,        /* operator for assembling pending tuples */    \
    bool is_matrix,         /* true if Pending->j must be allocated */      \
    int64_t nnew,           /* # of pending tuples to add */                \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_SUBASSIGN_08N_SLICE_PROTO(GX_subassign_08n_slice)       \
GrB_Info GX_subassign_08n_slice                                             \
(                                                                           \
    /* output: */                                                           \
    GB_task_struct **p_TaskList,    /* size max_ntasks */                   \
    size_t *p_TaskList_size,        /* size of TaskList */                  \
    int *p_ntasks,                  /* # of tasks constructed */            \
    int *p_nthreads,                /* # of threads to use */               \
    int64_t *p_Znvec,               /* # of vectors to compute in Z */      \
    const int64_t *restrict *Zh_handle,  /* Zh is A->h, M->h, or NULL */    \
    int64_t *restrict *Z_to_A_handle,    /* Z_to_A: size Znvec, or NULL */  \
    size_t *Z_to_A_size_handle,                                             \
    int64_t *restrict *Z_to_M_handle,    /* Z_to_M: size Znvec, or NULL */  \
    size_t *Z_to_M_size_handle,                                             \
    /* input: */                                                            \
    const GrB_Matrix C,             /* output matrix C */                   \
    const GrB_Index *I,                                                     \
    const int64_t nI,                                                       \
    const int Ikind,                                                        \
    const int64_t Icolon [3],                                               \
    const GrB_Index *J,                                                     \
    const int64_t nJ,                                                       \
    const int Jkind,                                                        \
    const int64_t Jcolon [3],                                               \
    const GrB_Matrix A,             /* matrix to slice */                   \
    const GrB_Matrix M,             /* matrix to slice */                   \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_BITMAP_ASSIGN_TO_FULL_PROTO(GX_bitmap_assign_to_full)   \
void GX_bitmap_assign_to_full   /* set all C->b to 1, or make C full */     \
(                                                                           \
    GrB_Matrix C,                                                           \
    int nthreads_max                                                        \
)

#define GB_CALLBACK_QSORT_1_PROTO(GX_qsort_1)                               \
void GX_qsort_1    /* sort array A of size 1-by-n */                        \
(                                                                           \
    int64_t *restrict A_0,      /* size n array */                          \
    const int64_t n                                                         \
)

#define GB_CALLBACK_P_SLICE_PROTO(GX_p_slice)                               \
void GX_p_slice                     /* slice Ap */                          \
(                                                                           \
    /* output: */                                                           \
    int64_t *restrict Slice,        /* size ntasks+1 */                     \
    /* input: */                                                            \
    const int64_t *restrict Ap,     /* array size n+1 (full/bitmap: NULL)*/ \
    const int64_t n,                                                        \
    const int ntasks,               /* # of tasks */                        \
    const bool perfectly_balanced                                           \
)

#endif

