//------------------------------------------------------------------------------
// GB_callback_proto.h: prototypes for functions for kernel callbacks
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Prototypes for kernel callbacks.  The JIT kernels are passed a struct
// containing pointers to all these functions, so that they do not have to be
// linked against libgraphblas.so when they are compiled.

//------------------------------------------------------------------------------

#ifndef GB_CALLBACK_PROTO_H
#define GB_CALLBACK_PROTO_H

#define GB_CALLBACK_SAXPY3_CUMSUM_PROTO(GX_AxB_saxpy3_cumsum)               \
GrB_Info GX_AxB_saxpy3_cumsum                                               \
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
    const uint64_t nzmax,   /* number of entries the matrix can hold; */    \
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
    int64_t *restrict A_ek_slicing, /* size 3*A_ntasks+1 */                 \
    /* input: */                                                            \
    GrB_Matrix A,                   /* matrix to slice */                   \
    int A_ntasks                    /* # of tasks */                        \
)

#define GB_CALLBACK_FREE_MEMORY_PROTO(GX_free_memory)                       \
void GX_free_memory         /* free memory */                               \
(                                                                           \
    /* input/output*/                                                       \
    void **p,               /* pointer to block of memory to free */        \
    /* input */                                                             \
    uint64_t mem            /* # of bytes allocated, and memlane */         \
)

#define GB_CALLBACK_MALLOC_MEMORY_PROTO(GX_malloc_memory)                   \
void *GX_malloc_memory      /* pointer to allocated block of memory */      \
(                                                                           \
    uint64_t nitems,        /* number of items to allocate */               \
    uint64_t size_of_item,  /* sizeof each item */                          \
    /* input/output */                                                      \
    uint64_t *mem           /* # of bytes allocated, and memlane */         \
)

#define GB_CALLBACK_CALLOC_MEMORY_PROTO(GX_calloc_memory)                   \
void *GX_calloc_memory      /* pointer to allocated block of memory */      \
(                                                                           \
    uint64_t nitems,        /* number of items to allocate */               \
    uint64_t size_of_item,  /* sizeof each item */                          \
    /* input/output */                                                      \
    uint64_t *mem           /* # of bytes allocated, and memlane */         \
)

#define GB_CALLBACK_MEMSET_PROTO(GX_memset)                                 \
void GX_memset                  /* parallel memset */                       \
(                                                                           \
    void *dest,                 /* destination */                           \
    const int c,                /* value to to set */                       \
    uint64_t n,                 /* # of bytes to set */                     \
    int nthreads                /* max # of threads to use */               \
)

#define GB_CALLBACK_WERK_POP_PROTO(GX_werk_pop)                             \
void *GX_werk_pop     /* free the top block of werkspace memory */          \
(                                                                           \
    /* input/output */                                                      \
    void *p,                    /* werkspace to free */                     \
    uint64_t *mem,              /* memsize and memlane of p */              \
    /* input */                                                             \
    bool on_stack,              /* true if werkspace is from Werk stack */  \
    uint64_t nitems,            /* # of items to allocate */                \
    uint64_t size_of_item,      /* size of each item */                     \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_WERK_PUSH_PROTO(GX_werk_push)                           \
void *GX_werk_push    /* return pointer to newly allocated space */         \
(                                                                           \
    /* output */                                                            \
    uint64_t *p_mem,            /* memsize and memlane of p */              \
    bool *on_stack,             /* true if werkspace is from Werk stack */  \
    /* input */                                                             \
    uint64_t nitems,            /* # of items to allocate */                \
    uint64_t size_of_item,      /* size of each item */                     \
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
    uint64_t *p_TaskList_mem,       /* size of TaskList and memlane */      \
    int *p_ntasks,                  /* # of tasks constructed */            \
    int *p_nthreads,                /* # of threads to use */               \
    /* input: */                                                            \
    const GrB_Matrix C,             /* output matrix C */                   \
    const void *I,                                                          \
    const bool I_is_32,                                                     \
    const int64_t nI,                                                       \
    const int Ikind,                                                        \
    const int64_t Icolon [3],                                               \
    const void *J,                                                          \
    const bool J_is_32,                                                     \
    const int64_t nJ,                                                       \
    const int Jkind,                                                        \
    const int64_t Jcolon [3],                                               \
    const GrB_Matrix M,             /* matrix to slice */                   \
    const int memlane,              /* memlane to use */                    \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_ADD_PHASE0_PROTO(GX_add_phase0)                         \
GrB_Info GX_add_phase0          /* find vectors in C for C=A+B or C<M>=A+B*/\
(                                                                           \
    int64_t *p_Cnvec,           /* # of vectors to compute in C */          \
    void **Ch_handle,           /* Ch: size Cnvec, or NULL */               \
    uint64_t *Ch_mem_handle,             /* size of Ch in bytes; memlane */ \
    int64_t *restrict *C_to_M_handle,    /* C_to_M: size Cnvec, or NULL */  \
    uint64_t *C_to_M_mem_handle,         /* size of C_to_M and memlane */   \
    int64_t *restrict *C_to_A_handle,    /* C_to_A: size Cnvec, or NULL */  \
    uint64_t *C_to_A_mem_handle,         /* size of C_to_A and memlane */   \
    int64_t *restrict *C_to_B_handle,    /* C_to_B: size Cnvec, or NULL */  \
    uint64_t *C_to_B_mem_handle,         /* size of C_to_A and memlane */   \
    bool *p_Ch_is_Mh,           /* if true, then Ch == Mh */                \
    bool *p_Cp_is_32,           /* if true, Cp is 32-bit; else 64-bit */    \
    bool *p_Cj_is_32,           /* if true, Ch is 32-bit; else 64-bit */    \
    bool *p_Ci_is_32,           /* if true, Ci is 32-bit; else 64-bit */    \
    int *C_sparsity,            /* sparsity structure of C */               \
    const GrB_Matrix M,         /* optional mask, may be NULL; not compl */ \
    const GrB_Matrix A,         /* first input matrix */                    \
    const GrB_Matrix B,         /* second input matrix */                   \
    int memlane,                /* memlane to use */                        \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_EWISE_SLICE_PROTO(GX_ewise_slice)                       \
GrB_Info GX_ewise_slice                                                     \
(                                                                           \
    /* output: */                                                           \
    GB_task_struct **p_TaskList,    /* array of structs */                  \
    uint64_t *p_TaskList_mem,       /* size of TaskList and memlane */      \
    int *p_ntasks,                  /* # of tasks constructed */            \
    int *p_nthreads,                /* # of threads for eWise operation */  \
    /* input: */                                                            \
    const int64_t Cnvec,            /* # of vectors of C */                 \
    const void *Ch,                 /* vectors of C, if hypersparse */      \
    const bool Cj_is_32,            /* if true, Ch is 32-bit, else 64-bit */\
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
    uint64_t *p_TaskList_mem,       /* size of TaskList and memlane */      \
    int *p_ntasks,                  /* # of tasks constructed */            \
    int *p_nthreads,                /* # of threads to use */               \
    /* input: */                                                            \
    const int64_t nI,                                                       \
    const int64_t nJ,                                                       \
    int memlane,                /* memlane to use */                        \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_PENDING_ENSURE_PROTO(GX_Pending_ensure)                 \
bool GX_Pending_ensure                                                      \
(                                                                           \
    GrB_Matrix C,           /* matrix with C->Pending */                    \
    bool iso,               /* if true, do not allocate Pending->x */       \
    GrB_Type type,          /* type of pending tuples */                    \
    GrB_BinaryOp op,        /* operator for assembling pending tuples */    \
    int64_t nnew,           /* # of pending tuples to add */                \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_SUBASSIGN_08N_SLICE_PROTO(GX_subassign_08n_slice)       \
GrB_Info GX_subassign_08n_slice                                             \
(                                                                           \
    /* output: */                                                           \
    GB_task_struct **p_TaskList,    /* size max_ntasks */                   \
    uint64_t *p_TaskList_mem,       /* size of TaskList and memlane */      \
    int *p_ntasks,                  /* # of tasks constructed */            \
    int *p_nthreads,                /* # of threads to use */               \
    int64_t *p_Znvec,               /* # of vectors to compute in Z */      \
    const void **Zh_handle,         /* Zh is A->h, M->h, or NULL */         \
    int64_t *restrict *Z_to_A_handle, /* Z_to_A: size Znvec, or NULL */     \
    uint64_t *Z_to_A_mem_handle,                                            \
    int64_t *restrict *Z_to_M_handle, /* Z_to_M: size Znvec, or NULL */     \
    uint64_t *Z_to_M_mem_handle,                                            \
    bool *Zj_is_32_handle,                                                  \
    /* input: */                                                            \
    const GrB_Matrix C,         /* output matrix C */                       \
    const void *I,              /* I index list */                          \
    const bool I_is_32,                                                     \
    const int64_t nI,                                                       \
    const int Ikind,                                                        \
    const int64_t Icolon [3],                                               \
    const void *J,              /* J index list */                          \
    const bool J_is_32,                                                     \
    const int64_t nJ,                                                       \
    const int Jkind,                                                        \
    const int64_t Jcolon [3],                                               \
    const GrB_Matrix A,         /* matrix to slice */                       \
    const GrB_Matrix M,         /* matrix to slice */                       \
    GB_Werk Werk                                                            \
)

#define GB_CALLBACK_BITMAP_ASSIGN_TO_FULL_PROTO(GX_bitmap_assign_to_full)   \
void GX_bitmap_assign_to_full   /* set all C->b to 1, or make C full */     \
(                                                                           \
    GrB_Matrix C,                                                           \
    int nthreads_max                                                        \
)

#define GB_CALLBACK_P_SLICE_PROTO(GX_p_slice)                               \
void GX_p_slice                 /* slice Ap, 32-bit or 64-bit */            \
(                                                                           \
    /* output: */                                                           \
    int64_t *restrict Slice,    /* size ntasks+1 */                         \
    /* input: */                                                            \
    const void *Work,           /* array size n+1 (full/bitmap: NULL)*/     \
    bool Work_is_32,            /* if true, Ap is uint32_t, else uint64_t */\
    const int64_t n,                                                        \
    const int ntasks,           /* # of tasks */                            \
    const bool perfectly_balanced                                           \
)

#define GB_CALLBACK_NEW_BIX_PROTO(GX_new_bix)                               \
GrB_Info GX_new_bix     /* create a new matrix, incl. A->b, A->i, A->x */   \
(                                                                           \
    GrB_Matrix *Ahandle,        /* output matrix to create */               \
    const GrB_Type type,        /* type of output matrix */                 \
    const int64_t vlen,         /* length of each vector */                 \
    const int64_t vdim,         /* number of vectors */                     \
    const GB_ph_code Ap_option, /* allocate A->p and A->h, or leave NULL */ \
    const bool is_csc,          /* true if CSC, false if CSR */             \
    const int sparsity,         /* hyper, sparse, bitmap, full, or auto */  \
    const bool bitmap_calloc,   /* if true, calloc A->b, else use malloc */ \
    const float hyper_switch,   /* A->hyper_switch, unless auto */          \
    const int64_t plen,         /* size of A->p and A->h, if hypersparse */ \
    const int64_t nzmax,        /* # of nonzeros the matrix must hold; */   \
                                /* ignored if A is iso and full */          \
    const bool numeric,         /* if true, allocate A->x, else it's NULL */\
    const bool A_iso,           /* if true, allocate A as iso */            \
    bool p_is_32,               /* if true, A->p is 32 bit; else 64 */      \
    bool j_is_32,               /* if true, A->h, A->Y are 32 bit else 64 */\
    bool i_is_32,               /* if true, A->i is 32 bit; else 64 */      \
    int memlane                 /* memlane for the matrix */                \
)

#define GB_CALLBACK_MATRIX_FREE_PROTO(GX_Matrix_free)                       \
void GX_Matrix_free             /* free a matrix */                         \
(                                                                           \
    GrB_Matrix *Ahandle         /* handle of matrix to free */              \
)

#endif

