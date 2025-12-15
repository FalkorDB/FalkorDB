//------------------------------------------------------------------------------
// GB_jit_kernel_proto.h: prototypes for all JIT kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_JIT_KERNEL_PROTO_H
#define GB_JIT_KERNEL_PROTO_H

//------------------------------------------------------------------------------
// JIT kernel prototypes
//------------------------------------------------------------------------------

#define GB_JIT_QUERY_PROTO(query_func)                                  \
bool query_func (uint64_t *hash, int v [3], const char *defn [5],       \
    void *id, void *term, size_t id_size, size_t term_size)

#define GB_JIT_KERNEL_USER_OP_PROTO(GB_jit_kernel_user_op)              \
GrB_Info GB_jit_kernel_user_op (void **user_function, char **defn)

#define GB_JIT_KERNEL_USER_TYPE_PROTO(GB_jit_kernel_user_type)          \
GrB_Info GB_jit_kernel_user_type (size_t *user_type_size, char **defn)

#define GB_JIT_KERNEL_ADD_PROTO(GB_jit_kernel_add)                      \
GrB_Info GB_jit_kernel_add                                              \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const bool Ch_is_Mh,                                                \
    const int64_t *restrict C_to_M,                                     \
    const int64_t *restrict C_to_A,                                     \
    const int64_t *restrict C_to_B,                                     \
    const GB_task_struct *restrict TaskList,                            \
    const int C_ntasks,                                                 \
    const int C_nthreads,                                               \
    const int64_t *M_ek_slicing,                                        \
    const int M_nthreads,                                               \
    const int M_ntasks,                                                 \
    const int64_t *A_ek_slicing,                                        \
    const int A_nthreads,                                               \
    const int A_ntasks,                                                 \
    const int64_t *B_ek_slicing,                                        \
    const int B_nthreads,                                               \
    const int B_ntasks,                                                 \
    const bool M_is_A,                                                  \
    const bool M_is_B,                                                  \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_APPLY_BIND1ST_PROTO(GB_jit_kernel_apply_bind1st)  \
GrB_Info GB_jit_kernel_apply_bind1st                                    \
(                                                                       \
    GB_void *Cx_output,         /* Cx and Bx may be aliased */          \
    const GB_void *x_input,                                             \
    const GB_void *Bx_input,                                            \
    const int8_t *restrict Bb,                                          \
    const int64_t bnz,                                                  \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_APPLY_BIND2ND_PROTO(GB_jit_kernel_apply_bind2nd)  \
GrB_Info GB_jit_kernel_apply_bind2nd                                    \
(                                                                       \
    GB_void *Cx_output,         /* Cx and Ax may be aliased */          \
    const GB_void *Ax_input,                                            \
    const GB_void *y_input,                                             \
    const int8_t *restrict Ab,                                          \
    const int64_t anz,                                                  \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_APPLY_UNOP_PROTO(GB_jit_kernel_apply_unop)        \
GrB_Info GB_jit_kernel_apply_unop                                       \
(                                                                       \
    GB_void *Cx_out,            /* Cx and Ax may be aliased */          \
    const GrB_Matrix A,                                                 \
    const void *ythunk,         /* for idx unops (op->ytype scalar) */  \
    const int64_t *A_ek_slicing,                                        \
    const int A_ntasks,                                                 \
    const int A_nthreads,                                               \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_AXB_DOT2_PROTO(GB_jit_kernel_AxB_dot2)            \
GrB_Info GB_jit_kernel_AxB_dot2                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const GrB_Matrix A,                                                 \
    const int64_t *restrict A_slice,                                    \
    const GrB_Matrix B,                                                 \
    const int64_t *restrict B_slice,                                    \
    const int nthreads,                                                 \
    const int naslice,                                                  \
    const int nbslice,                                                  \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_AXB_DOT2N_PROTO(GB_jit_kernel_AxB_dot2n)          \
GrB_Info GB_jit_kernel_AxB_dot2n                                        \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const GrB_Matrix A,                                                 \
    const int64_t *restrict A_slice,                                    \
    const GrB_Matrix B,                                                 \
    const int64_t *restrict B_slice,                                    \
    const int nthreads,                                                 \
    const int naslice,                                                  \
    const int nbslice,                                                  \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_AXB_DOT3_PROTO(GB_jit_kernel_AxB_dot3)            \
GrB_Info GB_jit_kernel_AxB_dot3                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const GB_task_struct *restrict TaskList,                            \
    const int ntasks,                                                   \
    const int nthreads,                                                 \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_AXB_DOT4_PROTO(GB_jit_kernel_AxB_dot4)            \
GrB_Info GB_jit_kernel_AxB_dot4                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const int64_t *restrict A_slice,                                    \
    const int64_t *restrict B_slice,                                    \
    const int naslice,                                                  \
    const int nbslice,                                                  \
    const int nthreads,                                                 \
    GB_Werk Werk,                                                       \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_AXB_SAXBIT_PROTO(GB_jit_kernel_AxB_saxbit)        \
GrB_Info GB_jit_kernel_AxB_saxbit                                       \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const int ntasks,                                                   \
    const int nthreads,                                                 \
    const int nfine_tasks_per_vector,                                   \
    const bool use_coarse_tasks,                                        \
    const bool use_atomics,                                             \
    const int64_t *M_ek_slicing,                                        \
    const int M_nthreads,                                               \
    const int M_ntasks,                                                 \
    const int64_t *restrict A_slice,                                    \
    const int64_t *restrict H_slice,                                    \
    GB_void *restrict Wcx,                                              \
    int8_t *restrict Wf,                                                \
    const int nthreads_max,                                             \
    double chunk,                                                       \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_AXB_SAXPY3_PROTO(GB_jit_kernel_AxB_saxpy3)        \
GrB_Info GB_jit_kernel_AxB_saxpy3                                       \
(                                                                       \
    GrB_Matrix C,   /* C<any M>=A*B, C sparse or hypersparse */         \
    const GrB_Matrix M,                                                 \
    const bool M_in_place,                                              \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    GB_saxpy3task_struct *restrict SaxpyTasks,                          \
    const int ntasks,                                                   \
    const int nfine,                                                    \
    const int nthreads,                                                 \
    const int do_sort,                                                  \
    const int nthreads_max,                                             \
    double chunk,                                                       \
    GB_Werk Werk,                                                       \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_AXB_SAXPY4_PROTO(GB_jit_kernel_AxB_saxpy4)        \
GrB_Info GB_jit_kernel_AxB_saxpy4                                       \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const int ntasks,                                                   \
    const int nthreads,                                                 \
    const int nfine_tasks_per_vector,                                   \
    const bool use_coarse_tasks,                                        \
    const bool use_atomics,                                             \
    const int64_t *restrict A_slice,                                    \
    const int64_t *restrict H_slice,                                    \
    GB_void *restrict Wcx,                                              \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_AXB_SAXPY5_PROTO(GB_jit_kernel_AxB_saxpy5)        \
GrB_Info GB_jit_kernel_AxB_saxpy5                                       \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const int ntasks,                                                   \
    const int nthreads,                                                 \
    const int64_t *restrict B_slice,                                    \
    bool cpu_has_avx2,                                                  \
    bool cpu_has_avx512f,                                               \
    bool cpu_has_rvv1,                                                  \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_BUILD_PROTO(GB_jit_kernel_build)                  \
GrB_Info GB_jit_kernel_build                                            \
(                                                                       \
    GB_void *restrict Tx_void,                                          \
    void *restrict Ti_void,                                             \
    const GB_void *restrict Sx_void,                                    \
    const int64_t nvals,                                                \
    const void *restrict I_work_void,                                   \
    const void *restrict K_work_void,                                   \
    const int64_t duplicate_entry_input,                                \
    const int64_t *restrict tstart_slice,                               \
    const int64_t *restrict tnz_slice,                                  \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_COLSCALE_PROTO(GB_jit_kernel_colscale)            \
GrB_Info GB_jit_kernel_colscale                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix D,                                                 \
    const int64_t *A_ek_slicing,                                        \
    const int A_ntasks,                                                 \
    const int A_nthreads,                                               \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_CONCAT_BITMAP_PROTO(GB_jit_kernel_concat_bitmap)  \
GrB_Info GB_jit_kernel_concat_bitmap                                    \
(                                                                       \
    GrB_Matrix C,                                                       \
    const int64_t cistart,                                              \
    const int64_t cvstart,                                              \
    const GrB_Matrix A,                                                 \
    const int nthreads_max,                                             \
    double chunk,                                                       \
    GB_Werk Werk,                                                       \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_CONCAT_FULL_PROTO(GB_jit_kernel_concat_full)      \
GrB_Info GB_jit_kernel_concat_full                                      \
(                                                                       \
    GrB_Matrix C,                                                       \
    const int64_t cistart,                                              \
    const int64_t cvstart,                                              \
    const GrB_Matrix A,                                                 \
    const int A_nthreads,                                               \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_CONCAT_SPARSE_PROTO(GB_jit_kernel_concat_sparse)  \
GrB_Info GB_jit_kernel_concat_sparse                                    \
(                                                                       \
    GrB_Matrix C,                                                       \
    const int64_t cistart,                                              \
    const GrB_Matrix A,                                                 \
    const void *W_parameter,                                            \
    const int64_t *A_ek_slicing,                                        \
    const int A_ntasks,                                                 \
    const int A_nthreads,                                               \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_CONVERT_S2B_PROTO(GB_jit_kernel_convert_s2b)      \
GrB_Info GB_jit_kernel_convert_s2b                                      \
(                                                                       \
    GB_void *restrict Cx_new,                                           \
    int8_t *restrict Cb,                                                \
    const GrB_Matrix A,                                                 \
    const int64_t *A_ek_slicing,                                        \
    const int A_ntasks,                                                 \
    const int A_nthreads,                                               \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_EMULT_02_PROTO(GB_jit_kernel_emult_02)            \
GrB_Info GB_jit_kernel_emult_02                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const bool Mask_struct,                                             \
    const bool Mask_comp,                                               \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const uint64_t *restrict Cp_kfirst,                                 \
    const int64_t *A_ek_slicing,                                        \
    const int A_ntasks,                                                 \
    const int A_nthreads,                                               \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_EMULT_03_PROTO(GB_jit_kernel_emult_03)            \
GrB_Info GB_jit_kernel_emult_03                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const bool Mask_struct,                                             \
    const bool Mask_comp,                                               \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const uint64_t *restrict Cp_kfirst,                                 \
    const int64_t *B_ek_slicing,                                        \
    const int B_ntasks,                                                 \
    const int B_nthreads,                                               \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_EMULT_04_PROTO(GB_jit_kernel_emult_04)            \
GrB_Info GB_jit_kernel_emult_04                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const bool Mask_struct,                                             \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const uint64_t *restrict Cp_kfirst,                                 \
    const int64_t *M_ek_slicing,                                        \
    const int M_ntasks,                                                 \
    const int M_nthreads,                                               \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_EMULT_08_PROTO(GB_jit_kernel_emult_08)            \
GrB_Info GB_jit_kernel_emult_08                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const bool Mask_struct,                                             \
    const bool Mask_comp,                                               \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const int64_t *restrict C_to_M,                                     \
    const int64_t *restrict C_to_A,                                     \
    const int64_t *restrict C_to_B,                                     \
    const GB_task_struct *restrict TaskList,                            \
    const int C_ntasks,                                                 \
    const int C_nthreads,                                               \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_EMULT_BITMAP_PROTO(GB_jit_kernel_emult_bitmap)    \
GrB_Info GB_jit_kernel_emult_bitmap                                     \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const bool Mask_struct,                                             \
    const bool Mask_comp,                                               \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const int64_t *M_ek_slicing,                                        \
    const int M_ntasks,                                                 \
    const int M_nthreads,                                               \
    const int C_nthreads,                                               \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_EWISE_FULLA_PROTO(GB_jit_kernel_ewise_fulla)      \
GrB_Info GB_jit_kernel_ewise_fulla                                      \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const int nthreads,                                                 \
    const bool A_is_B,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_EWISE_FULLN_PROTO(GB_jit_kernel_ewise_fulln)      \
GrB_Info GB_jit_kernel_ewise_fulln                                      \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_REDUCE_PROTO(GB_jit_kernel_reduce)                \
GrB_Info GB_jit_kernel_reduce                                           \
(                                                                       \
    GB_void *result,                                                    \
    const GrB_Matrix A,                                                 \
    GB_void *restrict Workspace,                                        \
    bool *restrict F,                                                   \
    const int ntasks,                                                   \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_ROWSCALE_PROTO(GB_jit_kernel_rowscale)            \
GrB_Info GB_jit_kernel_rowscale                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix D,                                                 \
    const GrB_Matrix B,                                                 \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_SELECT_BITMAP_PROTO(GB_jit_kernel_select_bitmap)  \
GrB_Info GB_jit_kernel_select_bitmap                                    \
(                                                                       \
    GrB_Matrix C,                                                       \
    GrB_Matrix A,                                                       \
    const GB_void *restrict ythunk,                                     \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_SELECT_PHASE1_PROTO(GB_jit_kernel_select_phase1)  \
GrB_Info GB_jit_kernel_select_phase1                                    \
(                                                                       \
    GrB_Matrix C,                                                       \
    uint64_t *restrict Wfirst,                                          \
    uint64_t *restrict Wlast,                                           \
    const GrB_Matrix A,                                                 \
    const GB_void *restrict ythunk,                                     \
    const int64_t *A_ek_slicing,                                        \
    const int A_ntasks,                                                 \
    const int A_nthreads,                                               \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_SELECT_PHASE2_PROTO(GB_jit_kernel_select_phase2)  \
GrB_Info GB_jit_kernel_select_phase2                                    \
(                                                                       \
    GrB_Matrix C,                                                       \
    const uint64_t *restrict Cp_kfirst,                                 \
    const GrB_Matrix A,                                                 \
    const GB_void *restrict ythunk,                                     \
    const int64_t *A_ek_slicing,                                        \
    const int A_ntasks,                                                 \
    const int A_nthreads,                                               \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_SPLIT_BITMAP_PROTO(GB_jit_kernel_split_bitmap)    \
GrB_Info GB_jit_kernel_split_bitmap                                     \
(                                                                       \
    GrB_Matrix C,                                                       \
    GrB_Matrix A,                                                       \
    const int64_t avstart,                                              \
    const int64_t aistart,                                              \
    const int C_nthreads,                                               \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_SPLIT_FULL_PROTO(GB_jit_kernel_split_full)        \
GrB_Info GB_jit_kernel_split_full                                       \
(                                                                       \
    GrB_Matrix C,                                                       \
    GrB_Matrix A,                                                       \
    const int64_t avstart,                                              \
    const int64_t aistart,                                              \
    const int C_nthreads,                                               \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_SPLIT_SPARSE_PROTO(GB_jit_kernel_split_sparse)    \
GrB_Info GB_jit_kernel_split_sparse                                     \
(                                                                       \
    GrB_Matrix C,                                                       \
    GrB_Matrix A,                                                       \
    const int64_t akstart,                                              \
    const int64_t aistart,                                              \
    const void *Wp_workspace,                                           \
    const int64_t *C_ek_slicing,                                        \
    const int C_ntasks,                                                 \
    const int C_nthreads,                                               \
    const GB_callback_struct *restrict my_callback                      \
)

// used for all subassign kernels:
#define GB_JIT_KERNEL_SUBASSIGN_PROTO(GB_jit_kernel_subassign_any)      \
GrB_Info GB_jit_kernel_subassign_any                                    \
(                                                                       \
    GrB_Matrix C,                                                       \
    const bool C_replace,                                               \
    const void *I_input,                                                \
    const int64_t ni,                                                   \
    const int64_t nI,                                                   \
    const int Ikind,                                                    \
    const int64_t Icolon [3],                                           \
    const void *J_input,                                                \
    const int64_t nj,                                                   \
    const int64_t nJ,                                                   \
    const int Jkind,                                                    \
    const int64_t Jcolon [3],                                           \
    const GrB_Matrix M,                                                 \
    const bool Mask_comp,                                               \
    const bool Mask_struct,                                             \
    const GrB_BinaryOp accum,                                           \
    const GrB_Matrix A,                                                 \
    const void *scalar,                                                 \
    const GrB_Type scalar_type,                                         \
    const GrB_Matrix S,                                                 \
    const int assign_kind,                                              \
    GB_Werk Werk,                                                       \
    const int nthreads_max,                                             \
    double chunk,                                                       \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_TRANS_BIND1ST_PROTO(GB_jit_kernel_trans_bind1st)  \
GrB_Info GB_jit_kernel_trans_bind1st                                    \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GB_void *x_input,                                             \
    const GrB_Matrix A,                                                 \
    void **Workspaces,                                                  \
    const int64_t *restrict A_slice,                                    \
    const int nworkspaces,                                              \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_TRANS_BIND2ND_PROTO(GB_jit_kernel_trans_bind2nd)  \
GrB_Info GB_jit_kernel_trans_bind2nd                                    \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix A,                                                 \
    const GB_void *y_input,                                             \
    void **Workspaces,                                                  \
    const int64_t *restrict A_slice,                                    \
    const int nworkspaces,                                              \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_TRANS_UNOP_PROTO(GB_jit_kernel_trans_unop)        \
GrB_Info GB_jit_kernel_trans_unop                                       \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix A,                                                 \
    void **Workspaces,                                                  \
    const int64_t *restrict A_slice,                                    \
    const int nworkspaces,                                              \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_UNION_PROTO(GB_jit_kernel_union)                  \
GrB_Info GB_jit_kernel_union                                            \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const GB_void *alpha_scalar_in,                                     \
    const GB_void *beta_scalar_in,                                      \
    const bool Ch_is_Mh,                                                \
    const int64_t *restrict C_to_M,                                     \
    const int64_t *restrict C_to_A,                                     \
    const int64_t *restrict C_to_B,                                     \
    const GB_task_struct *restrict TaskList,                            \
    const int C_ntasks,                                                 \
    const int C_nthreads,                                               \
    const int64_t *M_ek_slicing,                                        \
    const int M_nthreads,                                               \
    const int M_ntasks,                                                 \
    const int64_t *A_ek_slicing,                                        \
    const int A_nthreads,                                               \
    const int A_ntasks,                                                 \
    const int64_t *B_ek_slicing,                                        \
    const int B_nthreads,                                               \
    const int B_ntasks,                                                 \
    const bool M_is_A,                                                  \
    const bool M_is_B,                                                  \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_MASKER_PHASE1_PROTO(GB_jit_kernel_masker1)        \
GrB_Info GB_jit_kernel_masker1                                          \
(                                                                       \
    void *Rp_parameter,                                                 \
    int64_t *Rnvec_nonempty,                                            \
    GB_task_struct *restrict TaskList,                                  \
    const int R_ntasks,                                                 \
    const int R_nthreads,                                               \
    const int64_t Rnvec,                                                \
    const void *Rh_parameter,                                           \
    const int64_t *restrict R_to_M,                                     \
    const int64_t *restrict R_to_C,                                     \
    const int64_t *restrict R_to_Z,                                     \
    const GrB_Matrix M,                                                 \
    const bool Mask_comp,                                               \
    const bool Mask_struct,                                             \
    const GrB_Matrix C,                                                 \
    const GrB_Matrix Z,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_MASKER_PHASE2_PROTO(GB_jit_kernel_masker2)        \
GrB_Info GB_jit_kernel_masker2                                          \
(                                                                       \
    GrB_Matrix R,                                                       \
    const GB_task_struct *restrict TaskList,                            \
    const int R_ntasks,                                                 \
    const int R_nthreads,                                               \
    const int64_t *restrict R_to_M,                                     \
    const int64_t *restrict R_to_C,                                     \
    const int64_t *restrict R_to_Z,                                     \
    const GrB_Matrix M,                                                 \
    const bool Mask_comp,                                               \
    const bool Mask_struct,                                             \
    const GrB_Matrix C,                                                 \
    const GrB_Matrix Z,                                                 \
    const int64_t *restrict C_ek_slicing,                               \
    const int C_nthreads,                                               \
    const int C_ntasks,                                                 \
    const int64_t *restrict M_ek_slicing,                               \
    const int M_nthreads,                                               \
    const int M_ntasks,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_SUBREF_SPARSE_PROTO(GB_jit_kernel_subref_sparse)  \
GrB_Info GB_jit_kernel_subref_sparse                                    \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GB_task_struct *restrict TaskList,                            \
    const int ntasks,                                                   \
    const int nthreads,                                                 \
    const bool post_sort,                                               \
    const GrB_Matrix R,                                                 \
    const void *Ap_start_input,                                         \
    const void *Ap_end_input,                                           \
    const int64_t nI,                                                   \
    const int64_t Icolon [3],                                           \
    const GrB_Matrix A,                                                 \
    const void *I_input,                                                \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_BITMAP_SUBREF_PROTO(GB_jit_kernel_bitmap_subref)  \
GrB_Info GB_jit_kernel_bitmap_subref                                    \
(                                                                       \
    GrB_Matrix C,                                                       \
    GrB_Matrix A,                                                       \
    const void *I_input,                                                \
    const int64_t nI,                                                   \
    const int64_t Icolon [3],                                           \
    const void *J_input,                                                \
    const int64_t nJ,                                                   \
    const int64_t Jcolon [3],                                           \
    GB_Werk Werk,                                                       \
    const int nthreads_max,                                             \
    double chunk,                                                       \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_ISO_EXPAND_PROTO(GB_jit_kernel_iso_expand)        \
GrB_Info GB_jit_kernel_iso_expand                                       \
(                                                                       \
    void *restrict X,                                                   \
    const int64_t n,                                                    \
    const void *restrict scalar,                                        \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_UNJUMBLE_PROTO(GB_jit_kernel_unjumble)            \
GrB_Info GB_jit_kernel_unjumble                                         \
(                                                                       \
    const GrB_Matrix A,                                                 \
    const int64_t *A_slice,                                             \
    const int ntasks,                                                   \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_CONVERT_B2S_PROTO(GB_jit_kernel_convert_b2s)      \
GrB_Info GB_jit_kernel_convert_b2s                                      \
(                                                                       \
    const void *Cp_input,                                               \
    void *Ci_input,                                                     \
    void *Cj_input,                                                     \
    void *Cx_new,                                                       \
    const GrB_Matrix A,                                                 \
    const void *W_input,                                                \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_KRONER_PROTO(GB_jit_kernel_kroner)                \
GrB_Info GB_jit_kernel_kroner                                           \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    const int nthreads,                                                 \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_KERNEL_SORT_PROTO(GB_jit_kernel_sort)                    \
GrB_Info GB_jit_kernel_sort                                             \
(                                                                       \
    GrB_Matrix C,                                                       \
    int nthreads,                                                       \
    GB_Werk Werk,                                                       \
    const GB_callback_struct *restrict my_callback                      \
)

//------------------------------------------------------------------------------
// CUDA JIT prototypes
//------------------------------------------------------------------------------

#define GB_JIT_CUDA_KERNEL_REDUCE_PROTO(GB_jit_kernel_reduce)           \
GrB_Info GB_jit_kernel_reduce                                           \
(                                                                       \
    GB_void *zscalar,                                                   \
    GrB_Matrix V,                                                       \
    const GrB_Matrix A,                                                 \
    cudaStream_t stream,                                                \
    int32_t gridsz,                                                     \
    int32_t blocksz,                                                    \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_CUDA_KERNEL_ROWSCALE_PROTO(GB_jit_kernel_rowscale)       \
GrB_Info GB_jit_kernel_rowscale                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    GrB_Matrix D,                                                       \
    GrB_Matrix B,                                                       \
    cudaStream_t stream,                                                \
    int32_t gridsz,                                                     \
    int32_t blocksz,                                                    \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_CUDA_KERNEL_COLSCALE_PROTO(GB_jit_kernel_colscale)       \
GrB_Info GB_jit_kernel_colscale                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    GrB_Matrix A,                                                       \
    GrB_Matrix D,                                                       \
    cudaStream_t stream,                                                \
    int32_t gridsz,                                                     \
    int32_t blocksz,                                                    \
    const GB_callback_struct *restrict my_callback                      \
)

#define GB_JIT_CUDA_KERNEL_APPLY_BIND1ST_PROTO(GB_jit_kernel_apply_bind1st) \
GrB_Info GB_jit_kernel_apply_bind1st                                        \
(                                                                           \
    GB_void *Cx,                                                            \
    const GB_void *scalarx,                                                 \
    GrB_Matrix B,                                                           \
    cudaStream_t stream,                                                    \
    int32_t gridsz,                                                         \
    int32_t blocksz,                                                        \
    const GB_callback_struct *restrict my_callback                          \
)                                                                           \

#define GB_JIT_CUDA_KERNEL_APPLY_BIND2ND_PROTO(GB_jit_kernel_apply_bind2nd) \
GrB_Info GB_jit_kernel_apply_bind2nd                                        \
(                                                                           \
    GB_void *Cx,                                                            \
    GrB_Matrix A,                                                           \
    const GB_void *scalarx,                                                 \
    cudaStream_t stream,                                                    \
    int32_t gridsz,                                                         \
    int32_t blocksz,                                                        \
    const GB_callback_struct *restrict my_callback                          \
)                                                                           \

#define GB_JIT_CUDA_KERNEL_APPLY_UNOP_PROTO(GB_jit_kernel_apply_unop)       \
GrB_Info GB_jit_kernel_apply_unop                                           \
(                                                                           \
    GB_void *Cx,                                                            \
    GrB_Matrix A,                                                           \
    const GB_void *ythunk,                                                  \
    cudaStream_t stream,                                                    \
    int32_t gridsz,                                                         \
    int32_t blocksz,                                                        \
    const GB_callback_struct *restrict my_callback                          \
)                                                                           \

#define GB_JIT_CUDA_KERNEL_SELECT_BITMAP_PROTO(GB_jit_kernel_select_bitmap) \
GrB_Info GB_jit_kernel_select_bitmap                                        \
(                                                                           \
    GrB_Matrix C,                                                           \
    GrB_Matrix A,                                                           \
    const GB_void *ythunk,                                                  \
    cudaStream_t stream,                                                    \
    int32_t gridsz,                                                         \
    const GB_callback_struct *restrict my_callback                          \
)                                                                           \

#define GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO(GB_jit_kernel_select_sparse) \
GrB_Info GB_jit_kernel_select_sparse                                        \
(                                                                           \
    GrB_Matrix C,                                                           \
    GrB_Matrix A,                                                           \
    const GB_void *ythunk,                                                  \
    cudaStream_t stream,                                                    \
    int32_t gridsz,                                                         \
    const GB_callback_struct *restrict my_callback                          \
)

#define GB_JIT_CUDA_KERNEL_DOT3_PROTO(GB_jit_kernel_AxB_dot3)           \
GrB_Info GB_jit_kernel_AxB_dot3                                         \
(                                                                       \
    GrB_Matrix C,                                                       \
    const GrB_Matrix M,                                                 \
    const GrB_Matrix A,                                                 \
    const GrB_Matrix B,                                                 \
    cudaStream_t stream,                                                \
    int device,                                                         \
    int number_of_sms,                                                  \
    const void *theta,                                                  \
    const GB_callback_struct *restrict my_callback                      \
)

//------------------------------------------------------------------------------
// shorthand macros for GB_prejit.c:
//------------------------------------------------------------------------------

#define JIT_DOT2(g) GB_JIT_KERNEL_AXB_DOT2_PROTO(g) ;
#define JIT_DO2N(g) GB_JIT_KERNEL_AXB_DOT2N_PROTO(g) ;
#define JIT_DOT3(g) GB_JIT_KERNEL_AXB_DOT3_PROTO(g) ;
#define JIT_DOT4(g) GB_JIT_KERNEL_AXB_DOT4_PROTO(g) ;
#define JIT_SAXB(g) GB_JIT_KERNEL_AXB_SAXBIT_PROTO(g) ;
#define JIT_SAX3(g) GB_JIT_KERNEL_AXB_SAXPY3_PROTO(g) ;
#define JIT_SAX4(g) GB_JIT_KERNEL_AXB_SAXPY4_PROTO(g) ;
#define JIT_SAX5(g) GB_JIT_KERNEL_AXB_SAXPY5_PROTO(g) ;
#define JIT_ADD(g)  GB_JIT_KERNEL_ADD_PROTO(g) ;
#define JIT_AP1(g)  GB_JIT_KERNEL_APPLY_BIND1ST_PROTO(g) ;
#define JIT_AP2(g)  GB_JIT_KERNEL_APPLY_BIND2ND_PROTO(g) ;
#define JIT_AP0(g)  GB_JIT_KERNEL_APPLY_UNOP_PROTO(g) ;
#define JIT_BLD(g)  GB_JIT_KERNEL_BUILD_PROTO(g) ;
#define JIT_COLS(g) GB_JIT_KERNEL_COLSCALE_PROTO(g) ;
#define JIT_CONB(g) GB_JIT_KERNEL_CONCAT_BITMAP_PROTO(g) ;
#define JIT_CONF(g) GB_JIT_KERNEL_CONCAT_FULL_PROTO(g) ;
#define JIT_CONS(g) GB_JIT_KERNEL_CONCAT_SPARSE_PROTO(g) ;
#define JIT_CS2B(g) GB_JIT_KERNEL_CONVERT_S2B_PROTO(g) ;
#define JIT_CB2S(g) GB_JIT_KERNEL_CONVERT_B2S_PROTO(g) ;
#define JIT_EM2(g)  GB_JIT_KERNEL_EMULT_02_PROTO(g) ;
#define JIT_EM3(g)  GB_JIT_KERNEL_EMULT_03_PROTO(g) ;
#define JIT_EM4(g)  GB_JIT_KERNEL_EMULT_04_PROTO(g) ;
#define JIT_EM8(g)  GB_JIT_KERNEL_EMULT_08_PROTO(g) ;
#define JIT_EMB(g)  GB_JIT_KERNEL_EMULT_BITMAP_PROTO(g) ;
#define JIT_EWFA(g) GB_JIT_KERNEL_EWISE_FULLA_PROTO(g) ;
#define JIT_EWFN(g) GB_JIT_KERNEL_EWISE_FULLN_PROTO(g) ;
#define JIT_RED(g)  GB_JIT_KERNEL_REDUCE_PROTO(g) ;
#define JIT_ROWS(g) GB_JIT_KERNEL_ROWSCALE_PROTO(g) ;
#define JIT_SELB(g) GB_JIT_KERNEL_SELECT_BITMAP_PROTO(g) ;
#define JIT_SEL1(g) GB_JIT_KERNEL_SELECT_PHASE1_PROTO(g) ;
#define JIT_SEL2(g) GB_JIT_KERNEL_SELECT_PHASE2_PROTO(g) ;
#define JIT_SPB(g)  GB_JIT_KERNEL_SPLIT_BITMAP_PROTO(g) ;
#define JIT_SPF(g)  GB_JIT_KERNEL_SPLIT_FULL_PROTO(g) ;
#define JIT_SPS(g)  GB_JIT_KERNEL_SPLIT_SPARSE_PROTO(g) ;
#define JIT_SUB(g)  GB_JIT_KERNEL_SUBASSIGN_PROTO(g) ;
#define JIT_TR1(g)  GB_JIT_KERNEL_TRANS_BIND1ST_PROTO(g) ;
#define JIT_TR2(g)  GB_JIT_KERNEL_TRANS_BIND2ND_PROTO(g) ;
#define JIT_TR0(g)  GB_JIT_KERNEL_TRANS_UNOP_PROTO(g) ;
#define JIT_UNI(g)  GB_JIT_KERNEL_UNION_PROTO(g) ;
#define JIT_UOP(g)  GB_JIT_KERNEL_USER_OP_PROTO(g) ;
#define JIT_UTYP(g) GB_JIT_KERNEL_USER_TYPE_PROTO(g) ;
#define JIT_MAS1(g) GB_JIT_KERNEL_MASKER_PHASE1_PROTO(g) ;
#define JIT_MAS2(g) GB_JIT_KERNEL_MASKER_PHASE2_PROTO(g) ;
#define JIT_SREF(g) GB_JIT_KERNEL_SUBREF_SPARSE_PROTO(g) ;
#define JIT_BREF(g) GB_JIT_KERNEL_BITMAP_SUBREF_PROTO(g) ;
#define JIT_ISOE(g) GB_JIT_KERNEL_ISO_EXPAND_PROTO(g) ;
#define JIT_UNJU(g) GB_JIT_KERNEL_UNJUMBLE_PROTO(g) ;
#define JIT_SORT(g) GB_JIT_KERNEL_SORT_PROTO(g) ;
#define JIT_KRON(g) GB_JIT_KERNEL_KRONER_PROTO(g) ;
#define JIT_Q(q)    GB_JIT_QUERY_PROTO(q) ;

//------------------------------------------------------------------------------
// shorthand macros for GB_cuda_prejit.c:
//------------------------------------------------------------------------------

#define JIT_CUDA_RED(g)  GB_JIT_CUDA_KERNEL_REDUCE_PROTO(g) ;
#define JIT_CUDA_DOT3(g) GB_JIT_CUDA_KERNEL_DOT3_PROTO(g) ;

//------------------------------------------------------------------------------
// GB_GET_CALLBACK: get function pointer from the callback struct for JIT kernel
//------------------------------------------------------------------------------

#undef GB_GET_CALLBACK
#ifdef GB_JIT_RUNTIME
    // JIT kernels (CPU and CUDA) require the function pointers
    #define GB_GET_CALLBACKS                    \
        GB_abort = my_callback->GB_abort_func
    #define GB_GET_CALLBACK(function) \
        function ## _f function = my_callback->function ## _func
#else
    // PreJIT kernels link against -lgraphblas and do not need function pointers
    #define GB_GET_CALLBACKS
    #define GB_GET_CALLBACK(function)
#endif

#endif

