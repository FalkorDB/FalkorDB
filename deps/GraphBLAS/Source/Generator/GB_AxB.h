
// SPDX-License-Identifier: Apache-2.0
GrB_Info GB (_Adot2B)
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const bool A_not_transposed,
    const GrB_Matrix A, int64_t *restrict A_slice,
    const GrB_Matrix B, int64_t *restrict B_slice,
    int nthreads, int naslice, int nbslice
) ;

GrB_Info GB (_Adot3B)
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GB_task_struct *restrict TaskList,
    const int ntasks,
    const int nthreads
) ;

m4_divert(if_dot4_enabled)
GrB_Info GB (_Adot4B)
(
    GrB_Matrix C,
    const bool C_in_iso,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict A_slice,
    const int64_t *restrict B_slice,
    const int naslice,
    const int nbslice,
    const int nthreads,
    GB_Werk Werk
) ;
m4_divert(0)

GrB_Info GB (_Asaxpy3B)
(
    GrB_Matrix C,   // C<any M>=A*B, C sparse or hypersparse
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_saxpy3task_struct *restrict SaxpyTasks,
    const int ntasks, const int nfine, const int nthreads, const int do_sort,
    GB_Werk Werk
) ;

GrB_Info GB (_Asaxpy3B_noM)
(
    GrB_Matrix C,   // C=A*B, C sparse or hypersparse
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_saxpy3task_struct *restrict SaxpyTasks,
    const int ntasks, const int nfine, const int nthreads,
    const int do_sort,
    GB_Werk Werk
) ;

GrB_Info GB (_Asaxpy3B_M)
(
    GrB_Matrix C,   // C<M>=A*B, C sparse or hypersparse
    const GrB_Matrix M, const bool Mask_struct, const bool M_in_place,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_saxpy3task_struct *restrict SaxpyTasks,
    const int ntasks, const int nfine, const int nthreads,
    const int do_sort,
    GB_Werk Werk
) ;

GrB_Info GB (_Asaxpy3B_notM)
(
    GrB_Matrix C,   // C<!M>=A*B, C sparse or hypersparse
    const GrB_Matrix M, const bool Mask_struct, const bool M_in_place,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_saxpy3task_struct *restrict SaxpyTasks,
    const int ntasks, const int nfine, const int nthreads,
    const int do_sort,
    GB_Werk Werk
) ;

GrB_Info GB (_AsaxbitB)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int ntasks,
    const int nthreads,
    const int nfine_tasks_per_vector,
    const bool use_coarse_tasks,
    const bool use_atomics,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_slice,
    const int64_t *restrict H_slice,
    GB_void *restrict Wcx,
    int8_t *restrict Wf
) ;

m4_divert(if_saxpy4_enabled)
GrB_Info GB (_Asaxpy4B)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int ntasks,
    const int nthreads,
    const int nfine_tasks_per_vector,
    const bool use_coarse_tasks,
    const bool use_atomics,
    const int64_t *A_slice,
    const int64_t *H_slice,
    GB_void *restrict Wcx,
    int8_t *restrict Wf
) ;
m4_divert(0)

m4_divert(if_saxpy5_enabled)
GrB_Info GB (_Asaxpy5B)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int ntasks,
    const int nthreads,
    const int64_t *B_slice
) ;
m4_divert(0)

