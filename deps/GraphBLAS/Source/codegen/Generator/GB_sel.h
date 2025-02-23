
// SPDX-License-Identifier: Apache-2.0
m4_divert(if_phase1)
GrB_Info GB (_sel_phase1)
(
    GrB_Matrix C,
    uint64_t *restrict Wfirst,
    uint64_t *restrict Wlast,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

m4_divert(if_phase2)
GrB_Info GB (_sel_phase2)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

m4_divert(if_bitmap)
GrB_Info GB (_sel_bitmap)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;

