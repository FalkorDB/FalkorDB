
// SPDX-License-Identifier: Apache-2.0
m4_divert(if_uop_apply_enabled)
GrB_Info GB (_uop_apply)
(
    GB_void *Cx,
    const GB_void *Ax,
    const int8_t *restrict Ab,
    int64_t anz,
    int nthreads
) ;
m4_divert(0)

GrB_Info GB (_uop_tran)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    void **Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
) ;

