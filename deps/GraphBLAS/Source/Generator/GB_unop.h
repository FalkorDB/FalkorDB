
// SPDX-License-Identifier: Apache-2.0
m4_divert(if_unop_apply_enabled)
GrB_Info GB (_unop_apply)
(
    GB_void *Cx,
    const GB_void *Ax,
    const int8_t *restrict Ab,
    int64_t anz,
    int nthreads
) ;
m4_divert(0)

GrB_Info GB (_unop_tran)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
) ;

