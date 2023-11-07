
// SPDX-License-Identifier: Apache-2.0
GrB_Info GB (_bld)
(
    GB_ttype_parameter *restrict Tx,
    int64_t  *restrict Ti,
    const GB_stype_parameter *restrict Sx,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

