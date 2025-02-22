
// SPDX-License-Identifier: Apache-2.0
GrB_Info GB (_bld)
(
    GB_ttype_parameter *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GB_stype_parameter *restrict Sx,
    int64_t nvals,
    int64_t ndupl,
    const void *restrict I_work,
    bool I_is_32,
    const void *restrict K_work,
    bool K_is_32,
    const int64_t duplicate_entry,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

