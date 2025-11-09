//------------------------------------------------------------------------------
// GB_bld__include.h: definitions for GB_bld__*.c
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// This file has been automatically generated from Generator/GB_bld.h
#include "math/GB_math.h"

GrB_Info GB (_bld__min_int8)
(
    int8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int8_t *restrict Sx,
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

GrB_Info GB (_bld__min_int16)
(
    int16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int16_t *restrict Sx,
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

GrB_Info GB (_bld__min_int32)
(
    int32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int32_t *restrict Sx,
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

GrB_Info GB (_bld__min_int64)
(
    int64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int64_t *restrict Sx,
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

GrB_Info GB (_bld__min_uint8)
(
    uint8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint8_t *restrict Sx,
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

GrB_Info GB (_bld__min_uint16)
(
    uint16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint16_t *restrict Sx,
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

GrB_Info GB (_bld__min_uint32)
(
    uint32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint32_t *restrict Sx,
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

GrB_Info GB (_bld__min_uint64)
(
    uint64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint64_t *restrict Sx,
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

GrB_Info GB (_bld__min_fp32)
(
    float *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const float *restrict Sx,
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

GrB_Info GB (_bld__min_fp64)
(
    double *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const double *restrict Sx,
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

GrB_Info GB (_bld__max_int8)
(
    int8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int8_t *restrict Sx,
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

GrB_Info GB (_bld__max_int16)
(
    int16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int16_t *restrict Sx,
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

GrB_Info GB (_bld__max_int32)
(
    int32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int32_t *restrict Sx,
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

GrB_Info GB (_bld__max_int64)
(
    int64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int64_t *restrict Sx,
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

GrB_Info GB (_bld__max_uint8)
(
    uint8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint8_t *restrict Sx,
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

GrB_Info GB (_bld__max_uint16)
(
    uint16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint16_t *restrict Sx,
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

GrB_Info GB (_bld__max_uint32)
(
    uint32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint32_t *restrict Sx,
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

GrB_Info GB (_bld__max_uint64)
(
    uint64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint64_t *restrict Sx,
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

GrB_Info GB (_bld__max_fp32)
(
    float *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const float *restrict Sx,
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

GrB_Info GB (_bld__max_fp64)
(
    double *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const double *restrict Sx,
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

GrB_Info GB (_bld__any_bool)
(
    bool *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const bool *restrict Sx,
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

GrB_Info GB (_bld__any_int8)
(
    int8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int8_t *restrict Sx,
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

GrB_Info GB (_bld__any_int16)
(
    int16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int16_t *restrict Sx,
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

GrB_Info GB (_bld__any_int32)
(
    int32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int32_t *restrict Sx,
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

GrB_Info GB (_bld__any_int64)
(
    int64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int64_t *restrict Sx,
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

GrB_Info GB (_bld__any_uint8)
(
    uint8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint8_t *restrict Sx,
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

GrB_Info GB (_bld__any_uint16)
(
    uint16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint16_t *restrict Sx,
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

GrB_Info GB (_bld__any_uint32)
(
    uint32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint32_t *restrict Sx,
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

GrB_Info GB (_bld__any_uint64)
(
    uint64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint64_t *restrict Sx,
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

GrB_Info GB (_bld__any_fp32)
(
    float *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const float *restrict Sx,
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

GrB_Info GB (_bld__any_fp64)
(
    double *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const double *restrict Sx,
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

GrB_Info GB (_bld__any_fc32)
(
    GxB_FC32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GxB_FC32_t *restrict Sx,
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

GrB_Info GB (_bld__any_fc64)
(
    GxB_FC64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GxB_FC64_t *restrict Sx,
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

GrB_Info GB (_bld__plus_int8)
(
    int8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int8_t *restrict Sx,
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

GrB_Info GB (_bld__plus_int16)
(
    int16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int16_t *restrict Sx,
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

GrB_Info GB (_bld__plus_int32)
(
    int32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int32_t *restrict Sx,
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

GrB_Info GB (_bld__plus_int64)
(
    int64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int64_t *restrict Sx,
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

GrB_Info GB (_bld__plus_uint8)
(
    uint8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint8_t *restrict Sx,
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

GrB_Info GB (_bld__plus_uint16)
(
    uint16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint16_t *restrict Sx,
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

GrB_Info GB (_bld__plus_uint32)
(
    uint32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint32_t *restrict Sx,
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

GrB_Info GB (_bld__plus_uint64)
(
    uint64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint64_t *restrict Sx,
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

GrB_Info GB (_bld__plus_fp32)
(
    float *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const float *restrict Sx,
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

GrB_Info GB (_bld__plus_fp64)
(
    double *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const double *restrict Sx,
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

GrB_Info GB (_bld__plus_fc32)
(
    GxB_FC32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GxB_FC32_t *restrict Sx,
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

GrB_Info GB (_bld__plus_fc64)
(
    GxB_FC64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GxB_FC64_t *restrict Sx,
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

GrB_Info GB (_bld__times_int8)
(
    int8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int8_t *restrict Sx,
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

GrB_Info GB (_bld__times_int16)
(
    int16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int16_t *restrict Sx,
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

GrB_Info GB (_bld__times_int32)
(
    int32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int32_t *restrict Sx,
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

GrB_Info GB (_bld__times_int64)
(
    int64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int64_t *restrict Sx,
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

GrB_Info GB (_bld__times_uint8)
(
    uint8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint8_t *restrict Sx,
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

GrB_Info GB (_bld__times_uint16)
(
    uint16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint16_t *restrict Sx,
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

GrB_Info GB (_bld__times_uint32)
(
    uint32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint32_t *restrict Sx,
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

GrB_Info GB (_bld__times_uint64)
(
    uint64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint64_t *restrict Sx,
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

GrB_Info GB (_bld__times_fp32)
(
    float *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const float *restrict Sx,
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

GrB_Info GB (_bld__times_fp64)
(
    double *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const double *restrict Sx,
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

GrB_Info GB (_bld__times_fc32)
(
    GxB_FC32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GxB_FC32_t *restrict Sx,
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

GrB_Info GB (_bld__times_fc64)
(
    GxB_FC64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GxB_FC64_t *restrict Sx,
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

GrB_Info GB (_bld__lor_bool)
(
    bool *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const bool *restrict Sx,
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

GrB_Info GB (_bld__land_bool)
(
    bool *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const bool *restrict Sx,
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

GrB_Info GB (_bld__lxor_bool)
(
    bool *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const bool *restrict Sx,
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

GrB_Info GB (_bld__eq_bool)
(
    bool *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const bool *restrict Sx,
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

GrB_Info GB (_bld__any_bool)
(
    bool *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const bool *restrict Sx,
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

GrB_Info GB (_bld__first_bool)
(
    bool *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const bool *restrict Sx,
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

GrB_Info GB (_bld__first_int8)
(
    int8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int8_t *restrict Sx,
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

GrB_Info GB (_bld__first_int16)
(
    int16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int16_t *restrict Sx,
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

GrB_Info GB (_bld__first_int32)
(
    int32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int32_t *restrict Sx,
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

GrB_Info GB (_bld__first_int64)
(
    int64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int64_t *restrict Sx,
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

GrB_Info GB (_bld__first_uint8)
(
    uint8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint8_t *restrict Sx,
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

GrB_Info GB (_bld__first_uint16)
(
    uint16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint16_t *restrict Sx,
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

GrB_Info GB (_bld__first_uint32)
(
    uint32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint32_t *restrict Sx,
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

GrB_Info GB (_bld__first_uint64)
(
    uint64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint64_t *restrict Sx,
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

GrB_Info GB (_bld__first_fp32)
(
    float *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const float *restrict Sx,
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

GrB_Info GB (_bld__first_fp64)
(
    double *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const double *restrict Sx,
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

GrB_Info GB (_bld__first_fc32)
(
    GxB_FC32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GxB_FC32_t *restrict Sx,
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

GrB_Info GB (_bld__first_fc64)
(
    GxB_FC64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GxB_FC64_t *restrict Sx,
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

GrB_Info GB (_bld__second_bool)
(
    bool *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const bool *restrict Sx,
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

GrB_Info GB (_bld__second_int8)
(
    int8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int8_t *restrict Sx,
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

GrB_Info GB (_bld__second_int16)
(
    int16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int16_t *restrict Sx,
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

GrB_Info GB (_bld__second_int32)
(
    int32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int32_t *restrict Sx,
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

GrB_Info GB (_bld__second_int64)
(
    int64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const int64_t *restrict Sx,
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

GrB_Info GB (_bld__second_uint8)
(
    uint8_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint8_t *restrict Sx,
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

GrB_Info GB (_bld__second_uint16)
(
    uint16_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint16_t *restrict Sx,
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

GrB_Info GB (_bld__second_uint32)
(
    uint32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint32_t *restrict Sx,
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

GrB_Info GB (_bld__second_uint64)
(
    uint64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const uint64_t *restrict Sx,
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

GrB_Info GB (_bld__second_fp32)
(
    float *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const float *restrict Sx,
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

GrB_Info GB (_bld__second_fp64)
(
    double *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const double *restrict Sx,
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

GrB_Info GB (_bld__second_fc32)
(
    GxB_FC32_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GxB_FC32_t *restrict Sx,
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

GrB_Info GB (_bld__second_fc64)
(
    GxB_FC64_t *restrict Tx,
    void *restrict Ti,
    bool Ti_is_32,
    const GxB_FC64_t *restrict Sx,
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

