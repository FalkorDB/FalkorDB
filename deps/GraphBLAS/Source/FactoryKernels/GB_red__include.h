//------------------------------------------------------------------------------
// GB_red__include.h: definitions for GB_red__*.c
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// This file has been automatically generated from Generator/GB_red.h

GrB_Info GB (_red__min_int8)
(
    int8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__min_int16)
(
    int16_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__min_int32)
(
    int32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__min_int64)
(
    int64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__min_uint8)
(
    uint8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__min_uint16)
(
    uint16_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__min_uint32)
(
    uint32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__min_uint64)
(
    uint64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__min_fp32)
(
    float *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__min_fp64)
(
    double *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__max_int8)
(
    int8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__max_int16)
(
    int16_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__max_int32)
(
    int32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__max_int64)
(
    int64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__max_uint8)
(
    uint8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__max_uint16)
(
    uint16_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__max_uint32)
(
    uint32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__max_uint64)
(
    uint64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__max_fp32)
(
    float *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__max_fp64)
(
    double *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_bool)
(
    bool *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_int8)
(
    int8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_int16)
(
    int16_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_int32)
(
    int32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_int64)
(
    int64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_uint8)
(
    uint8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_uint16)
(
    uint16_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_uint32)
(
    uint32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_uint64)
(
    uint64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_fp32)
(
    float *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_fp64)
(
    double *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_fc32)
(
    GxB_FC32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_fc64)
(
    GxB_FC64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_int8)
(
    int8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_int16)
(
    int16_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_int32)
(
    int32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_int64)
(
    int64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_uint8)
(
    uint8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_uint16)
(
    uint16_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_uint32)
(
    uint32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_uint64)
(
    uint64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_fp32)
(
    float *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_fp64)
(
    double *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_fc32)
(
    GxB_FC32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__plus_fc64)
(
    GxB_FC64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_int8)
(
    int8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_int16)
(
    int16_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_int32)
(
    int32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_int64)
(
    int64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_uint8)
(
    uint8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_uint16)
(
    uint16_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_uint32)
(
    uint32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_uint64)
(
    uint64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_fp32)
(
    float *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_fp64)
(
    double *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_fc32)
(
    GxB_FC32_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__times_fc64)
(
    GxB_FC64_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__lor_bool)
(
    bool *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__land_bool)
(
    bool *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__lxor_bool)
(
    bool *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__eq_bool)
(
    bool *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB (_red__any_bool)
(
    bool *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

