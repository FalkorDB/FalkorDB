//------------------------------------------------------------------------------
// GB_sel__include.h: definitions for GB_sel__*.c
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// This file has been automatically generated from Generator/GB_sel.h
#include "math/GB_math.h"


GrB_Info GB (_sel_phase2__nonzombie_bool)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_int8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_int16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_int32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_int64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_uint8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_uint16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_uint32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_uint64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_fp32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_fp64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_fc32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase2__nonzombie_fc64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_int8)
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

GrB_Info GB (_sel_phase2__ne_thunk_int8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_int8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_int16)
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

GrB_Info GB (_sel_phase2__ne_thunk_int16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_int16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_int32)
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

GrB_Info GB (_sel_phase2__ne_thunk_int32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_int32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_int64)
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

GrB_Info GB (_sel_phase2__ne_thunk_int64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_int64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_uint8)
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

GrB_Info GB (_sel_phase2__ne_thunk_uint8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_uint8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_uint16)
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

GrB_Info GB (_sel_phase2__ne_thunk_uint16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_uint16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_uint32)
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

GrB_Info GB (_sel_phase2__ne_thunk_uint32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_uint32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_uint64)
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

GrB_Info GB (_sel_phase2__ne_thunk_uint64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_uint64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_fp32)
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

GrB_Info GB (_sel_phase2__ne_thunk_fp32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_fp32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_fp64)
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

GrB_Info GB (_sel_phase2__ne_thunk_fp64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_fp64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_fc32)
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

GrB_Info GB (_sel_phase2__ne_thunk_fc32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_fc32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ne_thunk_fc64)
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

GrB_Info GB (_sel_phase2__ne_thunk_fc64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ne_thunk_fc64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_bool)
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

GrB_Info GB (_sel_phase2__eq_thunk_bool)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_bool)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_int8)
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

GrB_Info GB (_sel_phase2__eq_thunk_int8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_int8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_int16)
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

GrB_Info GB (_sel_phase2__eq_thunk_int16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_int16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_int32)
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

GrB_Info GB (_sel_phase2__eq_thunk_int32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_int32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_int64)
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

GrB_Info GB (_sel_phase2__eq_thunk_int64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_int64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_uint8)
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

GrB_Info GB (_sel_phase2__eq_thunk_uint8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_uint8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_uint16)
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

GrB_Info GB (_sel_phase2__eq_thunk_uint16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_uint16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_uint32)
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

GrB_Info GB (_sel_phase2__eq_thunk_uint32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_uint32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_uint64)
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

GrB_Info GB (_sel_phase2__eq_thunk_uint64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_uint64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_fp32)
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

GrB_Info GB (_sel_phase2__eq_thunk_fp32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_fp32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_fp64)
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

GrB_Info GB (_sel_phase2__eq_thunk_fp64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_fp64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_fc32)
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

GrB_Info GB (_sel_phase2__eq_thunk_fc32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_fc32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__eq_thunk_fc64)
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

GrB_Info GB (_sel_phase2__eq_thunk_fc64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__eq_thunk_fc64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__gt_thunk_int8)
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

GrB_Info GB (_sel_phase2__gt_thunk_int8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__gt_thunk_int8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__gt_thunk_int16)
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

GrB_Info GB (_sel_phase2__gt_thunk_int16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__gt_thunk_int16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__gt_thunk_int32)
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

GrB_Info GB (_sel_phase2__gt_thunk_int32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__gt_thunk_int32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__gt_thunk_int64)
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

GrB_Info GB (_sel_phase2__gt_thunk_int64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__gt_thunk_int64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__gt_thunk_uint8)
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

GrB_Info GB (_sel_phase2__gt_thunk_uint8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__gt_thunk_uint8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__gt_thunk_uint16)
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

GrB_Info GB (_sel_phase2__gt_thunk_uint16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__gt_thunk_uint16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__gt_thunk_uint32)
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

GrB_Info GB (_sel_phase2__gt_thunk_uint32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__gt_thunk_uint32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__gt_thunk_uint64)
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

GrB_Info GB (_sel_phase2__gt_thunk_uint64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__gt_thunk_uint64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__gt_thunk_fp32)
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

GrB_Info GB (_sel_phase2__gt_thunk_fp32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__gt_thunk_fp32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__gt_thunk_fp64)
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

GrB_Info GB (_sel_phase2__gt_thunk_fp64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__gt_thunk_fp64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ge_thunk_int8)
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

GrB_Info GB (_sel_phase2__ge_thunk_int8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ge_thunk_int8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ge_thunk_int16)
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

GrB_Info GB (_sel_phase2__ge_thunk_int16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ge_thunk_int16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ge_thunk_int32)
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

GrB_Info GB (_sel_phase2__ge_thunk_int32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ge_thunk_int32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ge_thunk_int64)
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

GrB_Info GB (_sel_phase2__ge_thunk_int64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ge_thunk_int64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ge_thunk_uint8)
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

GrB_Info GB (_sel_phase2__ge_thunk_uint8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ge_thunk_uint8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ge_thunk_uint16)
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

GrB_Info GB (_sel_phase2__ge_thunk_uint16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ge_thunk_uint16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ge_thunk_uint32)
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

GrB_Info GB (_sel_phase2__ge_thunk_uint32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ge_thunk_uint32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ge_thunk_uint64)
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

GrB_Info GB (_sel_phase2__ge_thunk_uint64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ge_thunk_uint64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ge_thunk_fp32)
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

GrB_Info GB (_sel_phase2__ge_thunk_fp32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ge_thunk_fp32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__ge_thunk_fp64)
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

GrB_Info GB (_sel_phase2__ge_thunk_fp64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__ge_thunk_fp64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__lt_thunk_int8)
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

GrB_Info GB (_sel_phase2__lt_thunk_int8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__lt_thunk_int8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__lt_thunk_int16)
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

GrB_Info GB (_sel_phase2__lt_thunk_int16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__lt_thunk_int16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__lt_thunk_int32)
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

GrB_Info GB (_sel_phase2__lt_thunk_int32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__lt_thunk_int32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__lt_thunk_int64)
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

GrB_Info GB (_sel_phase2__lt_thunk_int64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__lt_thunk_int64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__lt_thunk_uint8)
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

GrB_Info GB (_sel_phase2__lt_thunk_uint8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__lt_thunk_uint8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__lt_thunk_uint16)
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

GrB_Info GB (_sel_phase2__lt_thunk_uint16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__lt_thunk_uint16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__lt_thunk_uint32)
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

GrB_Info GB (_sel_phase2__lt_thunk_uint32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__lt_thunk_uint32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__lt_thunk_uint64)
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

GrB_Info GB (_sel_phase2__lt_thunk_uint64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__lt_thunk_uint64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__lt_thunk_fp32)
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

GrB_Info GB (_sel_phase2__lt_thunk_fp32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__lt_thunk_fp32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__lt_thunk_fp64)
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

GrB_Info GB (_sel_phase2__lt_thunk_fp64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__lt_thunk_fp64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__le_thunk_int8)
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

GrB_Info GB (_sel_phase2__le_thunk_int8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__le_thunk_int8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__le_thunk_int16)
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

GrB_Info GB (_sel_phase2__le_thunk_int16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__le_thunk_int16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__le_thunk_int32)
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

GrB_Info GB (_sel_phase2__le_thunk_int32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__le_thunk_int32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__le_thunk_int64)
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

GrB_Info GB (_sel_phase2__le_thunk_int64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__le_thunk_int64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__le_thunk_uint8)
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

GrB_Info GB (_sel_phase2__le_thunk_uint8)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__le_thunk_uint8)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__le_thunk_uint16)
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

GrB_Info GB (_sel_phase2__le_thunk_uint16)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__le_thunk_uint16)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__le_thunk_uint32)
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

GrB_Info GB (_sel_phase2__le_thunk_uint32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__le_thunk_uint32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__le_thunk_uint64)
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

GrB_Info GB (_sel_phase2__le_thunk_uint64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__le_thunk_uint64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__le_thunk_fp32)
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

GrB_Info GB (_sel_phase2__le_thunk_fp32)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__le_thunk_fp32)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;


GrB_Info GB (_sel_phase1__le_thunk_fp64)
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

GrB_Info GB (_sel_phase2__le_thunk_fp64)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB (_sel_bitmap__le_thunk_fp64)
(
    GrB_Matrix C,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;

