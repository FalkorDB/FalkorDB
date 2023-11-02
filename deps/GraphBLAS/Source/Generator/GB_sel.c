//------------------------------------------------------------------------------
// GB_sel:  hard-coded functions for selection operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_select.h"
#include "GB_ek_slice.h"
#include "GB_sel__include.h"

#define GB_ENTRY_SELECTOR
GB_atype
GB_ytype
GB_test_value_of_entry
GB_select_entry
GB_iso_select

#include "GB_select_shared_definitions.h"

m4_divert(if_phase1)
//------------------------------------------------------------------------------
// GB_sel_phase1
//------------------------------------------------------------------------------

GrB_Info GB (_sel_phase1)
(
    int64_t *restrict Cp,
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #include "GB_select_entry_phase1_template.c"
    return (GrB_SUCCESS) ;
}

m4_divert(if_phase2)
//------------------------------------------------------------------------------
// GB_sel_phase2
//------------------------------------------------------------------------------

GrB_Info GB (_sel_phase2)
(
    int64_t *restrict Ci,
    GB_void *restrict Cx_out,
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    GB_A_TYPE *restrict Cx = (GB_A_TYPE *) Cx_out ;
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #include "GB_select_phase2.c"
    return (GrB_SUCCESS) ;
}

m4_divert(if_bitmap)
//------------------------------------------------------------------------------
// GB_sel_bitmap
//------------------------------------------------------------------------------

GrB_Info GB (_sel_bitmap)
(
    int8_t *Cb,
    int64_t *cnvals_handle,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
)
{ 
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #include "GB_select_bitmap_template.c"
    return (GrB_SUCCESS) ;
}

