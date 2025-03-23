//------------------------------------------------------------------------------
// GB_sel:  hard-coded functions for selection operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "select/GB_select.h"
#include "FactoryKernels/GB_sel__include.h"

#define GB_ENTRY_SELECTOR
GB_atype
GB_ytype
GB_test_value_of_entry
GB_select_entry
GB_iso_select

#include "select/include/GB_select_shared_definitions.h"

m4_divert(if_phase1)
//------------------------------------------------------------------------------
// GB_sel_phase1
//------------------------------------------------------------------------------

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
)
{ 
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #include "select/template/GB_select_entry_phase1_template.c"
    return (GrB_SUCCESS) ;
}

m4_divert(if_phase2)
//------------------------------------------------------------------------------
// GB_sel_phase2
//------------------------------------------------------------------------------

GrB_Info GB (_sel_phase2)
(
    GrB_Matrix C,
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #include "select/template/GB_select_phase2_template.c"
    return (GrB_SUCCESS) ;
}

m4_divert(if_bitmap)
//------------------------------------------------------------------------------
// GB_sel_bitmap
//------------------------------------------------------------------------------

GrB_Info GB (_sel_bitmap)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
)
{ 
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #include "select/template/GB_select_bitmap_template.c"
    return (GrB_SUCCESS) ;
}

