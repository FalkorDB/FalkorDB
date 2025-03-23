//------------------------------------------------------------------------------
// GB_sel:  hard-coded functions for selection operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "select/GB_select.h"
#include "FactoryKernels/GB_sel__include.h"

#define GB_ENTRY_SELECTOR
#define GB_A_TYPE uint8_t
#define GB_Y_TYPE uint8_t
#define GB_TEST_VALUE_OF_ENTRY(keep,p) bool keep = (Ax [p] <= y)
#define GB_SELECT_ENTRY(Cx,pC,Ax,pA) Cx [pC] = Ax [pA]

#include "select/include/GB_select_shared_definitions.h"

//------------------------------------------------------------------------------
// GB_sel_phase1
//------------------------------------------------------------------------------

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
)
{ 
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #include "select/template/GB_select_entry_phase1_template.c"
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_sel_phase2
//------------------------------------------------------------------------------

GrB_Info GB (_sel_phase2__le_thunk_uint8)
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

//------------------------------------------------------------------------------
// GB_sel_bitmap
//------------------------------------------------------------------------------

GrB_Info GB (_sel_bitmap__le_thunk_uint8)
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

