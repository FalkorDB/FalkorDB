//------------------------------------------------------------------------------
// GB_select_positional_phase1.c: count entries for C=select(A,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse or hypersparse

// JIT: not needed, but 3 variants possible (A sparse, hyper, or full for DIAG)

#include "GB_select.h"
#include "GB_ek_slice.h"

GrB_Info GB_select_positional_phase2
(
    int64_t *restrict Ci,
    GB_void *restrict Cx,
    const int64_t *restrict Zp,
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const bool flipij,
    const int64_t ithunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A) ||
        (opcode == GB_DIAG_idxunop_code)) ;
    ASSERT (GB_OPCODE_IS_POSITIONAL (opcode)
        || (opcode == GB_NONZOMBIE_idxunop_code && A->iso)) ;

    //--------------------------------------------------------------------------
    // phase1: positional operators and nonzombie iso selector
    //--------------------------------------------------------------------------

    #define GB_A_TYPE GB_void
    #include "GB_select_shared_definitions.h"

    if (A->iso)
    {

        //----------------------------------------------------------------------
        // A is iso
        //----------------------------------------------------------------------

        #undef  GB_ISO_SELECT
        #define GB_ISO_SELECT 1

        switch (opcode)
        {

            case GB_TRIL_idxunop_code      : 
                #define GB_TRIL_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_TRIU_idxunop_code      : 
                #define GB_TRIU_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_DIAG_idxunop_code      : 
                #define GB_DIAG_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_OFFDIAG_idxunop_code   : 
            case GB_DIAGINDEX_idxunop_code : 
                #define GB_OFFDIAG_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_ROWINDEX_idxunop_code  : 
                #define GB_ROWINDEX_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_ROWLE_idxunop_code     : 
                #define GB_ROWLE_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_ROWGT_idxunop_code     : 
                #define GB_ROWGT_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_NONZOMBIE_idxunop_code : 
                // keep A(i,j) if it's not a zombie
                #define GB_ENTRY_SELECTOR
                #define GB_TEST_VALUE_OF_ENTRY(keep,p) bool keep = (i >= 0)
                // A is iso and thus so is C, so do not select the value
                #define GB_SELECT_ENTRY(Cx,pC,Ax,pA)
                #include "GB_select_phase2.c"
                #undef  GB_ENTRY_SELECTOR
                break ;

            default: ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A is non-iso
        //----------------------------------------------------------------------

        #undef  GB_ISO_SELECT
        #define GB_ISO_SELECT 0

        switch (opcode)
        {

            case GB_TRIL_idxunop_code      : 
                #define GB_TRIL_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_TRIU_idxunop_code      : 
                #define GB_TRIU_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_DIAG_idxunop_code      : 
                #define GB_DIAG_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_OFFDIAG_idxunop_code   : 
            case GB_DIAGINDEX_idxunop_code : 
                #define GB_OFFDIAG_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_ROWINDEX_idxunop_code  : 
                #define GB_ROWINDEX_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_ROWLE_idxunop_code     : 
                #define GB_ROWLE_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            case GB_ROWGT_idxunop_code     : 
                #define GB_ROWGT_SELECTOR
                #include "GB_select_phase2.c"
                break ;

            default: ;
        }
    }

    return (GrB_SUCCESS) ;
}

