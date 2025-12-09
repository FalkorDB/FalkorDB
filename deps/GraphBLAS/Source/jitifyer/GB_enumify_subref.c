//------------------------------------------------------------------------------
// GB_enumify_subref: enumerate a GrB_extract problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Enumify a subref operation: C = A(I,J)

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_enumify_subref      // enumerate a GrB_extract problem
(
    // output:
    uint64_t *method_code,  // unique encoding of the entire operation
    // C matrix:
    GrB_Matrix C,
    // index types:
    bool I_is_32,           // if true, I is 32-bit; else 64-bit
    bool J_is_32,           // if true, J is 32-bit; else 64-bit (bitmap only)
    int Ikind,              // 0: all (no I), 1: range, 2: stride, 3: list
    int Jkind,              // ditto, or 0 if not used
    bool need_qsort,        // true if qsort needs to be called
    GrB_Matrix R,
    // A matrix:
    GrB_Matrix A
)
{ 

    //--------------------------------------------------------------------------
    // get the type of C (same as the type of A) and enumify it
    //--------------------------------------------------------------------------

    GrB_Type ctype = C->type ;
    ASSERT (!C->iso) ;
    int ccode = ctype->code ;               // 1 to 14

    //--------------------------------------------------------------------------
    // enumify the sparsity structures of C and A
    //--------------------------------------------------------------------------

    int C_sparsity = GB_sparsity (C) ;
    int A_sparsity = GB_sparsity (A) ;
    int R_sparsity = GB_sparsity (R) ;
    int csparsity, asparsity, rsparsity ;
    GB_enumify_sparsity (&csparsity, C_sparsity) ;
    GB_enumify_sparsity (&asparsity, A_sparsity) ;
    GB_enumify_sparsity (&rsparsity, R_sparsity) ;

    int needqsort = (need_qsort) ? 1 : 0 ;

    int i_is_32 = (I_is_32) ? 1 : 0 ;
    int j_is_32 = (J_is_32) ? 1 : 0 ;

    int cp_is_32 = (C->p_is_32) ? 1 : 0 ;
    int cj_is_32 = (C->j_is_32) ? 1 : 0 ;
    int ci_is_32 = (C->i_is_32) ? 1 : 0 ;

    int ap_is_32 = (A->p_is_32) ? 1 : 0 ;
    int aj_is_32 = (A->j_is_32) ? 1 : 0 ;
    int ai_is_32 = (A->i_is_32) ? 1 : 0 ;

    int rp_is_32 = (R != NULL && R->p_is_32) ? 1 : 0 ;
    int rj_is_32 = (R != NULL && R->j_is_32) ? 1 : 0 ;
    int ri_is_32 = (R != NULL && R->i_is_32) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // construct the subref method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 28 (7 hex digits)

    (*method_code) =
                                               // range        bits
                // R integer sizes and sparsity
                GB_LSHIFT (rp_is_32   , 27) |  // 0 to 1       1
                GB_LSHIFT (rj_is_32   , 26) |  // 0 to 1       1
                GB_LSHIFT (ri_is_32   , 25) |  // 0 to 1       1
                GB_LSHIFT (rsparsity  , 23) |  // 0 to 3       2

                // C, A integer sizes (2 hex digits)
                GB_LSHIFT (cp_is_32   , 21) |  // 0 to 1       1
                GB_LSHIFT (cj_is_32   , 20) |  // 0 to 1       1
                GB_LSHIFT (ci_is_32   , 19) |  // 0 to 1       1

                GB_LSHIFT (ap_is_32   , 18) |  // 0 to 1       1
                GB_LSHIFT (aj_is_32   , 17) |  // 0 to 1       1
                GB_LSHIFT (ai_is_32   , 16) |  // 0 to 1       1

                // need_qsort, I and J bits (1 hex digit)
                GB_LSHIFT (i_is_32    , 15) |  // 0 to 1       1
                GB_LSHIFT (j_is_32    , 14) |  // 0 to 1       1
                // 13: unused
                GB_LSHIFT (needqsort  , 12) |  // 0 to 1       1

                // Ikind, Jkind (1 hex digit)
                GB_LSHIFT (Ikind      , 10) |  // 0 to 3       2
                GB_LSHIFT (Jkind      ,  8) |  // 0 to 3       2

                // type of C and A (1 hex digit)
                GB_LSHIFT (ccode      ,  4) |  // 1 to 14      4

                // sparsity structures of C and A (1 hex digit)
                GB_LSHIFT (csparsity  ,  2) |  // 0 to 3       2
                GB_LSHIFT (asparsity  ,  0) ;  // 0 to 3       2

}

