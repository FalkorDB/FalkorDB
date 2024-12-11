//------------------------------------------------------------------------------
// GB_enumify_subref: enumerate a GrB_extract problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
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
    int Ikind,              // 0: all (no I), 1: range, 2: stride, 3: list
    int Jkind,              // ditto, or 0 if not used
    bool need_qsort,        // true if qsort needs to be called
    bool I_has_duplicates,  // true if I has duplicate entries
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
    int csparsity, asparsity ;
    GB_enumify_sparsity (&csparsity, C_sparsity) ;
    GB_enumify_sparsity (&asparsity, A_sparsity) ;

    int needqsort = (need_qsort) ? 1 : 0 ;
    int ihasdupl = (I_has_duplicates) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // construct the subref method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 14 (4 hex digits)

    (*method_code) =
                                               // range        bits
                /// need_qsort, I_has_duplicates (1 hex digit)
                GB_LSHIFT (ihasdupl   , 13) |  // 0 to 1       1
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

