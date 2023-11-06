//------------------------------------------------------------------------------
// GB_subassign_zombie: C(I,J)<!,repl> = empty ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// Method 00: C(I,J)<!,repl> = empty ; using S

// M:           NULL
// Mask_comp:   true
// C_replace:   true
// accum:       any (present or not; result is the same)
// A:           any (scalar or matrix; result is the same)
// S:           constructed

// C: not bitmap

// C->iso is not affected.

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_Matrix_free (&S) ;
#include "GB_static_header.h"

GrB_Info GB_subassign_zombie
(
    GrB_Matrix C,
    // input:
    const GrB_Index *I,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_BITMAP (C)) ; ASSERT (!GB_IS_FULL (C)) ;

    //--------------------------------------------------------------------------
    // S = C(I,J), but do not construct the S->H hyper_hash
    //--------------------------------------------------------------------------

    GrB_Info info ;
    struct GB_Matrix_opaque S_header ;
    GrB_Matrix S = NULL ;
    GB_CLEAR_STATIC_HEADER (S, &S_header) ;
    GB_OK (GB_subassign_symbolic (S, C, I, ni, J, nj, false, Werk)) ;
    ASSERT (GB_JUMBLED_OK (S)) ;        // S can be returned as jumbled
    // the S->Y hyper_hash is not needed

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    const int64_t *restrict Sx = (int64_t *) S->x ;
    int64_t *restrict Ci = C->i ;

    //--------------------------------------------------------------------------
    // Method 00: C(I,J)<!,repl> = empty ; using S
    //--------------------------------------------------------------------------

    // Time: Optimal, O(nnz(S)), assuming S has already been constructed.

    //--------------------------------------------------------------------------
    // Parallel: all entries in S can be processed entirely in parallel.
    //--------------------------------------------------------------------------

    // All entries in C(I,J) are deleted.  The result does not depend on A or
    // the scalar.

    int64_t snz = GB_nnz (S) ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (snz, chunk, nthreads_max) ;

    int64_t nzombies = C->nzombies ;

    int64_t pS ;
    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        reduction(+:nzombies)
    for (pS = 0 ; pS < snz ; pS++)
    {
        // S (inew,jnew) is a pointer back into C (I(inew), J(jnew))
        int64_t pC = Sx [pS] ;
        int64_t i = Ci [pC] ;
        // ----[X A 0] or [X . 0]-----------------------------------------------
        // action: ( X ): still a zombie
        // ----[C A 0] or [C . 0]-----------------------------------------------
        // action: C_repl: ( delete ): becomes a zombie
        if (!GB_IS_ZOMBIE (i))
        { 
            nzombies++ ;
            Ci [pC] = GB_FLIP (i) ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->nzombies = nzombies ;
    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

