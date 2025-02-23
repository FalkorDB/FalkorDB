//------------------------------------------------------------------------------
// GB_subassign_zombie: C(I,J)<!,repl> = empty ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 00: C(I,J)<!,repl> = empty ; using S

// M:           NULL
// Mask_comp:   true
// C_replace:   true
// accum:       any (present or not; result is the same)
// A:           any (scalar or matrix; result is the same)
// S:           constructed

// C: not bitmap

// C->iso is not affected.

#include "assign/GB_subassign_methods.h"
#define GB_GENERIC
#define GB_SCALAR_ASSIGN 0
#include "assign/include/GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_Matrix_free (&S) ;

GrB_Info GB_subassign_zombie
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
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

    GrB_Info info ;
    GrB_Matrix S = NULL ;
    ASSERT (!GB_IS_BITMAP (C)) ; ASSERT (!GB_IS_FULL (C)) ;

    //--------------------------------------------------------------------------
    // S = C(I,J), but do not construct the S->H hyper_hash
    //--------------------------------------------------------------------------

    struct GB_Matrix_opaque S_header ;
    GB_CLEAR_MATRIX_HEADER (S, &S_header) ;
    GB_OK (GB_subassign_symbolic (S, C, I, I_is_32, ni, J, J_is_32, nj,
        /* S_can_be_jumbled: */ false, Werk)) ;
    ASSERT (GB_JUMBLED_OK (S)) ;        // S can be returned as jumbled
    // the S->Y hyper_hash is not needed

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    ASSERT (S->type == GrB_UINT32 || S->type == GrB_UINT64) ;
    const bool Sx_is_32 = (S->type->code == GB_UINT32_code) ;
    GB_MDECL (Sx, const, u) ;
    Sx = S->x ;
    GB_IPTR (Sx, Sx_is_32) ;

    GB_Ci_DECLARE (Ci, ) ; GB_Ci_PTR (Ci, C) ;

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
        int64_t pC = GB_IGET (Sx, pS) ;
        int64_t i = GB_IGET (Ci, pC) ;
        // ----[X A 0] or [X . 0]-----------------------------------------------
        // action: ( X ): still a zombie
        // ----[C A 0] or [C . 0]-----------------------------------------------
        // action: C_repl: ( delete ): becomes a zombie
        if (!GB_IS_ZOMBIE (i))
        { 
            nzombies++ ;
            i = GB_ZOMBIE (i) ;
            GB_ISET (Ci, pC, i) ;   // Ci [pC] = i ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->nzombies = nzombies ;
    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

