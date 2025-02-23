//------------------------------------------------------------------------------
// GB_assign_zombie1: delete all entries in C(:,j) for GB_assign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(:,j)<!> = anything: GrB_Row_assign or GrB_Col_assign with an empty
// complemented mask requires all entries in the C(:,j) vector to be deleted.
// C must be sparse or hypersparse.

// C->iso is not affected.

#include "assign/GB_assign.h"
#include "assign/GB_assign_zombie.h"

GrB_Info GB_assign_zombie1
(
    GrB_Matrix C,
    const int64_t j
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_FULL (C)) ;
    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (!GB_PENDING (C)) ;

    //--------------------------------------------------------------------------
    // get C(:,j)
    //--------------------------------------------------------------------------

    GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;
    GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;
    const void *Ch = C->h ;
    int64_t pC_start, pC_end ;
    const int64_t Cnvec = C->nvec ;

    if (Ch != NULL)
    { 
        // C is hypersparse
        const void *C_Yp = (C->Y == NULL) ? NULL : C->Y->p ;
        const void *C_Yi = (C->Y == NULL) ? NULL : C->Y->i ;
        const void *C_Yx = (C->Y == NULL) ? NULL : C->Y->x ;
        const int64_t C_hash_bits = (C->Y == NULL) ? 0 : (C->Y->vdim - 1) ;
        GB_hyper_hash_lookup (C->p_is_32, C->j_is_32,
            Ch, Cnvec, Cp, C_Yp, C_Yi, C_Yx, C_hash_bits,
            j, &pC_start, &pC_end) ;
    }
    else
    { 
        // C is sparse
        pC_start = GB_IGET (Cp, j) ;
        pC_end   = GB_IGET (Cp, j+1) ;
    }

    int64_t cjnz = pC_end - pC_start ;
    int64_t nzombies = C->nzombies ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (cjnz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // C(:,j) = empty
    //--------------------------------------------------------------------------

    int64_t pC ;
    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        reduction(+:nzombies)
    for (pC = pC_start ; pC < pC_end ; pC++)
    {
        int64_t i = GB_IGET (Ci, pC) ;
        if (!GB_IS_ZOMBIE (i))
        { 
            // delete C(i,j) by marking it as a zombie
            nzombies++ ;
            i = GB_ZOMBIE (i) ;
            GB_ISET (Ci, pC, i) ;       // Ci [pC] = i ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nzombies = nzombies ;
    return (GrB_SUCCESS) ;
}

