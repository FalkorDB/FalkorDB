//------------------------------------------------------------------------------
// GB_assign_zombie2: delete all entries in C(i,:) for GB_assign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// C(i,:)<!> = anything: GrB_Row_assign or GrB_Col_assign with an empty
// complemented mask requires all entries in C(i,:) to be deleted.
// C must be sparse or hypersparse.

// C->iso is not affected.

#include "GB_assign.h"
#include "GB_assign_zombie.h"

GrB_Info GB_assign_zombie2
(
    GrB_Matrix C,
    const int64_t i
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_FULL (C)) ;
    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;      // binary search is used
    ASSERT (!GB_PENDING (C)) ;

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    const int64_t *restrict Cp = C->p ;
    int64_t *restrict Ci = C->i ;
    const int64_t Cnvec = C->nvec ;
    int64_t nzombies = C->nzombies ;
    const int64_t zorig = nzombies ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (Cnvec, chunk, nthreads_max) ;
    int ntasks = (nthreads == 1) ? 1 : (64 * nthreads) ;

    //--------------------------------------------------------------------------
    // C(i,:) = empty
    //--------------------------------------------------------------------------

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {
        int64_t kfirst, klast ;
        GB_PARTITION (kfirst, klast, Cnvec, taskid, ntasks) ;
        for (int64_t k = kfirst ; k < klast ; k++)
        {

            //------------------------------------------------------------------
            // find C(i,j)
            //------------------------------------------------------------------

            int64_t pC = Cp [k] ;
            int64_t pC_end = Cp [k+1] ;
            int64_t pright = pC_end - 1 ;
            bool found, is_zombie ;
            GB_BINARY_SEARCH_ZOMBIE (i, Ci, pC, pright, found, zorig,
                is_zombie) ;

            //------------------------------------------------------------------
            // if found and not a zombie, mark it as a zombie
            //------------------------------------------------------------------

            if (found && !is_zombie)
            { 
                ASSERT (i == Ci [pC]) ;
                nzombies++ ;
                Ci [pC] = GB_FLIP (i) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nzombies = nzombies ;
    return (GrB_SUCCESS) ;
}

