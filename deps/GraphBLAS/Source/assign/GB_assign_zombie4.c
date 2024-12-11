//------------------------------------------------------------------------------
// GB_assign_zombie4: delete entries in C(i,:) for C_replace_phase
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: possible: 96 variants. Could use one for each mask type (6: 1, 2,
// 4, 8, 16 bytes and structural), for each matrix type (4: bitmap/full/sparse/
// hyper), mask comp (2), C sparsity (2: sparse/hyper): 6*4*2*2 = 96 variants,
// so a JIT kernel is reasonable.

// For GrB_Row_assign or GrB_Col_assign, C(i,J)<M,repl>=..., if C_replace is
// true, and mask M is present, then any entry C(i,j) outside the list J must
// be deleted, if M(0,j)=0.

// GB_assign_zombie3 and GB_assign_zombie4 are transposes of each other.

// C must be sparse or hypersparse.
// M can have any sparsity structure: hypersparse, sparse, bitmap, or full

// C->iso is not affected.

#include "assign/GB_assign.h"
#include "assign/GB_assign_zombie.h"

GrB_Info GB_assign_zombie4
(
    GrB_Matrix C,                   // the matrix C, or a copy
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const int64_t i,                // index of entries to delete
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_FULL (C)) ;
    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;      // binary search on C
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (!GB_ZOMBIES (M)) ; 
    ASSERT (!GB_JUMBLED (M)) ;
    ASSERT (!GB_PENDING (M)) ; 
    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    const int64_t *restrict Ch = C->h ;
    const int64_t *restrict Cp = C->p ;
    const int64_t Cnvec = C->nvec ;
    int64_t *restrict Ci = C->i ;
    int64_t nzombies = C->nzombies ;
    const int64_t zorig = nzombies ;

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    const int64_t *restrict Mp = M->p ;
    const int64_t *restrict Mh = M->h ;
    const int8_t  *restrict Mb = M->b ;
    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
    const size_t msize = M->type->size ;
    const int64_t Mnvec = M->nvec ;
    ASSERT (M->vlen == 1) ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full = GB_IS_FULL (M) ;
    const int64_t *restrict M_Yp = (M->Y == NULL) ? NULL : M->Y->p ;
    const int64_t *restrict M_Yi = (M->Y == NULL) ? NULL : M->Y->i ;
    const int64_t *restrict M_Yx = (M->Y == NULL) ? NULL : M->Y->x ;
    const int64_t M_hash_bits = (M->Y == NULL) ? 0 : (M->Y->vdim - 1) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (Cnvec, chunk, nthreads_max) ;
    int ntasks = (nthreads == 1) ? 1 : (64 * nthreads) ;

    //--------------------------------------------------------------------------
    // delete entries in C(i,:)
    //--------------------------------------------------------------------------

    // The entry C(i,j) is deleted if j is not in the J, and if M(0,j)=0 (if
    // the mask is not complemented) or M(0,j)=1 (if the mask is complemented.

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
            // get C(:,j) and determine if j is outside the list J
            //------------------------------------------------------------------

            int64_t j = GBH_C (Ch, k) ;
            bool j_outside = !GB_ij_is_in_list (J, nJ, j, Jkind, Jcolon) ;
            if (j_outside)
            {

                //--------------------------------------------------------------
                // j is not in J; find C(i,j)
                //--------------------------------------------------------------

                int64_t pC = Cp [k] ;
                int64_t pC_end = Cp [k+1] ;
                int64_t pright = pC_end - 1 ;
                bool found, is_zombie ;
                GB_BINARY_SEARCH_ZOMBIE (i, Ci, pC, pright, found, zorig,
                    is_zombie) ;

                //--------------------------------------------------------------
                // delete C(i,j) if found, not a zombie, and M(0,j) allows it
                //--------------------------------------------------------------

                if (found && !is_zombie)
                {

                    //----------------------------------------------------------
                    // C(i,j) is a live entry not in the C(I,J) submatrix
                    //----------------------------------------------------------

                    // Check the mask M to see if it should be deleted.
                    bool mij = false ;

                    if (M_is_bitmap || M_is_full)
                    { 
                        // M is bitmap/full
                        int64_t pM = j ;
                        mij = GBB_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                    }
                    else
                    {
                        // M is sparse or hypersparse
                        int64_t pM, pM_end ;

                        if (M_is_hyper)
                        { 
                            // M is hypersparse
                            GB_hyper_hash_lookup (Mh, Mnvec, Mp, M_Yp, M_Yi,
                                M_Yx, M_hash_bits, j, &pM, &pM_end) ;
                        }
                        else
                        { 
                            // M is sparse
                            pM     = Mp [j] ;
                            pM_end = Mp [j+1] ;
                        }

                        if (pM < pM_end)
                        { 
                            // found it
                            mij = GB_MCAST (Mx, pM, msize) ;
                        }
                    }

                    if (Mask_comp)
                    { 
                        // negate the mask if Mask_comp is true
                        mij = !mij ;
                    }
                    if (!mij)
                    { 
                        // delete C(i,j) by marking it as a zombie
                        nzombies++ ;
                        Ci [pC] = GB_ZOMBIE (i) ;
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nzombies = nzombies ;
    return (GrB_SUCCESS) ;
}

