//------------------------------------------------------------------------------
// GB_assign_zombie4: delete entries in C(i,:) for C_replace_phase
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,         // if true, use the only structure of M
    const int64_t i,
    const void *J,
    const bool J_is_32,
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

    GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, const) ; GB_Ch_PTR (Ch, C) ;
    GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;
    const int64_t Cnvec = C->nvec ;
    int64_t nzombies = C->nzombies ;
    const bool Ci_is_32 = C->i_is_32 ;

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    const void *Mh = M->h ;
    const int8_t *restrict Mb = M->b ;
    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
    const size_t msize = M->type->size ;
    const int64_t Mnvec = M->nvec ;
    ASSERT (M->vlen == 1) ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full = GB_IS_FULL (M) ;
    const void *M_Yp = (M->Y == NULL) ? NULL : M->Y->p ;
    const void *M_Yi = (M->Y == NULL) ? NULL : M->Y->i ;
    const void *M_Yx = (M->Y == NULL) ? NULL : M->Y->x ;
    const bool Mp_is_32 = M->p_is_32 ;
    const bool Mj_is_32 = M->j_is_32 ;
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

            int64_t j = GBh_C (Ch, k) ;
            bool j_outside = !GB_ij_is_in_list (J, J_is_32, nJ, j, Jkind,
                Jcolon) ;
            if (j_outside)
            {

                //--------------------------------------------------------------
                // j is not in J; find C(i,j)
                //--------------------------------------------------------------

                int64_t pC = GB_IGET (Cp, k) ;
                int64_t pC_end = GB_IGET (Cp, k+1) ;
                int64_t pright = pC_end - 1 ;
                bool is_zombie ;
                bool found = GB_binary_search_zombie (i, Ci, Ci_is_32,
                    &pC, &pright, true, &is_zombie) ;

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
                        mij = GBb_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                    }
                    else
                    {
                        // M is sparse or hypersparse
                        int64_t pM, pM_end ;

                        if (M_is_hyper)
                        { 
                            // M is hypersparse
                            GB_hyper_hash_lookup (Mp_is_32, Mj_is_32,
                                Mh, Mnvec, Mp, M_Yp, M_Yi, M_Yx, M_hash_bits,
                                j, &pM, &pM_end) ;
                        }
                        else
                        { 
                            // M is sparse
                            pM     = GB_IGET (Mp, j) ;
                            pM_end = GB_IGET (Mp, j+1) ;
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
                        int64_t iC = GB_ZOMBIE (i) ;
                        GB_ISET (Ci, pC, iC) ;      // Ci [pC] = iC
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

