//------------------------------------------------------------------------------
// GB_assign_zombie5: delete entries in C for C_replace_phase
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: possible: 96 variants. Could use one for each mask type (6: 1, 2,
// 4, 8, 16 bytes and structural), for each matrix type (4: bitmap/full/sparse/
// hyper), mask comp (2), C sparsity (2: sparse/hyper): 6*4*2*2 = 96 variants,
// so a JIT kernel is reasonable.

// For GrB_Matrix_assign, C(I,J)<M,repl>=..., if C_replace is true, and mask M
// is present, then any entry C(i,j) outside IxJ must be be deleted, if
// M(i,j)=0.

// See also GB_assign_zombie3 and GB_assign_zombie4.

// C must be sparse or hypersparse.

// C->iso is not affected.

#include "assign/GB_assign.h"
#include "assign/GB_assign_zombie.h"
#include "assign/GB_subassign_methods.h"
#include "slice/GB_ek_slice.h"
#define GB_GENERIC
#define GB_SCALAR_ASSIGN 0
#include "assign/include/GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                         \
{                                           \
    GB_WERK_POP (C_ek_slicing, int64_t) ;   \
}

GrB_Info GB_assign_zombie5
(
    GrB_Matrix C,                   // the matrix C, or a copy
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    GB_Werk Werk
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
    ASSERT (!GB_ZOMBIES (M)) ; 
    ASSERT (!GB_JUMBLED (M)) ;      // binary search on M
    ASSERT (!GB_PENDING (M)) ; 
    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    const int64_t *restrict Ch = C->h ;
    const int64_t *restrict Cp = C->p ;
    // const int64_t Cnvec = C->nvec ;
    int64_t *restrict Ci = C->i ;
    int64_t nzombies = C->nzombies ;
    const int64_t zvlen = C->vlen ;

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    const int64_t *restrict Mp = M->p ;
    const int64_t *restrict Mh = M->h ;
    const int8_t  *restrict Mb = M->b ;
    const int64_t *restrict Mi = M->i ;
    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
    const size_t msize = M->type->size ;
    const int64_t Mnvec = M->nvec ;
    const int64_t Mvlen = M->vlen ;
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

    //--------------------------------------------------------------------------
    // slice the entries for each task
    //--------------------------------------------------------------------------

    int C_ntasks, C_nthreads ;
    GB_WERK_DECLARE (C_ek_slicing, int64_t) ;
    GB_SLICE_MATRIX (C, 64) ;

    //--------------------------------------------------------------------------
    // each task creates its own zombies
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(dynamic,1) \
        reduction(+:nzombies)
    for (tid = 0 ; tid < C_ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task description
        //----------------------------------------------------------------------

        int64_t kfirst = kfirst_Cslice [tid] ;
        int64_t klast  = klast_Cslice  [tid] ;

        //----------------------------------------------------------------------
        // scan vectors kfirst to klast for entries to delete
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get C(:,j) and determine if j is outside the list J
            //------------------------------------------------------------------

            int64_t j = GBH_C (Ch, k) ;
            // j_outside is true if column j is outside the C(I,J) submatrix
            bool j_outside = !GB_ij_is_in_list (J, nJ, j, Jkind, Jcolon) ;
            GB_GET_PA (pC_start, pC_end, tid, k, kfirst, klast, pstart_Cslice,
                Cp [k], Cp [k+1]) ;

            //------------------------------------------------------------------
            // get M(:,j)
            //------------------------------------------------------------------

            int64_t pM_start, pM_end ;
            GB_LOOKUP_VECTOR_M (j, pM_start, pM_end) ;

            bool mjdense = (pM_end - pM_start) == Mvlen ;

            //------------------------------------------------------------------
            // iterate over all entries in C(:,j)
            //------------------------------------------------------------------

            for (int64_t pC = pC_start ; pC < pC_end ; pC++)
            {

                //--------------------------------------------------------------
                // consider C(i,j)
                //--------------------------------------------------------------

                // C(i,j) is outside the C(I,J) submatrix if either i is
                // not in the list I, or j is not in J, or both.
                int64_t i = Ci [pC] ;
                if (!GB_IS_ZOMBIE (i) &&
                    (j_outside || !GB_ij_is_in_list (I, nI, i, Ikind, Icolon)))
                {

                    //----------------------------------------------------------
                    // C(i,j) is a live entry not in the C(I,J) submatrix
                    //----------------------------------------------------------

                    // Check the mask M to see if it should be deleted.
                    GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (i) ;
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
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->nzombies = nzombies ;
    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

