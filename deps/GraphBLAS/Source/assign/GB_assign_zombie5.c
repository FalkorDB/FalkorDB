//------------------------------------------------------------------------------
// GB_assign_zombie5: delete entries in C for C_replace_phase
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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
    const void *I,
    const bool I_is_32,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,
    const bool J_is_32,
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

    GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, const) ; GB_Ch_PTR (Ch, C) ;
    GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;
    int64_t nzombies = C->nzombies ;
    const int64_t zvlen = C->vlen ;

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mh_DECLARE (Mh, const) ; GB_Mh_PTR (Mh, M) ;
    GB_Mi_DECLARE (Mi, const) ; GB_Mi_PTR (Mi, M) ;
    const int8_t  *restrict Mb = M->b ;
    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
    const size_t msize = M->type->size ;
    const int64_t Mnvec = M->nvec ;
    const int64_t Mvlen = M->vlen ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full = GB_IS_FULL (M) ;
    const void *M_Yp = (M->Y == NULL) ? NULL : M->Y->p ;
    const void *M_Yi = (M->Y == NULL) ? NULL : M->Y->i ;
    const void *M_Yx = (M->Y == NULL) ? NULL : M->Y->x ;
    const bool Mp_is_32 = M->p_is_32 ;
    const bool Mj_is_32 = M->j_is_32 ;
    const bool Mi_is_32 = M->i_is_32 ;
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

            int64_t j = GBh_C (Ch, k) ;
            // j_outside is true if column j is outside the C(I,J) submatrix
            bool j_outside = !GB_ij_is_in_list (J, J_is_32, nJ, j, Jkind,
                Jcolon) ;
            GB_GET_PA (pC_start, pC_end, tid, k, kfirst, klast, pstart_Cslice,
                GB_IGET (Cp, k), GB_IGET (Cp, k+1)) ;

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
                int64_t i = GB_IGET (Ci, pC) ;
                if (!GB_IS_ZOMBIE (i) && (j_outside ||
                    !GB_ij_is_in_list (I, I_is_32, nI, i, Ikind, Icolon)))
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
                        i = GB_ZOMBIE (i) ;
                        GB_ISET (Ci, pC, i) ;   // Ci [pC] = i ;
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

