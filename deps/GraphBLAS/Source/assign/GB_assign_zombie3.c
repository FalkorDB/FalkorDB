//------------------------------------------------------------------------------
// GB_assign_zombie3: delete entries in C(:,j) for C_replace_phase
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: possible: 48 variants, one for each mask type (6: 1, 2,
// 4, 8, 16 bytes and structural), for each matrix type (4: bitmap/full/sparse
// & hyper), mask comp (2).  No variants needed for C.

// For GrB_Row_assign or GrB_Col_assign, C(I,j)<#M,repl>=any must delete all
// entries C(i,j) outside of C(I,j), if the mask M(i,0) (or its complement) is
// zero.  This step is not done for GxB_*_subassign, since that method does not
// modify anything outside IxJ.

// GB_assign_zombie3 and GB_assign_zombie4 are transposes of each other.

// C must be sparse or hypersparse.
// M can have any sparsity structure: hypersparse, sparse, bitmap, or full,
// and is always present.

// C->iso is not affected.

#include "assign/GB_assign.h"
#include "assign/GB_assign_zombie.h"
#include "assign/GB_subassign_methods.h"
#define GB_GENERIC
#define GB_SCALAR_ASSIGN 0
#include "assign/include/GB_assign_shared_definitions.h"

GrB_Info GB_assign_zombie3
(
    GrB_Matrix C,                   // the matrix C, or a copy
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const int64_t j,                // vector index with entries to delete
    const void *I,
    const bool I_is_32,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3]
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
    ASSERT (M != NULL) ;
    ASSERT (!GB_ZOMBIES (M)) ; 
    ASSERT (!GB_JUMBLED (M)) ;      // binary search on M
    ASSERT (!GB_PENDING (M)) ; 
    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M

    //--------------------------------------------------------------------------
    // get C (:,j)
    //--------------------------------------------------------------------------

    GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, const) ; GB_Ch_PTR (Ch, C) ;
    GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;
    int64_t pC_start, pC_end ;
    const int64_t Cnvec = C->nvec ;
    const bool Cp_is_32 = C->p_is_32 ;
    const bool Cj_is_32 = C->j_is_32 ;
    const bool Ci_is_32 = C->i_is_32 ;

    if (Ch != NULL)
    { 
        // C is hypersparse
        const void *C_Yp = (C->Y == NULL) ? NULL : C->Y->p ;
        const void *C_Yi = (C->Y == NULL) ? NULL : C->Y->i ;
        const void *C_Yx = (C->Y == NULL) ? NULL : C->Y->x ;
        const int64_t C_hash_bits = (C->Y == NULL) ? 0 : (C->Y->vdim - 1) ;
        GB_hyper_hash_lookup (Cp_is_32, Cj_is_32,
            Ch, Cnvec, Cp, C_Yp, C_Yi, C_Yx, C_hash_bits,
            j, &pC_start, &pC_end) ;
    }
    else
    { 
        // C is sparse
        pC_start = GB_IGET (Cp, j) ;
        pC_end   = GB_IGET (Cp, j+1) ;
    }

    int64_t nzombies = C->nzombies ;
    const int64_t zjnz = pC_end - pC_start ;

    //--------------------------------------------------------------------------
    // get M(:,0)
    //--------------------------------------------------------------------------

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mi_DECLARE (Mi, const) ; GB_Mi_PTR (Mi, M) ;
    const int8_t *restrict Mb = M->b ;
    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
    const size_t msize = M->type->size ;
    const int64_t Mvlen = M->vlen ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool Mp_is_32 = M->p_is_32 ;
    const bool Mj_is_32 = M->j_is_32 ;
    const bool Mi_is_32 = M->i_is_32 ;

    int64_t pM_start = 0 ; // Mp [0]
    int64_t pM_end = GBp_M (Mp, 1, Mvlen) ;
    const bool mjdense = (pM_end - pM_start) == Mvlen ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (zjnz, chunk, nthreads_max) ;
    int ntasks = (nthreads == 1) ? 1 : (64 * nthreads) ;

    //--------------------------------------------------------------------------
    // delete entries from C(:,j) that are outside I, if the mask M allows it
    //--------------------------------------------------------------------------

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {
        int64_t p1, p2 ;
        GB_PARTITION (p1, p2, zjnz, taskid, ntasks) ;
        for (int64_t pC = pC_start + p1 ; pC < pC_start + p2 ; pC++)
        {

            //------------------------------------------------------------------
            // get C(i,j)
            //------------------------------------------------------------------

            int64_t i = GB_IGET (Ci, pC) ;
            if (!GB_IS_ZOMBIE (i))
            {

                //--------------------------------------------------------------
                // C(i,j) is outside C(I,j) if i is not in the list I
                //--------------------------------------------------------------

                bool i_outside = !GB_ij_is_in_list (I, I_is_32, nI, i, Ikind,
                    Icolon) ;
                if (i_outside)
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
    // return result
    //--------------------------------------------------------------------------

    C->nzombies = nzombies ;
    return (GrB_SUCCESS) ;
}

