//------------------------------------------------------------------------------
// GB_subassign_05d_template: C<M> = x where C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 05d: C(:,:)<M> = scalar ; no S, C is dense

// M:           present, can be sparse, hypersparse, bitmap, or full
// Mask_comp:   false
// Mask_struct: true or false
// C_replace:   false
// accum:       NULL
// A:           scalar
// S:           none

// C can have any sparsity structure, but it must be entirely dense with
// all entries present.

#undef  GB_FREE_ALL
#define GB_FREE_ALL                         \
{                                           \
    GB_WERK_POP (M_ek_slicing, int64_t) ;   \
}

{

    //--------------------------------------------------------------------------
    // Parallel: slice M into equal-sized chunks
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;
    int M_ntasks, M_nthreads ;
    GB_M_NHELD (M_nnz_held) ;
    GB_SLICE_MATRIX_WORK (M, 8, M_nnz_held + M->nvec, M_nnz_held) ;

    //--------------------------------------------------------------------------
    // get C and M
    //--------------------------------------------------------------------------

    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!C->iso) ;

    const int64_t *restrict Mp = M->p ;
    const int8_t  *restrict Mb = M->b ;
    const int64_t *restrict Mh = M->h ;
    const int64_t *restrict Mi = M->i ;
    const GB_M_TYPE *restrict
        Mx = (GB_M_TYPE *) (GB_MASK_STRUCT ? NULL : (M->x)) ;
    const size_t Mvlen = M->vlen ;
    const size_t msize = M->type->size ;

    GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    const int64_t Cvlen = C->vlen ;

    //--------------------------------------------------------------------------
    // C<M> = x
    //--------------------------------------------------------------------------

    int taskid ;
    #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < M_ntasks ; taskid++)
    {

        // if kfirst > klast then taskid does no work at all
        int64_t kfirst = kfirst_Mslice [taskid] ;
        int64_t klast  = klast_Mslice  [taskid] ;

        //----------------------------------------------------------------------
        // C<M(:,kfirst:klast)> = x
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of M(:,k) to be operated on by this task
            //------------------------------------------------------------------

            int64_t j = GBH_M (Mh, k) ;
            GB_GET_PA (pM_start, pM_end, taskid, k,
                kfirst, klast, pstart_Mslice,
                GBP_M (Mp, k, Mvlen), GBP_M (Mp, k+1, Mvlen)) ;

            // pC_start points to the start of C(:,j)
            int64_t pC_start = j * Cvlen ;

            //------------------------------------------------------------------
            // C<M(:,j)> = x
            //------------------------------------------------------------------

            if (Mx == NULL && Mb == NULL)   // FIXME
//          if (GB_MASK_STRUCT && !GB_M_IS_BITMAP)  <--- use this instead
            {
                // mask is structural and not bitmap
                GB_PRAGMA_SIMD_VECTORIZE
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                { 
                    int64_t pC = pC_start + GBI_M (Mi, pM, Mvlen) ;
                    // Cx [pC] = cwork
                    GB_COPY_cwork_to_C (Cx, pC, cwork, false) ;
                }
            }
            else
            {
                GB_PRAGMA_SIMD_VECTORIZE
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    if (GBB_M (Mb, pM) && GB_MCAST (Mx, pM, msize))
                    { 
                        int64_t pC = pC_start + GBI_M (Mi, pM, Mvlen) ;
                        // Cx [pC] = cwork
                        GB_COPY_cwork_to_C (Cx, pC, cwork, false) ;
                    }
                }
            }
        }
    }

    GB_FREE_ALL ;
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

