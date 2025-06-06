//------------------------------------------------------------------------------
// GB_subassign_25_template: C<M> = A where C is empty and A is dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 25: C(:,:)<M,s> = A ; C is empty, M structural, A bitmap/as-if-full

// M:           present
// Mask_comp:   false
// Mask_struct: true
// C_replace:   effectively false (not relevant since C is empty)
// accum:       NULL
// A:           matrix
// S:           none

// C and M are sparse or hypersparse.  A can have any sparsity structure, even
// bitmap, but it must either be bitmap, or as-if-full.  M may be jumbled.  If
// so, C is constructed as jumbled.  C is reconstructed with the same structure
// as M and can have any sparsity structure on input.  The only constraint on C
// is nnz(C) is zero on input.

// C is iso if A is iso

// C<M> = A where C starts as empty, M is structural, and A is dense.  The
// pattern of C is an exact copy of M.  A is full, dense, or bitmap.
// M is sparse or hypersparse, and C is constructed with the same pattern as M.

#undef  GB_FREE_ALL
#define GB_FREE_ALL                         \
{                                           \
    GB_WERK_POP (M_ek_slicing, int64_t) ;   \
}

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_iso = A->iso ;
    ASSERT (GB_IS_FULL (A) || GB_A_IS_BITMAP) ;

    //--------------------------------------------------------------------------
    // Parallel: slice M into equal-sized chunks
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;
    int M_nthreads, M_ntasks ;
    GB_M_NHELD (M_nnz_held) ;
    GB_SLICE_MATRIX_WORK (M, 8, M_nnz_held + M->nvec, M_nnz_held) ;

    //--------------------------------------------------------------------------
    // get C, M, and A
    //--------------------------------------------------------------------------

    GB_Ci_DECLARE (Ci, ) ; GB_Ci_PTR (Ci, C) ;

    ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mh_DECLARE (Mh, const) ; GB_Mh_PTR (Mh, M) ;
    GB_Mi_DECLARE (Mi, const) ; GB_Mi_PTR (Mi, M) ;
    const int64_t Mvlen = M->vlen ;
    const int8_t *restrict Ab = A->b ;
    const int64_t avlen = A->vlen ;

    bool C_iso = C->iso ;
    ASSERT (C->iso == A->iso) ;

    #ifdef GB_ISO_ASSIGN
    ASSERT (C->iso) ;
    #else
    ASSERT (!C->iso) ;
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    GB_DECLAREC (cwork) ;
    if (GB_A_ISO)
    {
        // get the iso value of A and typecast to C->type
        // cwork = (ctype) Ax [0]
        // This is no longer used.  If A is iso, so is C, and in that case,
        // GB_ISO_ASSIGN is true and cwork is not used here.
        ASSERT (GB_DEAD_CODE) ;
        GB_COPY_aij_to_cwork (cwork, Ax, 0, true) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // C<M> = A
    //--------------------------------------------------------------------------

    if (GB_A_IS_BITMAP)
    {

        //----------------------------------------------------------------------
        // A is bitmap, so zombies can be created in C
        //----------------------------------------------------------------------

        int64_t nzombies = 0 ;

        int tid ;
        #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1) \
            reduction(+:nzombies)
        for (tid = 0 ; tid < M_ntasks ; tid++)
        {

            // if kfirst > klast then task tid does no work at all
            int64_t kfirst = kfirst_Mslice [tid] ;
            int64_t klast  = klast_Mslice  [tid] ;
            int64_t task_nzombies = 0 ;

            //------------------------------------------------------------------
            // C<M(:,kfirst:klast)> = A(:,kfirst:klast)
            //------------------------------------------------------------------

            for (int64_t k = kfirst ; k <= klast ; k++)
            {

                //--------------------------------------------------------------
                // find the part of M(:,k) to be operated on by this task
                //--------------------------------------------------------------

                int64_t j = GBh_M (Mh, k) ;
                GB_GET_PA (pM_start, pM_end, tid, k, kfirst, klast,
                    pstart_Mslice, GB_IGET (Mp, k), GB_IGET (Mp, k+1)) ;

                //--------------------------------------------------------------
                // C<M(:,j)> = A(:,j)
                //--------------------------------------------------------------

                // M is hypersparse or sparse.  C is the same as M.
                // pA points to the start of A(:,j) since A is dense
                int64_t pA = j * avlen ;
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    int64_t i = GB_IGET (Mi, pM) ;
                    int64_t p = pA + i ;
                    if (Ab [p])
                    { 
                        // C(i,j) = A(i,j)
                        #ifndef GB_ISO_ASSIGN
                        GB_COPY_aij_to_C (Cx, pM, Ax, p, GB_A_ISO, cwork,
                            GB_C_ISO) ;
                        #endif
                    }
                    else
                    { 
                        // C(i,j) becomes a zombie
                        task_nzombies++ ;
                        i = GB_ZOMBIE (i) ;
                        GB_ISET (Ci, pM, i) ;   // Ci [pM] = i
                    }
                }
            }
            nzombies += task_nzombies ;
        }
        C->nzombies = nzombies ;

    }
    else
    {

        //----------------------------------------------------------------------
        // A is full, so no zombies will appear in C
        //----------------------------------------------------------------------

        #ifndef GB_ISO_ASSIGN
        {

            int tid ;
            #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
            for (tid = 0 ; tid < M_ntasks ; tid++)
            {

                // if kfirst > klast then task tid does no work at all
                int64_t kfirst = kfirst_Mslice [tid] ;
                int64_t klast  = klast_Mslice  [tid] ;

                //--------------------------------------------------------------
                // C<M(:,kfirst:klast)> = A(:,kfirst:klast)
                //--------------------------------------------------------------

                for (int64_t k = kfirst ; k <= klast ; k++)
                {

                    //----------------------------------------------------------
                    // find the part of M(:,k) to be operated on by this task
                    //----------------------------------------------------------

                    int64_t j = GBh_M (Mh, k) ;
                    GB_GET_PA (pM_start, pM_end, tid, k, kfirst, klast,
                        pstart_Mslice, GB_IGET (Mp, k), GB_IGET (Mp, k+1)) ;

                    //----------------------------------------------------------
                    // C<M(:,j)> = A(:,j)
                    //----------------------------------------------------------

                    // M is hypersparse or sparse.  C is the same as M.
                    // pA points to the start of A(:,j) since A is dense
                    int64_t pA = j * avlen ;
                    GB_PRAGMA_SIMD_VECTORIZE
                    for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                    { 
                        // C(i,j) = A(i,j)
                        int64_t p = pA + GB_IGET (Mi, pM) ;
                        GB_COPY_aij_to_C (Cx, pM, Ax, p,
                            GB_A_ISO, cwork, GB_C_ISO) ;
                    }
                }
            }
        }
        #endif
    }

    GB_FREE_ALL ;
}

#undef GB_ISO_ASSIGN
#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

