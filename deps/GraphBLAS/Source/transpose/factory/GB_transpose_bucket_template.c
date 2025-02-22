//------------------------------------------------------------------------------
// GB_transpose_bucket_template: transpose and typecast and/or apply operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Symbolic phase for GB_transpose_bucket.

{

    GB_Cp_TYPE *restrict Cp = C->p ;
    int64_t nvec_nonempty ;

    if (nthreads == 1)
    {

        //----------------------------------------------------------------------
        // sequential method: A is not sliced
        //----------------------------------------------------------------------

        // Only requires a single workspace of size avlen for a single thread.
        // The resulting C matrix is not jumbled.
        GBURBLE ("(1-thread bucket transpose) ") ;

        // compute the row counts of A.  No need to scan the A->p pointers
        ASSERT (nworkspaces == 1) ;

        GB_Cp_TYPE *restrict workspace = Workspaces [0] ;
        memset (workspace, 0, (avlen + 1) * sizeof (GB_Cp_TYPE)) ;
        for (int64_t p = 0 ; p < anz ; p++)
        { 
            int64_t i = GB_IGET (Ai, p) ;
            workspace [i]++ ;
        }

        // cumulative sum of the workspace, and copy back into C->p
        GB_cumsum (workspace, Cp_is_32, avlen, &nvec_nonempty, 1, NULL) ;
        memcpy (Cp, workspace, (avlen + 1) * sizeof (GB_Cp_TYPE)) ;

    }
    else if (nworkspaces == 1)
    {

        //----------------------------------------------------------------------
        // atomic method: A is sliced but workspace is shared
        //----------------------------------------------------------------------

        // Only requires a single workspace of size avlen, shared by all
        // threads.  Scales well, but requires atomics.  If the # of rows is
        // very small and the average row degree is high, this can be very slow
        // because of contention on the atomic workspace.  Otherwise, it is
        // typically faster than the non-atomic method.  The resulting C matrix
        // is jumbled.

        GBURBLE ("(%d-thread atomic bucket transpose) ", nthreads) ;

        // compute the row counts of A.  No need to scan the A->p pointers
        GB_Cp_TYPE *restrict workspace = Workspaces [0] ;
        GB_memset (workspace, 0, (avlen + 1) * sizeof (GB_Cp_TYPE), nth) ;

        int64_t p ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < anz ; p++)
        { 
            int64_t i = GB_IGET (Ai, p) ;
            // update workspace [i]++ automically:
            GB_ATOMIC_UPDATE
            workspace [i]++ ;
        }

        C->jumbled = true ; // atomic transpose leaves C jumbled

        // cumulative sum of the workspace, and copy back into C->p
        GB_cumsum (workspace, Cp_is_32, avlen, &nvec_nonempty, nth, Werk) ;
        GB_memcpy (Cp, workspace, (avlen + 1) * sizeof (GB_Cp_TYPE), nth) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // non-atomic method
        //----------------------------------------------------------------------

        // compute the row counts of A for each slice, one per thread; This
        // method is parallel, but not highly scalable.  Each thread requires
        // workspace of size avlen, but no atomics are required.  The resulting
        // C matrix is not jumbled, so this can save work if C needs to be
        // unjumbled later.

        GBURBLE ("(%d-thread non-atomic bucket transpose) ", nthreads) ;

        ASSERT (nworkspaces == nthreads) ;

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            // get the row counts for this slice, of size A->vlen
            GB_Cp_TYPE *restrict workspace = Workspaces [tid] ;
            memset (workspace, 0, (avlen + 1) * sizeof (GB_Cp_TYPE)) ;
            for (int64_t k = A_slice [tid] ; k < A_slice [tid+1] ; k++)
            {
                // iterate over the entries in A(:,j); j not itself not needed
                int64_t pA_start = GB_IGET (Ap, k) ;
                int64_t pA_end = GB_IGET (Ap, k+1) ;
                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                { 
                    // count one more entry in C(i,:) for this slice
                    int64_t i = GB_IGET (Ai, pA) ;
                    workspace [i]++ ;
                }
            }
        }

        // cumulative sum of the workspaces across the slices
        int64_t i ;
        #pragma omp parallel for num_threads(nth) schedule(static)
        for (i = 0 ; i < avlen ; i++)
        {
            GB_Cp_TYPE s = 0 ;
            for (int tid = 0 ; tid < nthreads ; tid++)
            { 
                GB_Cp_TYPE *restrict workspace = Workspaces [tid] ;
                GB_Cp_TYPE c = workspace [i] ;
                workspace [i] = s ;
                s += c ;
            }
            Cp [i] = s ;
        }
        Cp [avlen] = 0 ;

        //----------------------------------------------------------------------
        // compute the vector pointers for C
        //----------------------------------------------------------------------

        GB_cumsum (Cp, Cp_is_32, avlen, &nvec_nonempty, nth, Werk) ;

        //----------------------------------------------------------------------
        // add Cp back to all Workspaces
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nth) schedule(static)
        for (i = 0 ; i < avlen ; i++)
        {
            GB_Cp_TYPE s = Cp [i] ;
            GB_Cp_TYPE *restrict workspace = Workspaces [0] ;
            workspace [i] = s ;
            for (int tid = 1 ; tid < nthreads ; tid++)
            { 
                GB_Cp_TYPE *restrict workspace = Workspaces [tid] ;
                workspace [i] += s ;
            }
        }
    }
    GB_nvec_nonempty_set (C, nvec_nonempty) ;
}

#undef GB_Cp_TYPE

