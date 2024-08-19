//------------------------------------------------------------------------------
// GB_bitmap_assign_A_template: traverse over A for bitmap assignment into C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This template traverses over all the entries of the matrix A and operates on
// the corresponding entry in C(i,j), using the GB_AIJ_WORK macro.  A can be
// hypersparse, sparse, bitmap, or full.  It is not a scalar.  The matrix
// C must be bitmap or full.

{

    //--------------------------------------------------------------------------
    // matrix assignment: slice the entries of A for each task
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    int A_ntasks, A_nthreads ;
    GB_A_NHELD (A_nnz_held) ;
    GB_SLICE_MATRIX_WORK (A, 8, A_nnz_held + A->nvec, A_nnz_held) ;

    //--------------------------------------------------------------------------
    // traverse of the entries of the matrix A
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {

        // if kfirst > klast then task tid does no work at all
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;
        int64_t task_cnvals = 0 ;

        //----------------------------------------------------------------------
        // traverse over A (:,kfirst:klast)
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,k) for this task
            //------------------------------------------------------------------

            int64_t jA = GBH_A (Ah, k) ;
            int64_t pA_start, pA_end ;
            GB_get_pA (&pA_start, &pA_end, tid, k, kfirst,
                klast, pstart_Aslice, Ap, nI) ;

            //------------------------------------------------------------------
            // traverse over A(:,jA), the kth vector of A
            //------------------------------------------------------------------

            int64_t jC = GB_ijlist (J, jA, Jkind, Jcolon) ;
            int64_t pC0 = jC * cvlen ;      // first entry in C(:,jC)

            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                if (!GBB_A (Ab, pA)) continue ;
                int64_t iA = GBI_A (Ai, pA, nI) ;
                int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                int64_t pC = iC + pC0 ;
                // operate on C(iC,jC) at pC, and A(iA,jA) at pA.  The mask
                // can be accessed at pC if M is bitmap or full.  A has any
                // sparsity format so only A(iA,jA) can be accessed at pA.
                // To access a full matrix M for the subassign case, use
                // the position (iA + jA*nI).
                GB_AIJ_WORK (pC, pA) ;
            }
        }
        cnvals += task_cnvals ;
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    GB_WERK_POP (A_ek_slicing, int64_t) ;
}

