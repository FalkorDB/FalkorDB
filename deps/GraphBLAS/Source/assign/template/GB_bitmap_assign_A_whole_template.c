//------------------------------------------------------------------------------
// GB_bitmap_assign_A_whole_template: traverse A for bitmap assignment into C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This template traverses over all the entries of the matrix A and operates on
// the corresponding entry in C(i,j), using the GB_AIJ_WORK macro.  A can be
// hypersparse or sparse (not bitmap or full).  It is not a scalar.  The matrix
// C must be bitmap or full.
//
// The workspace must already be declared as follows:
//
//      GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
//      int A_ntasks = 0, A_nthreads = 0 ;
//
// The workspace is allocated and tasks are computed, if not already done.
// It is not freed, so it can be used for subsequent uses of this template.
// To free the workspace, the method that uses this template must do:
//
//      GB_WERK_POP (A_ek_slicing, int64_t) ;

{

    //--------------------------------------------------------------------------
    // slice the matrix A
    //--------------------------------------------------------------------------

    if (A_ek_slicing == NULL)
    { 
        GB_A_NHELD (A_nnz_held) ;
        GB_SLICE_MATRIX_WORK (A, 8, A_nnz_held + A->nvec, A_nnz_held) ;
    }
    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // traverse the entries of the matrix A
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

            int64_t j = GBH_A (Ah, k) ;
            GB_GET_PA (pA_start, pA_end, tid, k, kfirst, klast, pstart_Aslice,
                GBP_A (Ap, k, Avlen), GBP_A (Ap, k+1, Avlen)) ;

            //------------------------------------------------------------------
            // traverse over A(:,j), the kth vector of A
            //------------------------------------------------------------------

            int64_t pC0 = j * Cvlen ;      // first entry in C(:,j)

            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                int64_t i = Ai [pA] ;
                int64_t pC = i + pC0 ;
                // operate on C(i,j) at pC, and A(i,j) at pA.  The mask
                // can be accessed at pC if M is bitmap or full.  A has any
                // sparsity format so only A(i,j) can be accessed at pA.
                GB_AIJ_WORK (pC, pA) ;
            }
        }
        #ifndef GB_NO_CNVALS
        cnvals += task_cnvals ;
        #endif
    }
}

