//------------------------------------------------------------------------------
// GB_convert_s2b_nozombies: convert A from sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse or hypersparse.  Cx and Cb have the same type as A,
// and represent a bitmap format.

{
    //--------------------------------------------------------------------------
    // convert from sparse/hyper to bitmap (no zombies)
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,j) to be operated on by this task
            //------------------------------------------------------------------

            int64_t j = GBh_A (Ah, k) ;
            GB_GET_PA (pA_start, pA_end, tid, k, kfirst, klast, pstart_Aslice,
                GB_IGET (Ap, k), GB_IGET (Ap, k+1)) ;

            // the start of A(:,j) in the new bitmap
            int64_t pC_start = j * avlen ;

            //------------------------------------------------------------------
            // convert A(:,j) from sparse to bitmap
            //------------------------------------------------------------------

            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                // A(i,j) has index i, value Ax [pA]
                int64_t i = GB_IGET (Ai, pA) ;
                int64_t pC = i + pC_start ;
                // move A(i,j) to its new place in the bitmap
                // Cx [pC] = Ax [pA]
                GB_COPY (Cx, pC, Ax, pA) ;
                Cb [pC] = 1 ;
            }
        }
    }
}

