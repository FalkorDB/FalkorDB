//------------------------------------------------------------------------------
// GB_convert_s2b_nozombies: convert A from sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse or hypersparse.  Axnew and Ab have the same type as A,
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

            int64_t j = GBH_A (Ah, k) ;
            GB_GET_PA (pA_start, pA_end, tid, k,
                kfirst, klast, pstart_Aslice, Ap [k], Ap [k+1]) ;

            // the start of A(:,j) in the new bitmap
            int64_t pA_new = j * avlen ;

            //------------------------------------------------------------------
            // convert A(:,j) from sparse to bitmap
            //------------------------------------------------------------------

            for (int64_t p = pA_start ; p < pA_end ; p++)
            { 
                // A(i,j) has index i, value Ax [p]
                int64_t i = Ai [p] ;
                int64_t pnew = i + pA_new ;
                // move A(i,j) to its new place in the bitmap
                // Axnew [pnew] = Ax [p]
                GB_COPY (Axnew, pnew, Ax, p) ;
                Ab [pnew] = 1 ;
            }
        }
    }
}

