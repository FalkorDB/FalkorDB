//------------------------------------------------------------------------------
// GB_bitmap_assign_C_whole_template: iterate over a bitmap matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The #include'ing file defines a GB_CIJ_WORK macro for the body of the loop,
// which operates on the entry C(iC,jC) at position Cx [pC] and Cb [pC].  The C
// matrix held in bitmap form.  If the mask matrix is also a bitmap matrix or
// full matrix, the GB_GET_MIJ macro can compute the effective value of the
// mask for the C(iC,jC) entry.

// C must be bitmap or full.  If M is accessed, it must also be bitmap or full.

#ifndef GB_GET_MIJ
#define GB_GET_MIJ(mij,pM) ;
#endif

{
    // iterate over all of C(:,:).
    const int nthreads = GB_nthreads (cnzmax, chunk, nthreads_max) ;
    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < nthreads ; tid++)
    {
        int64_t pC_start, pC_end, task_cnvals = 0 ;
        GB_PARTITION (pC_start, pC_end, cnzmax, tid, nthreads) ;
        for (int64_t pC = pC_start ; pC < pC_end ; pC++)
        { 
            // int64_t iC = pC % cvlen ;
            // int64_t jC = pC / cvlen ;
            GB_GET_MIJ (mij, pC) ;          // mij = Mask (pC)
            GB_CIJ_WORK (pC) ;              // operate on C(iC,jC)
        }
        cnvals += task_cnvals ;
    }
}
