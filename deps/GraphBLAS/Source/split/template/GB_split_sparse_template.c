//------------------------------------------------------------------------------
// GB_split_sparse_template: split a single tile from a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A and C, and the slicing of C
    //--------------------------------------------------------------------------

    #ifndef GB_ISO_SPLIT
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    #ifdef GB_JIT_KERNEL
    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
//  int64_t cvlen = C->vlen ;
    int64_t *restrict Ci = C->i ;
    int64_t *restrict Cp = C->p ;
    const int64_t *restrict kfirst_Cslice = C_ek_slicing ;
    const int64_t *restrict klast_Cslice  = C_ek_slicing + C_ntasks ;
    const int64_t *restrict pstart_Cslice = C_ek_slicing + C_ntasks * 2 ;
    #endif

    //--------------------------------------------------------------------------
    // copy the tile from A to C
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < C_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Cslice [tid] ;
        int64_t klast  = klast_Cslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // int64_t jA = GBH_A (Ah, k+akstart) ; not needed 
            int64_t p0 = Cp [k] ;
            GB_GET_PA (pC_start, pC_end, tid, k,
                kfirst, klast, pstart_Cslice, p0, Cp [k+1]) ;
            int64_t pA_offset = Wp [k + akstart] ;
            // copy the vector from A to C
            for (int64_t pC = pC_start ; pC < pC_end ; pC++)
            { 
                // get the index of A(iA,jA)
                int64_t pA = pA_offset + pC - p0 ;
                int64_t iA = Ai [pA] ;
                // shift the index and copy into C(i,j)
                Ci [pC] = iA - aistart ;
                GB_COPY (pC, pA) ;
            }
        }
    }
}

#undef GB_C_TYPE
#undef GB_A_TYPE
#undef GB_ISO_SPLIT

