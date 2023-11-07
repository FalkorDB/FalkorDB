//------------------------------------------------------------------------------
// GB_concat_sparse_template: concatenate a tile into a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The tile A is hypersparse, sparse, or full, not bitmap.  If C is iso, then
// so is A, and the values are not copied here.

{

    //--------------------------------------------------------------------------
    // get C and the tile A
    //--------------------------------------------------------------------------

    #ifndef GB_ISO_CONCAT
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    #ifdef GB_JIT_KERNEL
    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    int64_t avlen = A->vlen ;
    int64_t *restrict Ci = C->i ;
    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;
    #endif

    //--------------------------------------------------------------------------
    // copy the tile A into C
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(static)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            int64_t j = GBH_A (Ah, k) ;
            const int64_t pC_start = W [j] ;

            //------------------------------------------------------------------
            // find the part of the kth vector A(:,j) for this task
            //------------------------------------------------------------------

            int64_t pA_start, pA_end ;
            // as done by GB_get_pA, but also get p0 = Ap [k]
            const int64_t p0 = GBP_A (Ap, k, avlen) ;
            const int64_t p1 = GBP_A (Ap, k+1, avlen) ;
            if (k == kfirst)
            { 
                // First vector for task tid; may only be partially owned.
                pA_start = pstart_Aslice [tid] ;
                pA_end   = GB_IMIN (p1, pstart_Aslice [tid+1]) ;
            }
            else if (k == klast)
            { 
                // Last vector for task tid; may only be partially owned.
                pA_start = p0 ;
                pA_end   = pstart_Aslice [tid+1] ;
            }
            else
            { 
                // task tid entirely owns this vector A(:,k).
                pA_start = p0 ;
                pA_end   = p1 ;
            }

            //------------------------------------------------------------------
            // append A(:,j) onto C(:,j)
            //------------------------------------------------------------------

            GB_PRAGMA_SIMD
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                int64_t i = GBI_A (Ai, pA, avlen) ;       // i = Ai [pA]
                int64_t pC = pC_start + pA - p0 ;
                Ci [pC] = cistart + i ;
                // Cx [pC] = Ax [pA] ;
                GB_COPY (pC, pA, A_iso) ;
            }
        }
    }
}

#undef GB_C_TYPE
#undef GB_A_TYPE
#undef GB_ISO_CONCAT

