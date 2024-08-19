//------------------------------------------------------------------------------
// GB_select_phase2: C=select(A,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse or hypersparse, but it is present only as Cp, Ci, and Cx.
// A is never bitmap.  It is sparse or hypersparse in most cases.  It can also
// by full for DIAG.

{
    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int64_t  *restrict Ap = A->p ;
    const int64_t  *restrict Ah = A->h ;
    const int64_t  *restrict Ai = A->i ;
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    size_t asize = A->type->size ;
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;
    // if A is bitmap, the bitmap selector is always used instead
    ASSERT (!GB_IS_BITMAP (A)) ;
    #ifndef GB_DIAG_SELECTOR
    // if A is full, all opcodes except DIAG use the bitmap selector instead
    ASSERT (!GB_IS_FULL (A)) ;
    #endif

    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // C = select (A)
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {

        // if kfirst > klast then task tid does no work at all
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;

        //----------------------------------------------------------------------
        // selection from vectors kfirst to klast
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,k) to be operated on by this task
            //------------------------------------------------------------------

            GB_GET_PA_AND_PC (pA_start, pA_end, pC, tid, k, kfirst, klast,
                pstart_Aslice, Cp_kfirst,
                GBP_A (Ap, k, avlen), GBP_A (Ap, k+1, avlen), Cp [k]) ;

            //------------------------------------------------------------------
            // compact Ai and Ax [pA_start ... pA_end-1] into Ci and Cx
            //------------------------------------------------------------------

            #if defined ( GB_ENTRY_SELECTOR )

                int64_t j = GBH_A (Ah, k) ;
                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                {
                    // A is sparse or hypersparse
                    ASSERT (Ai != NULL) ;
                    int64_t i = Ai [pA] ;
                    GB_TEST_VALUE_OF_ENTRY (keep, pA) ;
                    if (keep)
                    { 
                        ASSERT (pC >= Cp [k] && pC < Cp [k+1]) ;
                        Ci [pC] = i ;
                        // Cx [pC] = Ax [pA] ;
                        GB_SELECT_ENTRY (Cx, pC, Ax, pA) ;
                        pC++ ;
                    }
                }

            #elif defined ( GB_TRIL_SELECTOR  ) || \
                  defined ( GB_ROWGT_SELECTOR )

                // keep Zp [k] to pA_end-1
                int64_t p = GB_IMAX (Zp [k], pA_start) ;
                int64_t mynz = pA_end - p ;
                if (mynz > 0)
                { 
                    // A and C are both sparse or hypersparse
                    ASSERT (pA_start <= p && p + mynz <= pA_end) ;
                    ASSERT (pC >= Cp [k] && pC + mynz <= Cp [k+1]) ;
                    ASSERT (Ai != NULL) ;
                    memcpy (Ci +pC, Ai +p, mynz*sizeof (int64_t)) ;
                    #if !GB_ISO_SELECT
                    memcpy (Cx +pC*asize, Ax +p*asize, mynz*asize) ;
                    #endif
                }

            #elif defined ( GB_TRIU_SELECTOR  ) || \
                  defined ( GB_ROWLE_SELECTOR )

                // keep pA_start to Zp[k]-1
                int64_t p = GB_IMIN (Zp [k], pA_end) ;
                int64_t mynz = p - pA_start ;
                if (mynz > 0)
                { 
                    // A and C are both sparse or hypersparse
                    ASSERT (pC >= Cp [k] && pC + mynz <= Cp [k+1]) ;
                    ASSERT (Ai != NULL) ;
                    memcpy (Ci +pC, Ai +pA_start, mynz*sizeof (int64_t)) ;
                    #if !GB_ISO_SELECT
                    memcpy (Cx +pC*asize, Ax +pA_start*asize, mynz*asize) ;
                    #endif
                }

            #elif defined ( GB_DIAG_SELECTOR )

                // task that owns the diagonal entry does this work
                // A can be sparse, hypersparse, or full, but not bitmap
                int64_t p = Zp [k] ;
                if (pA_start <= p && p < pA_end)
                { 
                    ASSERT (pC >= Cp [k] && pC + 1 <= Cp [k+1]) ;
                    Ci [pC] = GBI_A (Ai, p, avlen) ;
                    #if !GB_ISO_SELECT
                    memcpy (Cx +pC*asize, Ax +p*asize, asize) ;
                    #endif
                }

            #elif defined ( GB_OFFDIAG_SELECTOR  ) || \
                  defined ( GB_ROWINDEX_SELECTOR )

                // keep pA_start to Zp[k]-1
                int64_t p = GB_IMIN (Zp [k], pA_end) ;
                int64_t mynz = p - pA_start ;
                if (mynz > 0)
                { 
                    // A and C are both sparse or hypersparse
                    ASSERT (pC >= Cp [k] && pC + mynz <= Cp [k+1]) ;
                    ASSERT (Ai != NULL) ;
                    memcpy (Ci +pC, Ai +pA_start, mynz*sizeof (int64_t)) ;
                    #if !GB_ISO_SELECT
                    memcpy (Cx +pC*asize, Ax +pA_start*asize, mynz*asize) ;
                    #endif
                    pC += mynz ;
                }

                // keep Zp[k]+1 to pA_end-1
                p = GB_IMAX (Zp [k]+1, pA_start) ;
                mynz = pA_end - p ;
                if (mynz > 0)
                { 
                    // A and C are both sparse or hypersparse
                    ASSERT (pA_start <= p && p < pA_end) ;
                    ASSERT (pC >= Cp [k] && pC + mynz <= Cp [k+1]) ;
                    ASSERT (Ai != NULL) ;
                    memcpy (Ci +pC, Ai +p, mynz*sizeof (int64_t)) ;
                    #if !GB_ISO_SELECT
                    memcpy (Cx +pC*asize, Ax +p*asize, mynz*asize) ;
                    #endif
                }

            #endif
        }
    }
}

#undef GB_TRIL_SELECTOR
#undef GB_TRIU_SELECTOR
#undef GB_DIAG_SELECTOR
#undef GB_OFFDIAG_SELECTOR
#undef GB_ROWINDEX_SELECTOR
#undef GB_ROWLE_SELECTOR
#undef GB_ROWGT_SELECTOR

