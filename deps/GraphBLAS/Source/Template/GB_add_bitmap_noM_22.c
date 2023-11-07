//------------------------------------------------------------------------------
// GB_add_bitmap_noM_22: C=A+B, C bitmap, A bitmap, B sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.
// A is bitmap.  B is sparse/hyper.

{

    //--------------------------------------------------------------------------
    // Method22: C and A are bitmap; B is sparse or hypersparse
    //--------------------------------------------------------------------------

    #ifdef GB_ISO_ADD
        GB_memcpy (Cb, Ab, cnz, C_nthreads) ;
    #else
        int64_t p ;
        #pragma omp parallel for num_threads(C_nthreads) schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            int8_t a = Ab [p] ;
            if (a)
            { 
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = A(i,j) + beta
                    GB_LOAD_A (aij, Ax, p, A_iso) ;
                    GB_EWISEOP (Cx, p, aij, beta_scalar, p % vlen, p / vlen) ;
                }
                #else
                { 
                    // C (i,j) = A (i,j)
                    GB_COPY_A_to_C (Cx, p, Ax, p, A_iso) ;
                }
                #endif
            }
            Cb [p] = a ;
        }
    #endif
    cnvals = A->nvals ;

    const int64_t *kfirst_Bslice = B_ek_slicing ;
    const int64_t *klast_Bslice  = B_ek_slicing + B_ntasks ;
    const int64_t *pstart_Bslice = B_ek_slicing + B_ntasks*2 ;

    #pragma omp parallel for num_threads(B_nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (taskid = 0 ; taskid < B_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Bslice [taskid] ;
        int64_t klast  = klast_Bslice  [taskid] ;
        int64_t task_cnvals = 0 ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // find the part of B(:,k) for this task
            int64_t j = GBH_B (Bh, k) ;
            GB_GET_PA (pB_start, pB_end, taskid, k, kfirst, klast,
                pstart_Bslice, GBP_B (Bp, k, vlen), GBP_B (Bp, k+1, vlen)) ;
            int64_t pC_start = j * vlen ;
            // traverse over B(:,j), the kth vector of B
            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
            {
                int64_t i = Bi [pB] ;
                int64_t p = pC_start + i ;
                if (Cb [p])
                { 
                    // C (i,j) = A (i,j) + B (i,j)
                    #ifndef GB_ISO_ADD
                    GB_LOAD_A (aij, Ax, p , A_iso) ;
                    GB_LOAD_B (bij, Bx, pB, B_iso) ;
                    GB_EWISEOP (Cx, p, aij, bij, i, j) ;
                    #endif
                }
                else
                { 
                    #ifndef GB_ISO_ADD
                    #if GB_IS_EWISEUNION
                    { 
                        // C (i,j) = alpha + B(i,j)
                        GB_LOAD_B (bij, Bx, pB, B_iso) ;
                        GB_EWISEOP (Cx, p, alpha_scalar, bij, i, j) ;
                    }
                    #else
                    { 
                        // C (i,j) = B (i,j)
                        GB_COPY_B_to_C (Cx, p, Bx, pB, B_iso) ;
                    }
                    #endif
                    #endif
                    Cb [p] = 1 ;
                    task_cnvals++ ;
                }
            }
        }
        cnvals += task_cnvals ;
    }
}

