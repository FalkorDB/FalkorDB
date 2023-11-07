//------------------------------------------------------------------------------
// GB_add_bitmap_noM_21: C=A+B, C bitmap, A and B bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.
// A and B are bitmap.

{

    //--------------------------------------------------------------------------
    // Method21: C, A, and B are all bitmap
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < C_nthreads ; tid++)
    {
        int64_t pstart, pend, task_cnvals = 0 ;
        GB_PARTITION (pstart, pend, cnz, tid, C_nthreads) ;
        for (int64_t p = pstart ; p < pend ; p++)
        {
            #ifdef GB_ISO_ADD
            int8_t c = Ab [p] || Bb [p] ;
            #else
            int8_t c = 0 ;
            if (Ab [p] && Bb [p])
            { 
                // C (i,j) = A (i,j) + B (i,j)
                GB_LOAD_A (aij, Ax, p, A_iso) ;
                GB_LOAD_B (bij, Bx, p, B_iso) ;
                GB_EWISEOP (Cx, p, aij, bij, p % vlen, p / vlen) ;
                c = 1 ;
            }
            else if (Bb [p])
            { 
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = alpha + B(i,j)
                    GB_LOAD_B (bij, Bx, p, B_iso) ;
                    GB_EWISEOP (Cx, p, alpha_scalar, bij, p % vlen, p / vlen) ;
                }
                #else
                { 
                    // C (i,j) = B (i,j)
                    GB_COPY_B_to_C (Cx, p, Bx, p, B_iso) ;
                }
                #endif
                c = 1 ;
            }
            else if (Ab [p])
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
                c = 1 ;
            }
            #endif
            Cb [p] = c ;
            task_cnvals += c ;
        }
        cnvals += task_cnvals ;
    }
}

