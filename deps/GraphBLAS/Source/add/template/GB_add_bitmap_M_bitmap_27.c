//------------------------------------------------------------------------------
// GB_add_bitmap_M_bitmap_27: C<#M>=A+B, C bitmap; M, A, and B bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.
// M is bitmap or full, complemented or not, and either value or structural.
// A and B are both bitmap/full.

{

    //--------------------------------------------------------------------------
    // Method27: C is bitmap; M, A, and B are bitmap or full
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
            GB_GET_MIJ (p) ;
            if (mij)
            {
                // M(i,j) is true, so C(i,j) can be computed
                int8_t a = GBB_A (Ab, p) ;
                int8_t b = GBB_B (Bb, p) ;
                #ifdef GB_ISO_ADD
                int8_t c = a || b ;
                #else
                int8_t c = 0 ;
                if (a && b)
                { 
                    // C (i,j) = A (i,j) + B (i,j)
                    GB_LOAD_A (aij, Ax, p, A_iso) ;
                    GB_LOAD_B (bij, Bx, p, B_iso) ;
                    GB_EWISEOP (Cx, p, aij, bij, p % vlen, p / vlen) ;
                    c = 1 ;
                }
                else if (b)
                { 
                    #if GB_IS_EWISEUNION
                    { 
                        // C (i,j) = alpha + B(i,j)
                        GB_LOAD_B (bij, Bx, p, B_iso) ;
                        GB_EWISEOP (Cx, p, alpha_scalar, bij,
                            p % vlen, p / vlen) ;
                    }
                    #else
                    { 
                        // C (i,j) = B (i,j)
                        GB_COPY_B_to_C (Cx, p, Bx, p, B_iso) ;
                    }
                    #endif
                    c = 1 ;
                }
                else if (a)
                { 
                    #if GB_IS_EWISEUNION
                    { 
                        // C (i,j) = A(i,j) + beta
                        GB_LOAD_A (aij, Ax, p, A_iso) ;
                        GB_EWISEOP (Cx, p, aij, beta_scalar,
                            p % vlen, p / vlen) ;
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
            else
            { 
                // M(i,j) == 1, so C(i,j) is not computed
                Cb [p] = 0 ;
            }
        }
        cnvals += task_cnvals ;
    }
}

