//------------------------------------------------------------------------------
// GB_add_full_33:  C=A+B; C and B are full, A is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // Method33: C and B are full; A is bitmap
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(C_nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    {
        if (Ab [p])
        { 
            // C (i,j) = A (i,j) + B (i,j)
            GB_LOAD_A (aij, Ax, p, A_iso) ;
            GB_LOAD_B (bij, Bx, p, B_iso) ;
            GB_EWISEOP (Cx, p, aij, bij, p % vlen, p / vlen) ;
        }
        else
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
        }
    }
}

