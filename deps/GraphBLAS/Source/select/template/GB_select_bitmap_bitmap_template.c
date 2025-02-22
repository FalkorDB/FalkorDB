//------------------------------------------------------------------------------
// GB_select_bitmap_bitmap_template: C=select(A,thunk) if A is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C and A are bitmap.

{
    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        reduction(+:cnvals)
    for (p = 0 ; p < anz ; p++)
    { 
        int64_t i = p % avlen ;
        int64_t j = p / avlen ;
        int8_t cb = Ab [p] ;
        #if defined ( GB_ENTRY_SELECTOR )
        if (cb)
        { 
            GB_TEST_VALUE_OF_ENTRY (keep, p) ;
            cb = keep ;
        }
        #else
        { 
            #if defined ( GB_TRIL_SELECTOR )
                cb = cb && (j-i <= ithunk) ;
            #elif defined ( GB_TRIU_SELECTOR )
                cb = cb && (j-i >= ithunk) ;
            #elif defined ( GB_DIAG_SELECTOR )
                cb = cb && (j-i == ithunk) ;
            #elif defined ( GB_OFFDIAG_SELECTOR )
                cb = cb && (j-i != ithunk) ;
            #elif defined ( GB_ROWINDEX_SELECTOR )
                cb = cb && (i+ithunk != 0) ;
            #elif defined ( GB_COLINDEX_SELECTOR )
                cb = cb && (j+ithunk != 0) ;
            #elif defined ( GB_COLLE_SELECTOR )
                cb = cb && (j <= ithunk) ;
            #elif defined ( GB_COLGT_SELECTOR )
                cb = cb && (j > ithunk) ;
            #elif defined ( GB_ROWLE_SELECTOR )
                cb = cb && (i <= ithunk) ;
            #elif defined ( GB_ROWGT_SELECTOR )
                cb = cb && (i > ithunk) ;
            #endif
        }
        #endif
        Cb [p] = cb ;
        cnvals += cb ;
    }
}

