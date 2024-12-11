//------------------------------------------------------------------------------
// GB_select_bitmap_full_template: C=select(A,thunk) if A is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  A is full.

{
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    const size_t asize = A->type->size ;
    const int64_t anz = avlen * avdim ;
    int64_t p, cnvals = 0 ;
    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        reduction(+:cnvals)
    for (p = 0 ; p < anz ; p++)
    { 
        int64_t i = p % avlen ;
        int64_t j = p / avlen ;
        int8_t cb ;
        #if defined ( GB_ENTRY_SELECTOR )
            GB_TEST_VALUE_OF_ENTRY (keep, p) ;
            cb = keep ;
        #elif defined ( GB_TRIL_SELECTOR )
            cb = (j-i <= ithunk) ;
        #elif defined ( GB_TRIU_SELECTOR )
            cb = (j-i >= ithunk) ;
        #elif defined ( GB_DIAG_SELECTOR )
            cb = (j-i == ithunk) ;
        #elif defined ( GB_OFFDIAG_SELECTOR )
            cb = (j-i != ithunk) ;
        #elif defined ( GB_ROWINDEX_SELECTOR )
            cb = (i+ithunk != 0) ;
        #elif defined ( GB_COLINDEX_SELECTOR )
            cb = (j+ithunk != 0) ;
        #elif defined ( GB_COLLE_SELECTOR )
            cb = (j <= ithunk) ;
        #elif defined ( GB_COLGT_SELECTOR )
            cb = (j > ithunk) ;
        #elif defined ( GB_ROWLE_SELECTOR )
            cb = (i <= ithunk) ;
        #elif defined ( GB_ROWGT_SELECTOR )
            cb = (i > ithunk) ;
        #endif
        Cb [p] = cb ;
        cnvals += cb ;
    }
    (*cnvals_handle) = cnvals ;
}

