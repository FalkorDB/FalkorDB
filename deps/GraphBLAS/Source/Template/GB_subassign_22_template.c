//------------------------------------------------------------------------------
// GB_subassign_22_template: C += y where C is full and y is a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 22: C += scalar, where C is dense

// M:           NULL
// Mask_comp:   false
// Mask_struct: ignored
// C_replace:   false
// accum:       present
// A:           scalar
// S:           none

// C += scalar where C is full.

{

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_C_NVALS (cnz) ;      // int64_t cnz = GB_nnz (C) ;
    const int nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    ASSERT (!C->iso) ;

    //--------------------------------------------------------------------------
    // C += y where C is dense and y is a scalar
    //--------------------------------------------------------------------------

    int64_t pC ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (pC = 0 ; pC < cnz ; pC++)
    { 
        // Cx [pC] += ywork
        GB_ACCUMULATE_scalar (Cx, pC, ywork) ;
    }
}

