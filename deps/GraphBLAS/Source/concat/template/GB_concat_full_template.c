//------------------------------------------------------------------------------
// GB_concat_full_template: concatenate a full tile into a full matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If C and A are iso, this method is not used.

{

    //--------------------------------------------------------------------------
    // get C and the tile A
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    #define A_iso GB_A_ISO
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    const int64_t anz = avlen * avdim ;
    const int64_t cvlen = C->vlen ;
    #else
    const bool A_iso = A->iso ;
    #endif

    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;

    int64_t pA ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(static)
    for (pA = 0 ; pA < anz ; pA++)
    { 
        int64_t i = pA % avlen ;
        int64_t j = pA / avlen ;
        int64_t iC = cistart + i ;
        int64_t jC = cvstart + j ;
        int64_t pC = iC + jC * cvlen ;
        // Cx [pC] = Ax [pA] ;
        GB_COPY (pC, pA, A_iso) ;
    }
}

#undef GB_C_TYPE
#undef GB_A_TYPE

