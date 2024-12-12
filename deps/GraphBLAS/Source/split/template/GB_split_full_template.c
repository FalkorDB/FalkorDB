//------------------------------------------------------------------------------
// GB_split_full_template: split a full matrix into a full tile
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is not used when the matrices are iso.

{

    //--------------------------------------------------------------------------
    // get C and the tile A
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    int64_t avlen = A->vlen ;
    int64_t cvlen = C->vlen ;
    int64_t cvdim = C->vdim ;
    int64_t cnz = cvlen * cvdim ;
    #endif

    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    ASSERT (!A->iso) ;

    int64_t pC ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(static)
    for (pC = 0 ; pC < cnz ; pC++)
    { 
        int64_t i = pC % cvlen ;
        int64_t j = pC / cvlen ;
        int64_t iA = aistart + i ;
        int64_t jA = avstart + j ;
        int64_t pA = iA + jA * avlen ;
        // Cx [pC] = Ax [pA] ;
        GB_COPY (pC, pA) ;
    }
}

#undef GB_C_TYPE
#undef GB_A_TYPE

