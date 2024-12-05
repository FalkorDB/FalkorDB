//------------------------------------------------------------------------------
// GB_split_bitmap_template: split a bitmap matrix into a bitmap tile
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get C and the tile A
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    int64_t avlen = A->vlen ;
    int64_t cvlen = C->vlen ;
    int64_t cvdim = C->vdim ;
    int64_t cnzmax = cvlen * cvdim ;
    const int8_t *restrict Ab = A->b ;
          int8_t *restrict Cb = C->b ;
    #endif

    #ifndef GB_ISO_SPLIT
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    int64_t pC ;
    int64_t cnz = 0 ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
        reduction(+:cnz)
    for (pC = 0 ; pC < cnzmax ; pC++)
    {
        int64_t i = pC % cvlen ;
        int64_t j = pC / cvlen ;
        int64_t iA = aistart + i ;
        int64_t jA = avstart + j ;
        int64_t pA = iA + jA * avlen ;
        Cb [pC] = Ab [pA] ;
        if (Ab [pA])
        { 
            // Cx [pC] = Ax [pA] ;
            GB_COPY (pC, pA) ;
            cnz++ ;
        }
    }

    C->nvals = cnz ;
}

#undef GB_C_TYPE
#undef GB_A_TYPE
#undef GB_ISO_SPLIT

