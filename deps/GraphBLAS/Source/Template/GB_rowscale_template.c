//------------------------------------------------------------------------------
// GB_rowscale_template: C=D*B where D is a square diagonal matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This template is not used If C is iso, since all that is needed is to create
// C as a shallow-copy of the pattern of A.

// B and C can be jumbled.  D cannot, but it is a diagonal matrix so it is
// never jumbled.

{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (!GB_JUMBLED (D)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!C->iso) ;

    //--------------------------------------------------------------------------
    // get D and B
    //--------------------------------------------------------------------------

    const GB_A_TYPE *restrict Dx = (GB_A_TYPE *) D->x ;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;

    #ifdef GB_JIT_KERNEL
    #define D_iso GB_A_ISO
    #define B_iso GB_B_ISO
    #else
    const bool D_iso = D->iso ;
    const bool B_iso = B->iso ;
    #endif

    const int64_t *restrict Bi = B->i ;
    GB_B_NVALS (bnz) ;      // const int64_t bnz = GB_nnz (B) ;
    const int64_t bvlen = B->vlen ;

    //--------------------------------------------------------------------------
    // C=D*B
    //--------------------------------------------------------------------------

    int ntasks = nthreads ;
    ntasks = GB_IMIN (bnz, ntasks) ;

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        int64_t pstart, pend ;
        GB_PARTITION (pstart, pend, bnz, tid, ntasks) ;
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = pstart ; p < pend ; p++)
        { 
            int64_t i = GBI_B (Bi, p, bvlen) ;      // get row index of B(i,j)
            GB_DECLAREA (dii) ;
            GB_GETA (dii, Dx, i, D_iso) ;           // dii = D(i,i)
            GB_DECLAREB (bij) ;
            GB_GETB (bij, Bx, p, B_iso) ;           // bij = B(i,j)
            GB_EWISEOP (Cx, p, dii, bij, 0, 0) ;    // C(i,j) = dii*bij
        }
    }
}

