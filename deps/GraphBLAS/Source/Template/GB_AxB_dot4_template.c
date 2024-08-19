//------------------------------------------------------------------------------
// GB_AxB_dot4_template:  C+=A'*B via dot products, where C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C+=A'*B where C is full and computed in-place.  The monoid of the semiring
// matches the accum operator, and the type of C matches the ztype of accum.

// The PAIR and FIRSTJ multiplicative operators are important special cases.

// The matrix C is the user input matrix.  C is not iso on output, but might
// iso on input, in which case the input iso scalar is cinput, and C->x has
// been expanded to non-iso, and initialized if A and/or B are hypersparse.
// A and/or B can be iso.

// The #include'ing file must use GB_DECLARE_TERMINAL_CONST (zterminal) ;

#ifndef GB_MXM_SHARED_DEFINITIONS_H
#error "undefined"
#endif

#if GB_IS_ANY_MONOID
#error "dot4 is not used for the ANY monoid"
#endif

#undef  GB_GET4C
#define GB_GET4C(cij,p) cij = (C_in_iso) ? cinput : Cx [p]

#if ((GB_A_IS_BITMAP || GB_A_IS_FULL) && (GB_B_IS_BITMAP || GB_B_IS_FULL))
{

    //--------------------------------------------------------------------------
    // C += A'*B where A and B are both bitmap/full
    //--------------------------------------------------------------------------

    // FUTURE: This method is not particularly efficient when both A and B are
    // bitmap/full.  A better method would use tiles to reduce memory traffic.

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        const int a_tid = tid / nbslice ;
        const int b_tid = tid % nbslice ;
        const int64_t kA_start = A_slice [a_tid] ;
        const int64_t kA_end   = A_slice [a_tid+1] ;
        const int64_t kB_start = B_slice [b_tid] ;
        const int64_t kB_end   = B_slice [b_tid+1] ;

        for (int64_t j = kB_start ; j < kB_end ; j++)
        {

            //------------------------------------------------------------------
            // get B(:,j) and C(:,j)
            //------------------------------------------------------------------

            const int64_t pC_start = j * cvlen ;
            const int64_t pB_start = j * vlen ;

            //------------------------------------------------------------------
            // C(:,j) += A'*B(:,j)
            //------------------------------------------------------------------

            for (int64_t i = kA_start ; i < kA_end ; i++)
            {

                //--------------------------------------------------------------
                // get A(:,i)
                //--------------------------------------------------------------

                const int64_t pA = i * vlen ;

                //--------------------------------------------------------------
                // get C(i,j)
                //--------------------------------------------------------------

                int64_t pC = i + pC_start ;     // C(i,j) is at Cx [pC]
                GB_C_TYPE GB_GET4C (cij, pC) ;   // cij = Cx [pC]

                //--------------------------------------------------------------
                // C(i,j) += A (:,i)*B(:,j): a single dot product
                //--------------------------------------------------------------

                int64_t pB = pB_start ;

                #if ( GB_A_IS_FULL && GB_B_IS_FULL )
                {

                    //----------------------------------------------------------
                    // both A and B are full
                    //----------------------------------------------------------

                    #if GB_IS_LXOR_PAIR_SEMIRING
                    { 
                        // (boolean XOR monoid)_PAIR semiring
                        uint64_t t = ((uint64_t) cij) + vlen ;
                        cij = (GB_C_TYPE) (t & 0x1L) ;
                    }
                    #elif GB_IS_PLUS_PAIR_8_SEMIRING
                    { 
                        // (PLUS int8, uint8 monoids)_PAIR semirings
                        uint64_t t = ((uint64_t) cij) + vlen ;
                        cij = (GB_C_TYPE) (t & 0xFFL) ;
                    }
                    #elif GB_IS_PLUS_PAIR_16_SEMIRING
                    { 
                        // (PLUS int16, uint16 monoids)_PAIR semirings
                        uint64_t t = ((uint64_t) cij) + vlen ;
                        cij = (GB_C_TYPE) (t & 0xFFFFL) ;
                    }
                    #elif GB_IS_PLUS_PAIR_32_SEMIRING
                    { 
                        // (PLUS int32, uint32 monoids)_PAIR semirings
                        uint64_t t = ((uint64_t) cij) + vlen ;
                        cij = (GB_C_TYPE) (t & 0xFFFFFFFFL) ;
                    }
                    #elif GB_IS_PLUS_PAIR_BIG_SEMIRING
                    { 
                        // (PLUS int64, uint64, float, or double)_PAIR semirings
                        cij += (GB_C_TYPE) vlen ;
                    }
//                  #elif GB_IS_PLUS_PAIR_FC32_SEMIRING
//                  {
//                      // (PLUS monoid for float complex)_PAIR semiring
//                      cij = GJ_CMPLX32 (GB_crealf (cij) + (float) vlen, 0) ;
//                  }
//                  #elif GB_IS_PLUS_PAIR_FC64_SEMIRING
//                  {
//                      // (PLUS monoid for double complex)_PAIR semiring
//                      cij = GJ_CMPLX64 (GB_creal (cij) + (double) vlen, 0) ;
//                  }
                    #elif GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry
                        if (vlen > 0)
                        { 
                            int64_t k = GB_OFFSET ;
                            cij = GB_IMIN (cij, k) ;
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry
                        if (vlen > 0)
                        { 
                            int64_t k = vlen-1 + GB_OFFSET ;
                            cij = GB_IMAX (cij, k) ;
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_REDUCTION_MONOID (cij)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        { 
                            GB_DOT (k, pA+k, pB+k) ;    // cij += A(k,i)*B(k,j)
                        }
                    }
                    #endif

                }
                #elif ( GB_A_IS_FULL && GB_B_IS_BITMAP )
                {

                    //----------------------------------------------------------
                    // A is full and B is bitmap
                    //----------------------------------------------------------

                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry in B(:,j)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Bb [pB+k])
                            { 
                                cij = GB_IMIN (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry in B(:,j)
                        for (int64_t k = vlen-1 ; k >= 0 ; k--)
                        {
                            if (Bb [pB+k])
                            { 
                                cij = GB_IMAX (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_REDUCTION_MONOID (cij)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Bb [pB+k])
                            { 
                                GB_DOT (k, pA+k, pB+k) ; // cij += A(k,i)*B(k,j)
                            }
                        }
                    }
                    #endif

                }
                #elif ( GB_A_IS_BITMAP && GB_B_IS_FULL )
                {

                    //----------------------------------------------------------
                    // A is bitmap and B is full
                    //----------------------------------------------------------

                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry in A(:,i)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Ab [pA+k])
                            { 
                                cij = GB_IMIN (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry in A(:,i)
                        for (int64_t k = vlen-1 ; k >= 0 ; k--)
                        {
                            if (Ab [pA+k])
                            { 
                                cij = GB_IMAX (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_REDUCTION_MONOID (cij)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Ab [pA+k])
                            { 
                                GB_DOT (k, pA+k, pB+k) ; // cij += A(k,i)*B(k,j)
                            }
                        }
                    }
                    #endif

                }
                #elif ( GB_A_IS_BITMAP && GB_B_IS_BITMAP )
                {

                    //----------------------------------------------------------
                    // both A and B are bitmap
                    //----------------------------------------------------------

                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Ab [pA+k] && Bb [pB+k])
                            { 
                                cij = GB_IMIN (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry
                        for (int64_t k = vlen-1 ; k >= 0 ; k--)
                        {
                            if (Ab [pA+k] && Bb [pB+k])
                            { 
                                cij = GB_IMAX (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_REDUCTION_MONOID (cij)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Ab [pA+k] && Bb [pB+k])
                            { 
                                GB_DOT (k, pA+k, pB+k) ; // cij += A(k,i)*B(k,j)
                            }
                        }
                    }
                    #endif

                }
                #endif

                //--------------------------------------------------------------
                // save C(i,j)
                //--------------------------------------------------------------

                Cx [pC] = cij ;
            }
        }
    }
}
#elif ((GB_A_IS_SPARSE || GB_A_IS_HYPER) && (GB_B_IS_BITMAP || GB_B_IS_FULL ))
{

    //--------------------------------------------------------------------------
    // C += A'*B when A is sparse/hyper and B is bitmap/full
    //--------------------------------------------------------------------------

    // special cases: these methods are very fast, but cannot or do not need
    // to be unrolled.
    #undef  GB_SPECIAL_CASE_OR_TERMINAL
    #define GB_SPECIAL_CASE_OR_TERMINAL \
       (   GB_IS_PAIR_MULTIPLIER        /* the multiply op is PAIR */       \
        || GB_IS_MIN_FIRSTJ_SEMIRING    /* min_firstj semiring */           \
        || GB_IS_MAX_FIRSTJ_SEMIRING    /* max_firstj semiring */           \
        || GB_MONOID_IS_TERMINAL        /* monoid has a terminal value */   \
        || GB_B_IS_PATTERN )            /* B is pattern-only */

    // Transpose B and unroll the innermost loop if this condition holds: A
    // must be sparse, B must be full, and no special semirings or operators
    // can be used.  The monoid must not be terminal.  These conditions are
    // known at compile time.
    #undef  GB_UNROLL
    #define GB_UNROLL \
        ( GB_A_IS_SPARSE && GB_B_IS_FULL && !( GB_SPECIAL_CASE_OR_TERMINAL ) )

    // If GB_UNROLL is true at compile-time, the simpler variant can still be
    // used, without unrolling, for any of these conditions:  (1) A is very
    // sparse (fewer entries than the size of the W workspace) or (2) B is iso.

    // The unrolled method does not allow B to be iso or pattern-only (such as
    // for the FIRST multiplicative operator.  If B is iso or pattern-only, the
    // dense matrix G = B' would be a single scalar, or its values would not be
    // accessed at all, so there is no benefit to computing G.

    #if GB_UNROLL
    const int64_t wp = (bvdim == 1) ? 0 : GB_IMIN (bvdim, 4) ;
    GB_A_NVALS (anz) ;      // const int64_t anz = GB_nnz (A) ;
    if ((anz < wp * vlen) || B_iso)
    #endif
    {

        //----------------------------------------------------------------------
        // C += A'*B without workspace
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            const int64_t kA_start = A_slice [tid] ;
            const int64_t kA_end   = A_slice [tid+1] ;

            //------------------------------------------------------------------
            // C+=A'*B where A is sparse/hyper and B is bitmap/full
            //------------------------------------------------------------------

            if (bvdim == 1)
            {

                //--------------------------------------------------------------
                // C += A'*B where C is a single vector
                //--------------------------------------------------------------

                #define pC_start 0
                #define pB 0
                #define j 0
                for (int64_t kA = kA_start ; kA < kA_end ; kA++)
                { 
                    // get A(:,i)
                    #if GB_A_IS_HYPER
                    const int64_t i = Ah [kA] ;
                    #else
                    const int64_t i = kA ;
                    #endif
                    int64_t pA = Ap [kA] ;
                    const int64_t pA_end = Ap [kA+1] ;
                    const int64_t ainz = pA_end - pA ;
                    // C(i) += A(:,i)'*B(:,0)
                    #include "GB_AxB_dot4_cij.c"
                }
                #undef pC_start
                #undef pB
                #undef j

            }
            else
            {

                //--------------------------------------------------------------
                // C += A'*B where C is a matrix
                //--------------------------------------------------------------

                for (int64_t kA = kA_start ; kA < kA_end ; kA++)
                {
                    // get A(:,i)
                    #if GB_A_IS_HYPER
                    const int64_t i = Ah [kA] ;
                    #else
                    const int64_t i = kA ;
                    #endif
                    int64_t pA = Ap [kA] ;
                    const int64_t pA_end = Ap [kA+1] ;
                    const int64_t ainz = pA_end - pA ;
                    // C(i,:) += A(:,i)'*B
                    for (int64_t j = 0 ; j < bvdim ; j++)
                    { 
                        // get B(:,j) and C(:,j)
                        const int64_t pC_start = j * cvlen ;
                        const int64_t pB = j * vlen ;
                        // C(i,j) += A(:,i)'*B(:,j)
                        #include "GB_AxB_dot4_cij.c"
                    }
                }
            }
        }
    }
    #if GB_UNROLL
    else
    {

        //----------------------------------------------------------------------
        // C += A'*B: with workspace W for transposing B, one panel at a time
        //----------------------------------------------------------------------

        size_t W_size = 0 ;
        GB_B2TYPE *restrict W = NULL ;
        if (bvdim > 1)
        {
            W = GB_MALLOC_WORK (wp * vlen, GB_B2TYPE, &W_size) ;
            if (W == NULL)
            { 
                // out of memory
                return (GrB_OUT_OF_MEMORY) ;
            }
        }

        for (int64_t j1 = 0 ; j1 < bvdim ; j1 += 4)
        {

            //------------------------------------------------------------------
            // C(:,j1:j2-1) += A' * B (:,j1:j2-1) for a single panel
            //------------------------------------------------------------------

            const int64_t j2 = GB_IMIN (j1 + 4, bvdim) ;
            switch (j2 - j1)
            {

                default :
                case 1 :
                {

                    //----------------------------------------------------------
                    // C(:,j1:j2-1) is a single vector; use B(:,j1) in place
                    //----------------------------------------------------------

                    const GB_B_TYPE *restrict Bxj1 = Bx + j1 * vlen ;
                    int tid ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(dynamic,1)
                    for (tid = 0 ; tid < ntasks ; tid++)
                    {
                        // get the task descriptor
                        const int64_t kA_start = A_slice [tid] ;
                        const int64_t kA_end   = A_slice [tid+1] ;
                        for (int64_t i = kA_start ; i < kA_end ; i++)
                        {
                            // get A(:,i)
                            const int64_t pA = Ap [i] ;
                            const int64_t pA_end = Ap [i+1] ;
                            // cx [0] = C(i,j1)
                            GB_C_TYPE cx [1] ;
                            GB_GET4C (cx [0], i + j1*cvlen) ;
                            // cx [0] += A (:,i)'*B(:,j1)
                            for (int64_t p = pA ; p < pA_end ; p++)
                            { 
                                // aki = A(k,i)
                                const int64_t k = Ai [p] ;
                                GB_DECLAREA (aki) ;
                                GB_GETA (aki, Ax, p, A_iso) ;
                                GB_DECLAREB (bkj) ;
                                GB_GETB (bkj, Bxj1, k, false) ;
                                // cx [0] += A(k,i)*B(k,j1)
                                GB_MULTADD (cx [0], aki, bkj, i, k, j1) ;
                            }
                            // C(i,j1) = cx [0]
                            Cx [i + j1*cvlen] = cx [0] ;
                        }
                    }
                }
                break ;

                case 2 :
                {

                    //----------------------------------------------------------
                    // G = B(:,j1:j1+1) and convert to row-form
                    //----------------------------------------------------------

                    GB_B2TYPE *restrict G = W ;
                    int64_t k ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(static)
                    for (k = 0 ; k < vlen ; k++)
                    { 
                        // G (k,0:1) = B (k,j1:j1+1)
                        const int64_t k2 = k << 1 ;
                        GB_GETB (G [k2    ], Bx, k + (j1    ) * vlen, false) ;
                        GB_GETB (G [k2 + 1], Bx, k + (j1 + 1) * vlen, false) ;
                    }

                    //----------------------------------------------------------
                    // C += A'*G where G is vlen-by-2 in row-form
                    //----------------------------------------------------------

                    int tid ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(dynamic,1)
                    for (tid = 0 ; tid < ntasks ; tid++)
                    {
                        // get the task descriptor
                        const int64_t kA_start = A_slice [tid] ;
                        const int64_t kA_end   = A_slice [tid+1] ;
                        for (int64_t i = kA_start ; i < kA_end ; i++)
                        {
                            // get A(:,i)
                            const int64_t pA = Ap [i] ;
                            const int64_t pA_end = Ap [i+1] ;
                            // cx [0:1] = C(i,j1:j1+1)
                            GB_C_TYPE cx [2] ;
                            GB_GET4C (cx [0], i + (j1  )*cvlen) ;
                            GB_GET4C (cx [1], i + (j1+1)*cvlen) ;
                            // cx [0:1] += A (:,i)'*G
                            for (int64_t p = pA ; p < pA_end ; p++)
                            { 
                                // aki = A(k,i)
                                const int64_t k = Ai [p] ;
                                GB_DECLAREA (aki) ;
                                GB_GETA (aki, Ax, p, A_iso) ;
                                const int64_t k2 = k << 1 ;
                                // cx [0:1] += A(k,i)*G(k,0:1)
                                GB_MULTADD (cx [0], aki, G [k2],   i, k, j1) ;
                                GB_MULTADD (cx [1], aki, G [k2+1], i, k, j1+1) ;
                            }
                            // C(i,j1:j1+1) = cx [0:1]
                            Cx [i + (j1  )*cvlen] = cx [0] ;
                            Cx [i + (j1+1)*cvlen] = cx [1] ;
                        }
                    }
                }
                break ;

                case 3 :
                {

                    //----------------------------------------------------------
                    // G = B(:,j1:j1+2) and convert to row-form
                    //----------------------------------------------------------

                    GB_B2TYPE *restrict G = W ;
                    int64_t k ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(static)
                    for (k = 0 ; k < vlen ; k++)
                    { 
                        // G (k,0:2) = B (k,j1:j1+2)
                        const int64_t k3 = k * 3 ;
                        GB_GETB (G [k3    ], Bx, k + (j1    ) * vlen, false) ;
                        GB_GETB (G [k3 + 1], Bx, k + (j1 + 1) * vlen, false) ;
                        GB_GETB (G [k3 + 2], Bx, k + (j1 + 2) * vlen, false) ;
                    }

                    //----------------------------------------------------------
                    // C += A'*G where G is vlen-by-3 in row-form
                    //----------------------------------------------------------

                    int tid ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(dynamic,1)
                    for (tid = 0 ; tid < ntasks ; tid++)
                    {
                        // get the task descriptor
                        const int64_t kA_start = A_slice [tid] ;
                        const int64_t kA_end   = A_slice [tid+1] ;
                        for (int64_t i = kA_start ; i < kA_end ; i++)
                        {
                            // get A(:,i)
                            const int64_t pA = Ap [i] ;
                            const int64_t pA_end = Ap [i+1] ;
                            // cx [0:2] = C(i,j1:j1+2)
                            GB_C_TYPE cx [3] ;
                            GB_GET4C (cx [0], i + (j1  )*cvlen) ;
                            GB_GET4C (cx [1], i + (j1+1)*cvlen) ;
                            GB_GET4C (cx [2], i + (j1+2)*cvlen) ;
                            // cx [0:2] += A (:,i)'*G
                            for (int64_t p = pA ; p < pA_end ; p++)
                            { 
                                // aki = A(k,i)
                                const int64_t k = Ai [p] ;
                                GB_DECLAREA (aki) ;
                                GB_GETA (aki, Ax, p, A_iso) ;
                                const int64_t k3 = k * 3 ;
                                // cx [0:2] += A(k,i)*G(k,0:2)
                                GB_MULTADD (cx [0], aki, G [k3  ], i, k, j1) ;
                                GB_MULTADD (cx [1], aki, G [k3+1], i, k, j1+1) ;
                                GB_MULTADD (cx [2], aki, G [k3+2], i, k, j1+2) ;
                            }
                            // C(i,j1:j1+2) = cx [0:2]
                            Cx [i + (j1  )*cvlen] = cx [0] ;
                            Cx [i + (j1+1)*cvlen] = cx [1] ;
                            Cx [i + (j1+2)*cvlen] = cx [2] ;
                        }
                    }
                }
                break ;

                case 4 :
                {

                    //----------------------------------------------------------
                    // G = B(:,j1:j1+3) and convert to row-form
                    //----------------------------------------------------------

                    GB_B2TYPE *restrict G = W ;

                    int64_t k ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(static)
                    for (k = 0 ; k < vlen ; k++)
                    { 
                        // G (k,0:3) = B (k,j1:j1+3)
                        const int64_t k4 = k << 2 ;
                        GB_GETB (G [k4    ], Bx, k + (j1    ) * vlen, false) ;
                        GB_GETB (G [k4 + 1], Bx, k + (j1 + 1) * vlen, false) ;
                        GB_GETB (G [k4 + 2], Bx, k + (j1 + 2) * vlen, false) ;
                        GB_GETB (G [k4 + 3], Bx, k + (j1 + 3) * vlen, false) ;
                    }

                    //----------------------------------------------------------
                    // C += A'*G where G is vlen-by-4 in row-form
                    //----------------------------------------------------------

                    int tid ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(dynamic,1)
                    for (tid = 0 ; tid < ntasks ; tid++)
                    {
                        // get the task descriptor
                        const int64_t kA_start = A_slice [tid] ;
                        const int64_t kA_end   = A_slice [tid+1] ;
                        for (int64_t i = kA_start ; i < kA_end ; i++)
                        {
                            // get A(:,i)
                            const int64_t pA = Ap [i] ;
                            const int64_t pA_end = Ap [i+1] ;
                            // cx [0:3] = C(i,j1:j1+3)
                            GB_C_TYPE cx [4] ;
                            GB_GET4C (cx [0], i + (j1  )*cvlen) ;
                            GB_GET4C (cx [1], i + (j1+1)*cvlen) ;
                            GB_GET4C (cx [2], i + (j1+2)*cvlen) ;
                            GB_GET4C (cx [3], i + (j1+3)*cvlen) ;
                            // cx [0:3] += A (:,i)'*G
                            for (int64_t p = pA ; p < pA_end ; p++)
                            { 
                                // aki = A(k,i)
                                const int64_t k = Ai [p] ;
                                GB_DECLAREA (aki) ;
                                GB_GETA (aki, Ax, p, A_iso) ;
                                const int64_t k4 = k << 2 ;
                                // cx [0:3] += A(k,i)*G(k,0:3)
                                GB_MULTADD (cx [0], aki, G [k4  ], i, k, j1) ;
                                GB_MULTADD (cx [1], aki, G [k4+1], i, k, j1+1) ;
                                GB_MULTADD (cx [2], aki, G [k4+2], i, k, j1+2) ;
                                GB_MULTADD (cx [3], aki, G [k4+3], i, k, j1+3) ;
                            }
                            // C(i,j1:j1+3) = cx [0:3]
                            Cx [i + (j1  )*cvlen] = cx [0] ;
                            Cx [i + (j1+1)*cvlen] = cx [1] ;
                            Cx [i + (j1+2)*cvlen] = cx [2] ;
                            Cx [i + (j1+3)*cvlen] = cx [3] ;
                        }
                    }
                }
                break ;
            }
        }

        // free workspace
        GB_FREE_WORK (&W, W_size) ;
    }
    #endif

}
#elif ( (GB_A_IS_BITMAP || GB_A_IS_FULL) && (GB_B_IS_SPARSE || GB_B_IS_HYPER))
{

    //--------------------------------------------------------------------------
    // C += A'*B where A is bitmap/full and B is sparse/hyper
    //--------------------------------------------------------------------------

    // FUTURE: this can be unrolled, like the case above

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        const int64_t kB_start = B_slice [tid] ;
        const int64_t kB_end   = B_slice [tid+1] ;

        for (int64_t kB = kB_start ; kB < kB_end ; kB++)
        {

            //------------------------------------------------------------------
            // get B(:,j) and C(:,j)
            //------------------------------------------------------------------

            #if GB_B_IS_HYPER
            const int64_t j = Bh [kB] ;
            #else
            const int64_t j = kB ;
            #endif
            const int64_t pC_start = j * cvlen ;
            const int64_t pB_start = Bp [kB] ;
            const int64_t pB_end = Bp [kB+1] ;
            const int64_t bjnz = pB_end - pB_start ;

            //------------------------------------------------------------------
            // C(:,j) += A'*B(:,j)
            //------------------------------------------------------------------

            for (int64_t i = 0 ; i < avdim ; i++)
            {

                //--------------------------------------------------------------
                // get A(:,i)
                //--------------------------------------------------------------

                const int64_t pA = i * vlen ;

                //--------------------------------------------------------------
                // get C(i,j)
                //--------------------------------------------------------------

                int64_t pC = i + pC_start ;     // C(i,j) is at Cx [pC]
                GB_C_TYPE GB_GET4C (cij, pC) ;   // cij = Cx [pC]

                //--------------------------------------------------------------
                // C(i,j) += A (:,i)*B(:,j): a single dot product
                //--------------------------------------------------------------

                int64_t pB = pB_start ;

                #if ( GB_A_IS_FULL )
                {

                    //----------------------------------------------------------
                    // A is full and B is sparse/hyper
                    //----------------------------------------------------------

                    #if GB_IS_LXOR_PAIR_SEMIRING
                    { 
                        // (boolean XOR monoid)_PAIR semiring
                        uint64_t t = ((uint64_t) cij) + bjnz ;
                        cij = (GB_C_TYPE) (t & 0x1L) ;
                    }
                    #elif GB_IS_PLUS_PAIR_8_SEMIRING
                    { 
                        // (PLUS int8, uint8 monoids)_PAIR semirings
                        uint64_t t = ((uint64_t) cij) + bjnz ;
                        cij = (GB_C_TYPE) (t & 0xFFL) ;
                    }
                    #elif GB_IS_PLUS_PAIR_16_SEMIRING
                    { 
                        // (PLUS int16, uint16 monoids)_PAIR semirings
                        uint64_t t = ((uint64_t) cij) + bjnz ;
                        cij = (GB_C_TYPE) (t & 0xFFFFL) ;
                    }
                    #elif GB_IS_PLUS_PAIR_32_SEMIRING
                    { 
                        // (PLUS int32, uint32 monoids)_PAIR semirings
                        uint64_t t = ((uint64_t) cij) + bjnz ;
                        cij = (GB_C_TYPE) (t & 0xFFFFFFFFL) ;
                    }
                    #elif GB_IS_PLUS_PAIR_BIG_SEMIRING
                    { 
                        // (PLUS int64, uint64, float, or double)_PAIR semirings
                        cij += (GB_C_TYPE) bjnz ;
                    }
//                  #elif GB_IS_PLUS_PAIR_FC32_SEMIRING
//                  {
//                      // (PLUS monoid for float complex)_PAIR semiring
//                      cij = GJ_CMPLX32 (GB_crealf (cij) + (float) bjnz, 0) ;
//                  }
//                  #elif GB_IS_PLUS_PAIR_FC64_SEMIRING
//                  {
//                      // (PLUS monoid for double complex)_PAIR semiring
//                      cij = GJ_CMPLX64 (GB_creal (cij) + (double) bjnz, 0) ;
//                  }
                    #elif GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry in B(:,j)
                        if (bjnz > 0)
                        { 
                            int64_t k = Bi [pB] + GB_OFFSET ;
                            cij = GB_IMIN (cij, k) ;
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry in B(:,j)
                        if (bjnz > 0)
                        { 
                            int64_t k = Bi [pB_end-1] + GB_OFFSET ;
                            cij = GB_IMAX (cij, k) ;
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_REDUCTION_MONOID (cij)
                        for (int64_t p = pB ; p < pB_end ; p++)
                        { 
                            int64_t k = Bi [p] ;
                            GB_DOT (k, pA+k, p) ;   // cij += A(k,i)*B(k,j)
                        }
                    }
                    #endif

                }
                #else
                {

                    //----------------------------------------------------------
                    // A is bitmap and B is sparse/hyper
                    //----------------------------------------------------------

                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry
                        for (int64_t p = pB ; p < pB_end ; p++)
                        {
                            int64_t k = Bi [p] ;
                            if (Ab [pA+k])
                            { 
                                cij = GB_IMIN (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry
                        for (int64_t p = pB_end-1 ; p >= pB ; p--)
                        {
                            int64_t k = Bi [p] ;
                            if (Ab [pA+k])
                            { 
                                cij = GB_IMAX (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_REDUCTION_MONOID (cij)
                        for (int64_t p = pB ; p < pB_end ; p++)
                        {
                            int64_t k = Bi [p] ;
                            if (Ab [pA+k])
                            { 
                                GB_DOT (k, pA+k, p) ;   // cij += A(k,i)*B(k,j)
                            }
                        }
                    }
                    #endif

                }
                #endif

                //--------------------------------------------------------------
                // save C(i,j)
                //--------------------------------------------------------------

                Cx [pC] = cij ;
            }
        }
    }
}
#elif ( (GB_A_IS_SPARSE || GB_A_IS_HYPER) && (GB_B_IS_SPARSE || GB_B_IS_HYPER))
{

    //--------------------------------------------------------------------------
    // C+=A'*B where A and B are both sparse/hyper
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        const int a_tid = tid / nbslice ;
        const int b_tid = tid % nbslice ;
        const int64_t kA_start = A_slice [a_tid] ;
        const int64_t kA_end   = A_slice [a_tid+1] ;
        const int64_t kB_start = B_slice [b_tid] ;
        const int64_t kB_end   = B_slice [b_tid+1] ;

        //----------------------------------------------------------------------
        // C+=A'*B via dot products
        //----------------------------------------------------------------------

        for (int64_t kB = kB_start ; kB < kB_end ; kB++)
        {

            //------------------------------------------------------------------
            // get B(:,j) and C(:,j)
            //------------------------------------------------------------------

            #if GB_B_IS_HYPER
            const int64_t j = Bh [kB] ;
            #else
            const int64_t j = kB ;
            #endif
            const int64_t pC_start = j * cvlen ;
            const int64_t pB_start = Bp [kB] ;
            const int64_t pB_end = Bp [kB+1] ;
            const int64_t bjnz = pB_end - pB_start ;

            //------------------------------------------------------------------
            // C(:,j) += A'*B(:,j) where C is full
            //------------------------------------------------------------------

            for (int64_t kA = kA_start ; kA < kA_end ; kA++)
            {

                //--------------------------------------------------------------
                // get A(:,i)
                //--------------------------------------------------------------

                #if GB_A_IS_HYPER
                const int64_t i = Ah [kA] ;
                #else
                const int64_t i = kA ;
                #endif
                int64_t pA = Ap [kA] ;
                const int64_t pA_end = Ap [kA+1] ;
                const int64_t ainz = pA_end - pA ;

                //--------------------------------------------------------------
                // get C(i,j)
                //--------------------------------------------------------------

                int64_t pC = i + pC_start ;     // C(i,j) is at Cx [pC]
                GB_C_TYPE GB_GET4C (cij, pC) ;   // cij = Cx [pC]

                //--------------------------------------------------------------
                // C(i,j) += A (:,i)*B(:,j): a single dot product
                //--------------------------------------------------------------

                int64_t pB = pB_start ;

                //----------------------------------------------------------
                // both A and B are sparse/hyper
                //----------------------------------------------------------

                // The MIN_FIRSTJ semirings are exploited, by terminating as
                // soon as any entry is found.  The MAX_FIRSTJ semirings are
                // not treated specially here.  They could be done with a
                // backwards traversal of the sparse vectors A(:,i) and
                // B(:,j).

                if (ainz == 0 || bjnz == 0 || 
                    Ai [pA_end-1] < Bi [pB_start] ||
                    Bi [pB_end-1] < Ai [pA])
                { 

                    //------------------------------------------------------
                    // A(:,i) and B(:,j) don't overlap, or are empty
                    //------------------------------------------------------

                }
                else if (ainz > 8 * bjnz)
                {

                    //------------------------------------------------------
                    // B(:,j) is very sparse compared to A(:,i)
                    //------------------------------------------------------

                    while (pA < pA_end && pB < pB_end)
                    {
                        int64_t ia = Ai [pA] ;
                        int64_t ib = Bi [pB] ;
                        if (ia < ib)
                        { 
                            // A(ia,i) appears before B(ib,j)
                            // discard all entries A(ia:ib-1,i)
                            int64_t pleft = pA + 1 ;
                            int64_t pright = pA_end - 1 ;
                            GB_TRIM_BINARY_SEARCH (ib, Ai, pleft, pright) ;
                            ASSERT (pleft > pA) ;
                            pA = pleft ;
                        }
                        else if (ib < ia)
                        { 
                            // B(ib,j) appears before A(ia,i)
                            pB++ ;
                        }
                        else // ia == ib == k
                        { 
                            // A(k,i) and B(k,j) are next entries to merge
                            GB_DOT (ia, pA, pB) ;   // cij += A(k,i)*B(k,j)
                            #if GB_IS_MIN_FIRSTJ_SEMIRING
                            break ;
                            #endif
                            pA++ ;
                            pB++ ;
                        }
                    }

                }
                else if (bjnz > 8 * ainz)
                {

                    //------------------------------------------------------
                    // A(:,i) is very sparse compared to B(:,j)
                    //------------------------------------------------------

                    while (pA < pA_end && pB < pB_end)
                    {
                        int64_t ia = Ai [pA] ;
                        int64_t ib = Bi [pB] ;
                        if (ia < ib)
                        { 
                            // A(ia,i) appears before B(ib,j)
                            pA++ ;
                        }
                        else if (ib < ia)
                        { 
                            // B(ib,j) appears before A(ia,i)
                            // discard all entries B(ib:ia-1,j)
                            int64_t pleft = pB + 1 ;
                            int64_t pright = pB_end - 1 ;
                            GB_TRIM_BINARY_SEARCH (ia, Bi, pleft, pright) ;
                            ASSERT (pleft > pB) ;
                            pB = pleft ;
                        }
                        else // ia == ib == k
                        { 
                            // A(k,i) and B(k,j) are next entries to merge
                            GB_DOT (ia, pA, pB) ;   // cij += A(k,i)*B(k,j)
                            #if GB_IS_MIN_FIRSTJ_SEMIRING
                            break ;
                            #endif
                            pA++ ;
                            pB++ ;
                        }
                    }

                }
                else
                {

                    //------------------------------------------------------
                    // A(:,i) and B(:,j) have about the same sparsity
                    //------------------------------------------------------

                    while (pA < pA_end && pB < pB_end)
                    {
                        int64_t ia = Ai [pA] ;
                        int64_t ib = Bi [pB] ;
                        if (ia < ib)
                        { 
                            // A(ia,i) appears before B(ib,j)
                            pA++ ;
                        }
                        else if (ib < ia)
                        { 
                            // B(ib,j) appears before A(ia,i)
                            pB++ ;
                        }
                        else // ia == ib == k
                        { 
                            // A(k,i) and B(k,j) are the entries to merge
                            GB_DOT (ia, pA, pB) ;   // cij += A(k,i)*B(k,j)
                            #if GB_IS_MIN_FIRSTJ_SEMIRING
                            break ;
                            #endif
                            pA++ ;
                            pB++ ;
                        }
                    }
                }

                //--------------------------------------------------------------
                // save C(i,j)
                //--------------------------------------------------------------

                Cx [pC] = cij ;
            }
        }
    }
}
#endif

