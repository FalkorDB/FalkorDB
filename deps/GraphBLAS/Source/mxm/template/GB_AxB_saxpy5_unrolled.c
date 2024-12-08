//------------------------------------------------------------------------------
// GB_AxB_saxpy5_unrolled.c: C+=A*B when C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is full.
// A is full and not iso-valued nor pattern-only
// B is sparse or hypersparse.  It may be iso and/or pattern.

// The type of A and C must match the multiply op.  B can be typecasted.

// The monoid is identical to the accum op, and is not the ANY operator.
// The type of A must match the multiply operator input.
// The type of C must match the monoid/accum op.  B can be typecasted in
// the JIT version, but not for the FactoryKernel.

// This method is used for both Factory and JIT kernels.

#ifdef GB_GENERIC
#error "saxpy5 generic kernel undefined"
#endif

{

    //--------------------------------------------------------------------------
    // get C, A, and B
    //--------------------------------------------------------------------------

    const int64_t m = C->vlen ;     // # of rows of C and A
    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Bi = B->i ;
    #ifdef GB_JIT_KERNEL
    #define B_iso GB_B_ISO
    #else
    const bool B_iso = B->iso ;
    #endif
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    #if !GB_B_IS_PATTERN
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
    #endif
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;

    //--------------------------------------------------------------------------
    // define the vectors
    //--------------------------------------------------------------------------

    // GB_CIJ_MULTADD:  C(i,j) += A(i,k) * B(k,j)
    // the semiring is not positional (or A would be pattern-only), so the
    // i, k, j values are not needed
    #define GB_CIJ_MULTADD(cij,aik,bkj) \
        GB_MULTADD (cij, aik, bkj, ignore, ignore, ignore) ;

    #if GB_V16
    typedef GB_C_TYPE __attribute__ ((vector_size (16 * sizeof (GB_C_TYPE)))) v16 ;
    typedef GB_C_TYPE __attribute__ ((vector_size (16 * sizeof (GB_C_TYPE)), aligned (sizeof (GB_C_TYPE)))) v16u ;
    #endif

    #if GB_V16 || GB_V8
    typedef GB_C_TYPE __attribute__ ((vector_size (8 * sizeof (GB_C_TYPE)))) v8 ;
    typedef GB_C_TYPE __attribute__ ((vector_size (8 * sizeof (GB_C_TYPE)), aligned (sizeof (GB_C_TYPE)))) v8u ;
    #endif

    #if GB_V16 || GB_V8 || GB_V4
    typedef GB_C_TYPE __attribute__ ((vector_size (4 * sizeof (GB_C_TYPE)))) v4 ;
    typedef GB_C_TYPE __attribute__ ((vector_size (4 * sizeof (GB_C_TYPE)), aligned (sizeof (GB_C_TYPE)))) v4u ;
    typedef GB_C_TYPE __attribute__ ((vector_size (2 * sizeof (GB_C_TYPE)))) v2 ;
    typedef GB_C_TYPE __attribute__ ((vector_size (2 * sizeof (GB_C_TYPE)), aligned (sizeof (GB_C_TYPE)))) v2u ;
    #endif

    //--------------------------------------------------------------------------
    // C += A*B where A is full (and not iso or pattern-only)
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        #if ! ( GB_V16 || GB_V8 || GB_V4 )
        // get workspace
        GB_C_TYPE cx [16] ;
        #endif
        // get the task descriptor
        const int64_t jB_start = B_slice [tid] ;
        const int64_t jB_end   = B_slice [tid+1] ;
        // C(:,jB_start:jB_end-1) += A * B(:,jB_start:jB_end-1)
        for (int64_t jB = jB_start ; jB < jB_end ; jB++)
        {
            // get B(:,j) and C(:,j)
            const int64_t j = GBH_B (Bh, jB) ;
            GB_C_TYPE *restrict Cxj = Cx + (j * m) ;
            const int64_t pB_start = Bp [jB] ;
            const int64_t pB_end   = Bp [jB+1] ;

            //------------------------------------------------------------------
            // C(:,j) += A*B(:,j), on sets of 16 rows of C and A at a time
            //------------------------------------------------------------------

            for (int64_t i = 0 ; i < m - 15 ; i += 16)
            {
                // get C(i:i+15,j)
                #if GB_V16
                v16 c1 = (*((v16u *) (Cxj + i    ))) ;
                #elif GB_V8
                v8  c1 = (*((v8u  *) (Cxj + i    ))) ;
                v8  c2 = (*((v8u  *) (Cxj + i + 8))) ;
                #elif GB_V4
                v4  c1 = (*((v4u  *) (Cxj + i    ))) ;
                v4  c2 = (*((v4u  *) (Cxj + i + 4))) ;
                v4  c3 = (*((v4u  *) (Cxj + i + 8))) ;
                v4  c4 = (*((v4u  *) (Cxj + i +12))) ;
                #else
                memcpy (cx, Cxj + i, 16 * sizeof (GB_C_TYPE)) ;
                #endif
                // get A(i,0)
                const GB_A_TYPE *restrict Axi = Ax + i ;
                for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                { 
                    // bkj = B(k,j)
                    const int64_t k = Bi [pB] ;
                    GB_DECLAREB (bkj) ;
                    GB_GETB (bkj, Bx, pB, B_iso) ;
                    // get A(i,k)
                    const GB_A_TYPE *restrict ax = Axi + (k * m) ;
                    // C(i:i+15,j) += A(i:i+15,k)*B(k,j)
                    #if GB_V16
                    GB_CIJ_MULTADD (c1, (*((v16u *) (ax    ))), bkj) ;
                    #elif GB_V8
                    GB_CIJ_MULTADD (c1, (*((v8u  *) (ax    ))), bkj) ;
                    GB_CIJ_MULTADD (c2, (*((v8u  *) (ax + 8))), bkj) ;
                    #elif GB_V4
                    GB_CIJ_MULTADD (c1, (*((v4u  *) (ax    ))), bkj) ;
                    GB_CIJ_MULTADD (c2, (*((v4u  *) (ax + 4))), bkj) ;
                    GB_CIJ_MULTADD (c3, (*((v4u  *) (ax + 8))), bkj) ;
                    GB_CIJ_MULTADD (c4, (*((v4u  *) (ax +12))), bkj) ;
                    #else
                    GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                    GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                    GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                    GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                    GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                    GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                    GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                    GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                    GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                    GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                    GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                    GB_CIJ_MULTADD (cx [11], ax [11], bkj) ;
                    GB_CIJ_MULTADD (cx [12], ax [12], bkj) ;
                    GB_CIJ_MULTADD (cx [13], ax [13], bkj) ;
                    GB_CIJ_MULTADD (cx [14], ax [14], bkj) ;
                    GB_CIJ_MULTADD (cx [15], ax [15], bkj) ;
                    #endif
                }
                // save C(i:i+15,j)
                #if GB_V16
                (*((v16u *) (Cxj + i    ))) = c1 ;
                #elif GB_V8
                (*((v8u  *) (Cxj + i    ))) = c1 ;
                (*((v8u  *) (Cxj + i + 8))) = c2 ;
                #elif GB_V4
                (*((v4u  *) (Cxj + i    ))) = c1 ;
                (*((v4u  *) (Cxj + i + 4))) = c2 ;
                (*((v4u  *) (Cxj + i + 8))) = c3 ;
                (*((v4u  *) (Cxj + i +12))) = c4 ;
                #else
                memcpy (Cxj + i, cx, 16 * sizeof (GB_C_TYPE)) ;
                #endif
            }

            //------------------------------------------------------------------
            // C(m-N:m-1,j) += A(m-N:m-1,j)*B(:,j) for last 0 to 15 rows
            //------------------------------------------------------------------

            switch (m & 15)
            {

                //--------------------------------------------------------------
                // C(m-15:m-1,j) += A(m-15:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 15:
                    {
                        // load C(m-15:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 15 ;
                        #if GB_V16 || GB_V8
                        v8 c1 = (*((v8u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  8))) ;
                        #elif GB_V4
                        v4 c1 = (*((v4u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  4))) ;
                        v4 c3 = (*((v4u *) (Cxm +  8))) ;
                        #endif
                        #if GB_V16 || GB_V8 || GB_V4
                        v2 c4 = (*((v2u *) (Cxm + 12))) ;
                        GB_C_TYPE c5 = Cxm [14] ;
                        #else
                        memcpy (cx, Cxm, 15 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-15,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 15 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-15,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-15:m-1,j) += A(m-15:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8
                            GB_CIJ_MULTADD (c1, (*((v8u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 8))), bkj) ;
                            #elif GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 4))), bkj) ;
                            GB_CIJ_MULTADD (c3, (*((v4u *) (ax + 8))), bkj) ;
                            #endif
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c4, (*((v2u *) (ax +12))), bkj) ;
                            GB_CIJ_MULTADD (c5, ax [14], bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                            GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                            GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                            GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                            GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                            GB_CIJ_MULTADD (cx [11], ax [11], bkj) ;
                            GB_CIJ_MULTADD (cx [12], ax [12], bkj) ;
                            GB_CIJ_MULTADD (cx [13], ax [13], bkj) ;
                            GB_CIJ_MULTADD (cx [14], ax [14], bkj) ;
                            #endif
                        }
                        // save C(m-15:m-1,j)
                        #if GB_V16 || GB_V8
                        (*((v8u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  8))) = c2 ;
                        #elif GB_V4
                        (*((v4u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  4))) = c2 ;
                        (*((v4u *) (Cxm +  8))) = c3 ;
                        #endif
                        #if GB_V16 || GB_V8 || GB_V4
                        (*((v2u *) (Cxm + 12))) = c4 ;
                        Cxm [14] = c5 ;
                        #else
                        memcpy (Cxm, cx, 15 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-14:m-1,j) += A(m-14:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 14:
                    {
                        // load C(m-14:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 14 ;
                        #if GB_V16 || GB_V8
                        v8 c1 = (*((v8u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  8))) ;
                        #elif GB_V4
                        v4 c1 = (*((v4u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  4))) ;
                        v4 c3 = (*((v4u *) (Cxm +  8))) ;
                        #endif
                        #if GB_V16 || GB_V8 || GB_V4
                        v2 c4 = (*((v2u *) (Cxm + 12))) ;
                        #else
                        memcpy (cx, Cxm, 14 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-14,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 14 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-14,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-14:m-1,j) += A(m-14:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8
                            GB_CIJ_MULTADD (c1, (*((v8u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 8))), bkj) ;
                            #elif GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 4))), bkj) ;
                            GB_CIJ_MULTADD (c3, (*((v4u *) (ax + 8))), bkj) ;
                            #endif
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c4, (*((v2u *) (ax +12))), bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                            GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                            GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                            GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                            GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                            GB_CIJ_MULTADD (cx [11], ax [11], bkj) ;
                            GB_CIJ_MULTADD (cx [12], ax [12], bkj) ;
                            GB_CIJ_MULTADD (cx [13], ax [13], bkj) ;
                            #endif
                        }
                        // save C(m-14:m-1,j)
                        #if GB_V16 || GB_V8
                        (*((v8u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  8))) = c2 ;
                        #elif GB_V4
                        (*((v4u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  4))) = c2 ;
                        (*((v4u *) (Cxm +  8))) = c3 ;
                        #endif
                        #if GB_V16 || GB_V8 || GB_V4
                        (*((v2u *) (Cxm + 12))) = c4 ;
                        #else
                        memcpy (Cxm, cx, 14 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-13:m-1,j) += A(m-13:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 13:
                    {
                        // load C(m-13:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 13 ;
                        #if GB_V16 || GB_V8
                        v8 c1 = (*((v8u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  8))) ;
                        #elif GB_V4
                        v4 c1 = (*((v4u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  4))) ;
                        v4 c3 = (*((v4u *) (Cxm +  8))) ;
                        #endif
                        #if GB_V16 || GB_V8 || GB_V4
                        GB_C_TYPE c4 = Cxm [12] ;
                        #else
                        memcpy (cx, Cxm, 13 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-13,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 13 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-13,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-13:m-1,j) += A(m-13:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8
                            GB_CIJ_MULTADD (c1, (*((v8u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 8))), bkj) ;
                            #elif GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 4))), bkj) ;
                            GB_CIJ_MULTADD (c3, (*((v4u *) (ax + 8))), bkj) ;
                            #endif
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c4, ax [12], bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                            GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                            GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                            GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                            GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                            GB_CIJ_MULTADD (cx [11], ax [11], bkj) ;
                            GB_CIJ_MULTADD (cx [12], ax [12], bkj) ;
                            #endif
                        }
                        // save C(m-13:m-1,j)
                        #if GB_V16 || GB_V8
                        (*((v8u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  8))) = c2 ;
                        #elif GB_V4
                        (*((v4u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  4))) = c2 ;
                        (*((v4u *) (Cxm +  8))) = c3 ;
                        #endif
                        #if GB_V16 || GB_V8 || GB_V4
                        Cxm [12] = c4 ;
                        #else
                        memcpy (Cxm, cx, 13 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-12:m-1,j) += A(m-12:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 12:
                    {
                        // load C(m-12:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 12 ;
                        #if GB_V16 || GB_V8
                        v8 c1 = (*((v8u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  8))) ;
                        #elif GB_V4
                        v4 c1 = (*((v4u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  4))) ;
                        v4 c3 = (*((v4u *) (Cxm +  8))) ;
                        #else
                        memcpy (cx, Cxm, 12 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-12,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 12 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-12,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-12:m-1,j) += A(m-12:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8
                            GB_CIJ_MULTADD (c1, (*((v8u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 8))), bkj) ;
                            #elif GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 4))), bkj) ;
                            GB_CIJ_MULTADD (c3, (*((v4u *) (ax + 8))), bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                            GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                            GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                            GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                            GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                            GB_CIJ_MULTADD (cx [11], ax [11], bkj) ;
                            #endif
                        }
                        // save C(m-12:m-1,j)
                        #if GB_V16 || GB_V8
                        (*((v8u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  8))) = c2 ;
                        #elif GB_V4
                        (*((v4u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  4))) = c2 ;
                        (*((v4u *) (Cxm +  8))) = c3 ;
                        #else
                        memcpy (Cxm, cx, 12 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-11:m-1,j) += A(m-11:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 11:
                    {
                        // load C(m-11:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 11 ;
                        #if GB_V16 || GB_V8
                        v8 c1 = (*((v8u *) (Cxm     ))) ;
                        v2 c2 = (*((v2u *) (Cxm +  8))) ;
                        #elif GB_V4
                        v4 c1 = (*((v4u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  4))) ;
                        v2 c3 = (*((v2u *) (Cxm +  8))) ;
                        #endif
                        #if GB_V16 || GB_V8 || GB_V4
                        GB_C_TYPE c4 = Cxm [10] ;
                        #else
                        memcpy (cx, Cxm, 11 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-11,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 11 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-11,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-11:m-1,j) += A(m-11:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8
                            GB_CIJ_MULTADD (c1, (*((v8u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v2u *) (ax + 8))), bkj) ;
                            #elif GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 4))), bkj) ;
                            GB_CIJ_MULTADD (c3, (*((v2u *) (ax + 8))), bkj) ;
                            #endif
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c4, ax [10], bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                            GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                            GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                            GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                            GB_CIJ_MULTADD (cx [10], ax [10], bkj) ;
                            #endif
                        }
                        // save C(m-11:m-1,j)
                        #if GB_V16 || GB_V8
                        (*((v8u *) (Cxm     ))) = c1 ;
                        (*((v2u *) (Cxm +  8))) = c2 ;
                        #elif GB_V4
                        (*((v4u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  4))) = c2 ;
                        (*((v2u *) (Cxm +  8))) = c3 ;
                        #endif
                        #if GB_V16 || GB_V8 || GB_V4
                        Cxm [10] = c4 ;
                        #else
                        memcpy (Cxm, cx, 11 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-10:m-1,j) += A(m-10:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 10:
                    {
                        // load C(m-10:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 10 ;
                        #if GB_V16 || GB_V8
                        v8 c1 = (*((v8u *) (Cxm     ))) ;
                        v2 c2 = (*((v2u *) (Cxm +  8))) ;
                        #elif GB_V4
                        v4 c1 = (*((v4u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  4))) ;
                        v2 c3 = (*((v2u *) (Cxm +  8))) ;
                        #else
                        memcpy (cx, Cxm, 10 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-10,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 10 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-10,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-10:m-1,j) += A(m-10:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8
                            GB_CIJ_MULTADD (c1, (*((v8u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v2u *) (ax + 8))), bkj) ;
                            #elif GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 4))), bkj) ;
                            GB_CIJ_MULTADD (c3, (*((v2u *) (ax + 8))), bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                            GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                            GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                            GB_CIJ_MULTADD (cx [ 9], ax [ 9], bkj) ;
                            #endif
                        }
                        // save C(m-10:m-1,j)
                        #if GB_V16 || GB_V8
                        (*((v8u *) (Cxm     ))) = c1 ;
                        (*((v2u *) (Cxm +  8))) = c2 ;
                        #elif GB_V4
                        (*((v4u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  4))) = c2 ;
                        (*((v2u *) (Cxm +  8))) = c3 ;
                        #else
                        memcpy (Cxm, cx, 10 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-9:m-1,j) += A(m-9:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 9:
                    {
                        // load C(m-9:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 9 ;
                        #if GB_V16 || GB_V8
                        v8 c1 = (*((v8u *) (Cxm     ))) ;
                        #elif GB_V4
                        v4 c1 = (*((v4u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  4))) ;
                        #endif
                        #if GB_V16 || GB_V8 || GB_V4
                        GB_C_TYPE c3 = Cxm [8] ;
                        #else
                        memcpy (cx, Cxm, 9 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-9,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 9 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-9,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-9:m-1,j) += A(m-9:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8
                            GB_CIJ_MULTADD (c1, (*((v8u *) (ax    ))), bkj) ;
                            #elif GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 4))), bkj) ;
                            #endif
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c3, ax [ 8], bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                            GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                            GB_CIJ_MULTADD (cx [ 8], ax [ 8], bkj) ;
                            #endif
                        }
                        // save C(m-9:m-1,j)
                        #if GB_V16 || GB_V8
                        (*((v8u *) (Cxm     ))) = c1 ;
                        #elif GB_V4
                        (*((v4u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  4))) = c2 ;
                        #endif
                        #if GB_V16 || GB_V8 || GB_V4
                        Cxm [8] = c3 ;
                        #else
                        memcpy (Cxm, cx, 9 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-8:m-1,j) += A(m-8:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 8:
                    {
                        // load C(m-8:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 8 ;
                        #if GB_V16 || GB_V8
                        v8 c1 = (*((v8u *) (Cxm     ))) ;
                        #elif GB_V4
                        v4 c1 = (*((v4u *) (Cxm     ))) ;
                        v4 c2 = (*((v4u *) (Cxm +  4))) ;
                        #else
                        memcpy (cx, Cxm, 8 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-8,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 8 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-8,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-8:m-1,j) += A(m-8:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8
                            GB_CIJ_MULTADD (c1, (*((v8u *) (ax    ))), bkj) ;
                            #elif GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v4u *) (ax + 4))), bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                            GB_CIJ_MULTADD (cx [ 7], ax [ 7], bkj) ;
                            #endif
                        }
                        // save C(m-8:m-1,j)
                        #if GB_V16 || GB_V8
                        (*((v8u *) (Cxm     ))) = c1 ;
                        #elif GB_V4
                        (*((v4u *) (Cxm     ))) = c1 ;
                        (*((v4u *) (Cxm +  4))) = c2 ;
                        #else
                        memcpy (Cxm, cx, 8 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-7:m-1,j) += A(m-7:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 7:
                    {
                        // load C(m-7:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 7 ;
                        #if GB_V16 || GB_V8 || GB_V4
                        v4 c1 = (*((v4u *) (Cxm     ))) ;
                        v2 c2 = (*((v2u *) (Cxm +  4))) ;
                        GB_C_TYPE c3 = Cxm [6] ;
                        #else
                        memcpy (cx, Cxm, 7 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-7,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 7 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-7,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-7:m-1,j) += A(m-7:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v2u *) (ax + 4))), bkj) ;
                            GB_CIJ_MULTADD (c3, ax [ 6], bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            GB_CIJ_MULTADD (cx [ 6], ax [ 6], bkj) ;
                            #endif
                        }
                        // save C(m-7:m-1,j)
                        #if GB_V16 || GB_V8 || GB_V4
                        (*((v4u *) (Cxm     ))) = c1 ;
                        (*((v2u *) (Cxm +  4))) = c2 ;
                        Cxm [6] = c3 ;
                        #else
                        memcpy (Cxm, cx, 7 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-6:m-1,j) += A(m-6:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 6:
                    {
                        // load C(m-6:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 6 ;
                        #if GB_V16 || GB_V8 || GB_V4
                        v4 c1 = (*((v4u *) (Cxm     ))) ;
                        v2 c2 = (*((v2u *) (Cxm +  4))) ;
                        #else
                        memcpy (cx, Cxm, 6 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-6,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 6 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-6,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-6:m-1,j) += A(m-6:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, (*((v2u *) (ax + 4))), bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            GB_CIJ_MULTADD (cx [ 5], ax [ 5], bkj) ;
                            #endif
                        }
                        // save C(m-6:m-1,j)
                        #if GB_V16 || GB_V8 || GB_V4
                        (*((v4u *) (Cxm     ))) = c1 ;
                        (*((v2u *) (Cxm +  4))) = c2 ;
                        #else
                        memcpy (Cxm, cx, 6 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-5:m-1,j) += A(m-5:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 5:
                    {
                        // load C(m-5:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 5 ;
                        #if GB_V16 || GB_V8 || GB_V4
                        v4 c1 = (*((v4u *) (Cxm))) ;
                        GB_C_TYPE c2 = Cxm [4] ;
                        #else
                        memcpy (cx, Cxm, 5 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-5,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 5 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-5,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-5:m-1,j) += A(m-5:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) (ax    ))), bkj) ;
                            GB_CIJ_MULTADD (c2, ax [ 4], bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            GB_CIJ_MULTADD (cx [ 4], ax [ 4], bkj) ;
                            #endif
                        }
                        // save C(m-5:m-1,j)
                        #if GB_V16 || GB_V8 || GB_V4
                        (*((v4u *) (Cxm))) = c1 ;
                        Cxm [4] = c2 ;
                        #else
                        memcpy (Cxm, cx, 5 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-4:m-1,j) += A(m-4:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 4:
                    {
                        // load C(m-4:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 4 ;
                        #if GB_V16 || GB_V8 || GB_V4
                        v4 c1 = (*((v4u *) (Cxm))) ;
                        #else
                        memcpy (cx, Cxm, 4 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-4,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 4 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-4,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-4:m-1,j) += A(m-4:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c1, (*((v4u *) ax)), bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            GB_CIJ_MULTADD (cx [ 3], ax [ 3], bkj) ;
                            #endif
                        }
                        // save C(m-4:m-1,j)
                        #if GB_V16 || GB_V8 || GB_V4
                        (*((v4u *) (Cxm))) = c1 ;
                        #else
                        memcpy (Cxm, cx, 4 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-3:m-1,j) += A(m-3:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 3:
                    {
                        // load C(m-3:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 3 ;
                        #if GB_V16 || GB_V8 || GB_V4
                        v2 c1 = (*((v2u *) (Cxm))) ;
                        GB_C_TYPE c2 = Cxm [2] ;
                        #else
                        memcpy (cx, Cxm, 3 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-3,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 3 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-3,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-3:m-1,j) += A(m-3:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c1, (*((v2u *) ax)), bkj) ;
                            GB_CIJ_MULTADD (c2, ax [ 2], bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            GB_CIJ_MULTADD (cx [ 2], ax [ 2], bkj) ;
                            #endif
                        }
                        // save C(m-3:m-1,j)
                        #if GB_V16 || GB_V8 || GB_V4
                        (*((v2u *) (Cxm))) = c1 ;
                        Cxm [2] = c2 ;
                        #else
                        memcpy (Cxm, cx, 3 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-2:m-1,j) += A(m-2:m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 2:
                    {
                        // load C(m-2:m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 2 ;
                        #if GB_V16 || GB_V8 || GB_V4
                        v2 c1 = (*((v2u *) (Cxm))) ;
                        #else
                        memcpy (cx, Cxm, 2 * sizeof (GB_C_TYPE)) ;
                        #endif
                        // get A(m-2,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 2 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-2,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-2:m-1,j) += A(m-2:m-1,k)*B(k,j)
                            #if GB_V16 || GB_V8 || GB_V4
                            GB_CIJ_MULTADD (c1, (*((v2u *) ax)), bkj) ;
                            #else
                            GB_CIJ_MULTADD (cx [ 0], ax [ 0], bkj) ;
                            GB_CIJ_MULTADD (cx [ 1], ax [ 1], bkj) ;
                            #endif
                        }
                        // save C(m-2:m-1,j)
                        #if GB_V16 || GB_V8 || GB_V4
                        (*((v2u *) (Cxm))) = c1 ;
                        #else
                        memcpy (Cxm, cx, 2 * sizeof (GB_C_TYPE)) ;
                        #endif
                    }
                    break ;

                //--------------------------------------------------------------
                // C(m-1,j) += A(m-1,j)*B(:,j)
                //--------------------------------------------------------------

                case 1:
                    {
                        // load C(m-1,j)
                        GB_C_TYPE *restrict Cxm = Cxj + m - 1 ;
                        GB_C_TYPE c1 = Cxm [0] ;
                        // get A(m-1,0)
                        const GB_A_TYPE *restrict Axm = Ax + m - 1 ;
                        for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                        { 
                            // bkj = B(k,j)
                            const int64_t k = Bi [pB] ;
                            GB_DECLAREB (bkj) ;
                            GB_GETB (bkj, Bx, pB, B_iso) ;
                            // get A(m-1,k)
                            const GB_A_TYPE *restrict ax = Axm + (k * m) ;
                            // C(m-1,j) += A(m-1,k)*B(k,j)
                            GB_CIJ_MULTADD (c1, ax [0], bkj) ;
                        }
                        // save C(m-1,j)
                        Cxm [0] = c1 ;
                    }
                    break ;

                default:
                    break ;
            }
        }
    }
}

#undef GB_V16
#undef GB_V8
#undef GB_V4
#undef GB_CIJ_MULTADD

