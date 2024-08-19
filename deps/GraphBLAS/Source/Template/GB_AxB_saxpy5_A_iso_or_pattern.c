//------------------------------------------------------------------------------
// GB_AxB_saxpy5_A_iso_or_pattern.c: C+=A*B; C full, A bitmap/full & iso/pattern
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is full.
// A is bitmap or full, and either iso or pattern-only
// B is sparse or hypersparse.

// The monoid is identical to the accum op, and is not the ANY operator.
// The type of A must match the multiply operator input.
// The type of C must match the monoid/accum op.  B can be typecasted for a
// JIT kernel but not the FactoryKernels.

// This method is used for both built-in semirings with no typecasting, in
// the FactoryKernels, and the JIT kernels.

#ifdef GB_GENERIC
#error "saxpy5 generic kernel undefined"
#endif

{
    //--------------------------------------------------------------------------
    // get C, A, and B
    //--------------------------------------------------------------------------

    const int64_t m = C->vlen ;     // # of rows of C and A
    #if GB_A_IS_BITMAP
    const int8_t  *restrict Ab = A->b ;
    #endif
    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Bi = B->i ;
    #ifdef GB_JIT_KERNEL
    #define B_iso GB_B_ISO
    #else
    const bool B_iso = B->iso ;
    #endif
    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
    #endif
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;

    //--------------------------------------------------------------------------
    // C += A*B where A is bitmap/full, and either iso-valued or pattern-only
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        #if !GB_A_IS_PATTERN
        // get the iso value of A (no typecast required)
        const GB_A_TYPE a = Ax [0] ;
        #endif
        // get the task descriptor
        const int64_t jB_start = B_slice [tid] ;
        const int64_t jB_end   = B_slice [tid+1] ;
        // C(:,jB_start:jB_end-1) += A * B(:,jB_start:jB_end-1)
        for (int64_t jB = jB_start ; jB < jB_end ; jB++)
        {
            // get B(:,j) and C(:,j)
            const int64_t j = GBH_B (Bh, jB) ;
            const int64_t pC = j * m ;
            const int64_t pB_start = Bp [jB] ;
            const int64_t pB_end   = Bp [jB+1] ;
            // C(:,j) += A*B(:,j)
            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
            {
                // get B(k,j)
                const int64_t k = Bi [pB] ;
                #if GB_A_IS_BITMAP
                // get A(:,k)
                const int64_t pA = k * m ;
                #endif
                #if GB_IS_FIRSTI_MULTIPLIER
                    // s depends on i
                    #define s (i + GB_OFFSET)
                #else
                    // s = a * bkj, not dependent on i
                    GB_C_TYPE s ;
                    GB_DECLAREB (bkj) ;
                    GB_GETB (bkj, Bx, pB, B_iso) ;
                    GB_MULT (s, a, bkj, ignore, k, j) ;
                #endif
                // C(:,j) += s
                for (int64_t i = 0 ; i < m ; i++)
                { 
                    #if GB_A_IS_BITMAP
                    if (!Ab [pA + i]) continue ;
                    #endif
                    // C(i,j) += s ;
                    GB_CIJ_UPDATE (pC + i, s) ;
                }
            }
        }
    }
}

#undef s

