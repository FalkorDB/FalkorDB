//------------------------------------------------------------------------------
// GB_AxB_saxpy4_template: C+=A*B, C is full, A is sparse/hyper, B bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is full. A is hyper/sparse, B is bitmap/full.  M is not present.

// C += A*B is computed with the accumulator identical to the monoid.

// This template is used by mxm/template/GB_AxB_saxpy4_meta.  It is not used
// for the generic case, nor for the ANY_PAIR case.  It is only used for the
// pre-generated kernels, and for the JIT.

#ifndef GB_B_SIZE
#define GB_B_SIZE sizeof (GB_B_TYPE)
#endif

#ifndef GB_C_SIZE
#define GB_C_SIZE sizeof (GB_C_TYPE)
#endif

#ifndef GB_C_ISO
#define GB_C_ISO 0
#endif

{

    if (use_coarse_tasks)
    {

        //----------------------------------------------------------------------
        // C += A*B using coarse tasks
        //----------------------------------------------------------------------

        int64_t cvlenx = cvlen * GB_C_SIZE ;

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // determine the vectors of B and C for this coarse task
            //------------------------------------------------------------------

            int64_t jstart, jend ;
            GB_PARTITION (jstart, jend, bvdim, tid, ntasks) ;
            int64_t jtask = jend - jstart ;
            int64_t jpanel = GB_IMIN (jtask, GB_SAXPY4_PANEL_SIZE) ;

            //------------------------------------------------------------------
            // get the workspace for this task
            //------------------------------------------------------------------

            // Hx workspace to compute the panel of C
            GB_C_TYPE *restrict Hx = (GB_C_TYPE *)
                (Wcx + H_slice [tid] * cvlenx) ;

            //------------------------------------------------------------------
            // C(:,jstart:jend-1) += A * B(:,jstart:jend-1) by panel
            //------------------------------------------------------------------

            for (int64_t j1 = jstart ; j1 < jend ; j1 += jpanel)
            {

                //--------------------------------------------------------------
                // get the panel of np vectors j1:j2-1
                //--------------------------------------------------------------

                int64_t j2 = GB_IMIN (jend, j1 + jpanel) ;
                int64_t np = j2 - j1 ;

                //--------------------------------------------------------------
                // G = B(:,j1:j2-1), of size bvlen-by-np, in column major order
                //--------------------------------------------------------------

                int8_t *restrict Gb = (int8_t *) (Bb + (j1 * bvlen)) ;
                #if !GB_B_IS_PATTERN
                GB_B_TYPE *restrict Gx = (GB_B_TYPE *)
                     (((GB_void *) (B->x)) +
                       (B_iso ? 0 : ((j1 * bvlen) * GB_B_SIZE))) ;
                #endif

                //--------------------------------------------------------------
                // clear the panel H to compute C(:,j1:j2-1)
                //--------------------------------------------------------------

                if (np == 1)
                { 
                    // Make H an alias to C(:,j1)
                    int64_t j = j1 ;
                    int64_t pC_start = j * cvlen ;    // get pointer to C(:,j)
                    // Hx is GB_C_TYPE, not GB_void, so pointer arithmetic on
                    // it is by units of size sizeof (GB_C_TYPE), not bytes.
                    Hx = Cx + pC_start ;
                }
                else
                { 
                    // C is full: set Hx = identity
                    int64_t nc = np * cvlen ;
                    #if GB_HAS_IDENTITY_BYTE
                        memset (Hx, GB_IDENTITY_BYTE, nc * GB_C_SIZE) ;
                    #else
                        for (int64_t i = 0 ; i < nc ; i++)
                        { 
                            GB_HX_WRITE (i, zidentity) ; // Hx(i) = identity
                        }
                    #endif
                }

                #if GB_IS_PLUS_FC32_MONOID
                float  *restrict Hx_real = (float *) Hx ;
                float  *restrict Hx_imag = Hx_real + 1 ;
                #elif GB_IS_PLUS_FC64_MONOID
                double *restrict Hx_real = (double *) Hx ;
                double *restrict Hx_imag = Hx_real + 1 ;
                #endif

                //--------------------------------------------------------------
                // H += A*G for one panel
                //--------------------------------------------------------------

                // GB_HX_COMPUTE:  computes H (i,jj) += A(i,k) * B(k,j)
                // where H(i,jj) is located at Hx [pH+jj], aij = A(i,k),
                // and gkj = B(k,j).  gb is the bitmap value for B(k,j).

                #undef GB_HX_COMPUTE
                #if GB_B_IS_BITMAP
                    #define GB_HX_COMPUTE(pH,i,gkj,gb,jj)                   \
                    if (gb)                                                 \
                    {                                                       \
                        GB_MULTADD2 (Hx, pH+jj, aik, gkj, i, k, j1+jj) ;    \
                    }
                #else
                    #define GB_HX_COMPUTE(pH,i,gkj,gb,jj)                   \
                    {                                                       \
                        GB_MULTADD2 (Hx, pH+jj, aik, gkj, i, k, j1+jj) ;    \
                    }
                #endif

                #include "template/GB_AxB_saxpy4_panel.c"
                #undef GB_HX_COMPUTE

                //--------------------------------------------------------------
                // C(:,j1:j2-1) = H
                //--------------------------------------------------------------

                if (np == 1)
                { 
                    // Hx is already aliased to Cx; no more work to do
                    continue ;
                }

                for (int64_t jj = 0 ; jj < np ; jj++)
                {

                    //----------------------------------------------------------
                    // C(:,j) = H (:,jj)
                    //----------------------------------------------------------

                    int64_t j = j1 + jj ;
                    int64_t pC_start = j * cvlen ;  // get pointer to C(:,j)

                    for (int64_t i = 0 ; i < cvlen ; i++)
                    { 
                        int64_t pC = pC_start + i ;     // pointer to C(i,j)
                        int64_t pH = i * np + jj ;      // pointer to H(i,jj)

                        // C(i,j) = H(i,jj)
                        GB_CIJ_GATHER_UPDATE (pC, pH) ;
                    }
                }
            }
        }

    }
    else if (use_atomics)
    {

        //----------------------------------------------------------------------
        // C += A*B using fine tasks and atomics
        //----------------------------------------------------------------------

        // allocate workspace to implement the atomic update
        #if !GB_Z_HAS_ATOMIC_UPDATE
        int8_t *restrict Wf = NULL ; size_t Wf_size = 0 ;
        Wf = GB_CALLOC_MEMORY (C->vlen * C->vdim, sizeof (int8_t), &Wf_size) ;
        if (Wf == NULL)
        { 
            return (GrB_OUT_OF_MEMORY) ;
        }
        #endif

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // determine the vector of B and C for this fine task
            //------------------------------------------------------------------

            // The fine task operates on C(:,j) and B(:,j).  Its fine task
            // id ranges from 0 to nfine_tasks_per_vector-1, and determines
            // which slice of A to operate on.

            int64_t j    = tid / nfine_tasks_per_vector ;
            int fine_tid = tid % nfine_tasks_per_vector ;
            int64_t kfirst = A_slice [fine_tid] ;
            int64_t klast = A_slice [fine_tid + 1] ;
            int64_t pB_start = j * bvlen ;      // pointer to B(:,j)
            int64_t pC_start = j * cvlen ;      // pointer to C(:,j)
            GB_GET_T_FOR_SECONDJ ;              // t = j or j+1 for SECONDJ*

            // for Hx Gustavason workspace: use C(:,j) in-place:
            GB_C_TYPE *restrict Hx = (GB_C_TYPE *)
                (((GB_void *) Cx) + (pC_start * GB_C_SIZE)) ;
            #if !GB_Z_HAS_ATOMIC_UPDATE
            int8_t *restrict Hf = Wf + pC_start ;
            #endif
            #if GB_IS_PLUS_FC32_MONOID
            float  *restrict Hx_real = (float *) Hx ;
            float  *restrict Hx_imag = Hx_real + 1 ;
            #elif GB_IS_PLUS_FC64_MONOID
            double *restrict Hx_real = (double *) Hx ;
            double *restrict Hx_imag = Hx_real + 1 ;
            #endif

            //------------------------------------------------------------------
            // C(:,j) += A(:,k1:k2) * B(k1:k2,j)
            //------------------------------------------------------------------

            for (int64_t kk = kfirst ; kk < klast ; kk++)
            {

                //--------------------------------------------------------------
                // C(:,j) += A(:,k) * B(k,j)
                //--------------------------------------------------------------

                int64_t k = GBh_A (Ah, kk) ;      // k in range k1:k2
                int64_t pB = pB_start + k ;     // get pointer to B(k,j)
                #if GB_B_IS_BITMAP
                if (!GBb_B (Bb, pB)) continue ;   
                #endif
                int64_t pA = GB_IGET (Ap, kk) ;
                int64_t pA_end = GB_IGET (Ap, kk+1) ;
                GB_GET_B_kj ;                   // bkj = B(k,j)

                for ( ; pA < pA_end ; pA++)
                {

                    //----------------------------------------------------------
                    // get A(i,k)
                    //----------------------------------------------------------

                    int64_t i = GB_IGET (Ai, pA) ;       // get A(i,k) index

                    //----------------------------------------------------------
                    // C(i,j) += A(i,k) * B(k,j)
                    //----------------------------------------------------------

                    GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)

                    #if GB_Z_HAS_ATOMIC_UPDATE
                    { 
                        // the monoid has an atomic update
                        GB_Z_ATOMIC_UPDATE_HX (i, t) ;    // C(i,j) += t
                    }
                    #else
                    { 
                        // the update must be done in a critical section using
                        // the mutex byte Hf (i,j), located at Hf [i].  If
                        // zero, the mutex is unlocked.
                        int8_t f ;
                        do
                        {
                            // do this atomically:
                            // { f = Hf [i] ; Hf [i] = 1 ; }
                            GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 1) ;
                        }
                        while (f == 1) ;
                        GB_Z_ATOMIC_UPDATE_HX (i, t) ;    // C(i,j) += t
                        GB_ATOMIC_WRITE
                        Hf [i] = 0 ;
                    }
                    #endif
                }
            }
        }

        // free workspace for atomic update
        #if !GB_Z_HAS_ATOMIC_UPDATE
        GB_FREE_MEMORY (&Wf, Wf_size) ;
        #endif

    }
    else
    {

        //----------------------------------------------------------------------
        // C += A*B using fine tasks and workspace, with no atomics
        //----------------------------------------------------------------------

        // Each fine task is given size-cvlen workspace to compute its result
        // in the first phase, W(:,tid) = A(:,k1:k2) * B(k1:k2,j), where k1:k2
        // is defined by the fine_tid of the task.  The workspaces are then
        // summed into C in the second phase.

        //----------------------------------------------------------------------
        // first phase: W (:,tid) = A (:,k1:k2) * B (k2:k2,j) for each fine task
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // determine the vector of B and C for this fine task
            //------------------------------------------------------------------

            // The fine task operates on C(:,j) and B(:,j).  Its fine task
            // id ranges from 0 to nfine_tasks_per_vector-1, and determines
            // which slice of A to operate on.

            int64_t j    = tid / nfine_tasks_per_vector ;
            int fine_tid = tid % nfine_tasks_per_vector ;
            int64_t kfirst = A_slice [fine_tid] ;
            int64_t klast = A_slice [fine_tid + 1] ;
            int64_t pB_start = j * bvlen ;      // pointer to B(:,j)
            int64_t pW_start = tid * cvlen ;    // pointer to W for this thread
            GB_GET_T_FOR_SECONDJ ;              // t = j or j+1 for SECONDJ*

            GB_C_TYPE *restrict Hx = (GB_C_TYPE *)
                (Wcx + (pW_start * GB_C_SIZE)) ;
            #if GB_IS_PLUS_FC32_MONOID
            float  *restrict Hx_real = (float *) Hx ;
            float  *restrict Hx_imag = Hx_real + 1 ;
            #elif GB_IS_PLUS_FC64_MONOID
            double *restrict Hx_real = (double *) Hx ;
            double *restrict Hx_imag = Hx_real + 1 ;
            #endif

            //------------------------------------------------------------------
            // clear the panel
            //------------------------------------------------------------------

            // Hx = identity
            #if GB_HAS_IDENTITY_BYTE
            { 
                memset (Hx, GB_IDENTITY_BYTE, cvlen * GB_C_SIZE) ;
            }
            #else
            {
                for (int64_t i = 0 ; i < cvlen ; i++)
                { 
                    GB_HX_WRITE (i, zidentity) ; // Hx(i) = identity
                }
            }
            #endif

            //------------------------------------------------------------------
            // W = A(:,k1:k2) * B(k1:k2,j)
            //------------------------------------------------------------------

            for (int64_t kk = kfirst ; kk < klast ; kk++)
            {

                //--------------------------------------------------------------
                // W(:,k) += A(:,k) * B(k,j)
                //--------------------------------------------------------------

                int64_t k = GBh_A (Ah, kk) ;    // k in range k1:k2
                int64_t pB = pB_start + k ;     // get pointer to B(k,j)
                #if GB_B_IS_BITMAP
                if (!GBb_B (Bb, pB)) continue ;   
                #endif
                int64_t pA = GB_IGET (Ap, kk) ;
                int64_t pA_end = GB_IGET (Ap, kk+1) ;
                GB_GET_B_kj ;                   // bkj = B(k,j)

                for ( ; pA < pA_end ; pA++)
                { 
                    int64_t i = GB_IGET (Ai, pA) ;       // get A(i,k) index
                    // W(i,k) += A(i,k) * B(k,j)
                    GB_MULT_A_ik_B_kj ;         // t = A(i,k)*B(k,j)
                    GB_HX_UPDATE (i, t) ;       // Hx(i) += t
                }
            }
        }

        //----------------------------------------------------------------------
        // second phase: C += reduce (W)
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // determine the W and C for this fine task
            //------------------------------------------------------------------

            // The fine task operates on C(i1:i2,j) and W(i1:i2,w1:w2), where
            // i1:i2 is defined by the fine task id.  Its fine task id ranges
            // from 0 to nfine_tasks_per_vector-1.
            
            // w1:w2 are the updates to C(:,j), where w1:w2 =
            // [j*nfine_tasks_per_vector : (j+1)*nfine_tasks_per_vector-1].

            int64_t j    = tid / nfine_tasks_per_vector ;
            int fine_tid = tid % nfine_tasks_per_vector ;
            int64_t istart, iend ;
            GB_PARTITION (istart, iend, cvlen, fine_tid,
                nfine_tasks_per_vector) ;
            int64_t pC_start = j * cvlen ;          // pointer to C(:,j)
            int64_t wstart = j * nfine_tasks_per_vector ;
            int64_t wend = (j + 1) * nfine_tasks_per_vector ;

            // Hx = (typecasted) Wcx workspace
            GB_C_TYPE *restrict Hx = ((GB_C_TYPE *) Wcx) ;
            #if GB_IS_PLUS_FC32_MONOID
            float  *restrict Hx_real = (float *) Hx ;
            float  *restrict Hx_imag = Hx_real + 1 ;
            #elif GB_IS_PLUS_FC64_MONOID
            double *restrict Hx_real = (double *) Hx ;
            double *restrict Hx_imag = Hx_real + 1 ;
            #endif

            //------------------------------------------------------------------
            // C(i1:i2,j) += reduce (W (i2:i2, wstart:wend))
            //------------------------------------------------------------------

            for (int64_t w = wstart ; w < wend ; w++)
            {

                //--------------------------------------------------------------
                // C(i1:i2,j) += W (i1:i2,w)
                //--------------------------------------------------------------
            
                int64_t pW_start = w * cvlen ;      // pointer to W (:,w)

                for (int64_t i = istart ; i < iend ; i++)
                { 
                    int64_t pW = pW_start + i ;     // pointer to W(i,w)
                    int64_t pC = pC_start + i ;     // pointer to C(i,j)
                    // C(i,j) += W(i,w)
                    GB_CIJ_GATHER_UPDATE (pC, pW) ;
                }
            }
        }
    }
}

