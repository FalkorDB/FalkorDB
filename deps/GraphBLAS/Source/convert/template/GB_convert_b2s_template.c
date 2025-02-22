//------------------------------------------------------------------------------
// GB_convert_b2s_template: construct triplets or CSC/CSR from bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    const int64_t avdim = A->vdim ;
    const int64_t avlen = A->vlen ;
    const int8_t *restrict Ab = A->b ;
    const GB_Cp_TYPE *restrict W  = W_input ;
    const GB_Cp_TYPE *restrict Cp = Cp_input ;
          GB_Ci_TYPE *restrict Ci = Ci_input ;
          GB_Cj_TYPE *restrict Cj = Cj_input ;
    #endif

    #ifdef GB_A_TYPE
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) Cx_new ;
    #endif

    //--------------------------------------------------------------------------
    // convert A from bitmap to C sparse (Cp, Ci, Cj, and Cx)
    //--------------------------------------------------------------------------

    if (W == NULL)
    {

        //----------------------------------------------------------------------
        // construct all vectors in parallel (no workspace)
        //----------------------------------------------------------------------

        int64_t j ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (j = 0 ; j < avdim ; j++)
        {
            // gather from the bitmap into the new A (:,j)
            int64_t pC = GB_IGET (Cp, j) ;
            int64_t pA_start = j * avlen ;
            for (int64_t i = 0 ; i < avlen ; i++)
            {
                int64_t pA = i + pA_start ;
                if (Ab [pA])
                {
                    // A(i,j) is in the bitmap
                    if (Ci != NULL) GB_ISET (Ci, pC, i) ; /* Ci [pC] = i */
                    if (Cj != NULL) GB_ISET (Cj, pC, j) ; /* Cj [pC] = j */
                    // Cx [pC] = Ax [pA])
                    GB_COPY (Cx, pC, Ax, pA) ;
                    pC++ ;
                }
            }
            ASSERT (pC == GB_IGET (Cp, j+1)) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // compute blocks of rows in parallel
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            #ifdef GB_JIT_KERNEL
            const GB_Cp_TYPE *restrict Wtask = W + tid * avdim ;
            #else
            uint32_t *restrict Wtask32 = Cp_is_32 ? (W32 + tid * avdim) : NULL ;
            uint64_t *restrict Wtask64 = Cp_is_32 ? NULL : (W64 + tid * avdim) ;
            #endif
            int64_t istart, iend ;
            GB_PARTITION (istart, iend, avlen, tid, nthreads) ;
            for (int64_t j = 0 ; j < avdim ; j++)
            {
                // gather from the bitmap into the new A (:,j)
                int64_t pC = GB_IGET (Cp, j) + GB_IGET (Wtask, j) ;
                int64_t pA_start = j * avlen ;
                for (int64_t i = istart ; i < iend ; i++)
                {
                    // see if A(i,j) is present in the bitmap
                    int64_t pA = i + pA_start ;
                    if (Ab [pA])
                    {
                        // A(i,j) is in the bitmap
                        if (Ci != NULL) GB_ISET (Ci, pC, i) ; /* Ci [pC] = i */
                        if (Cj != NULL) GB_ISET (Cj, pC, j) ; /* Cj [pC] = j */
                        GB_COPY (Cx, pC, Ax, pA) ;
                        pC++ ;
                    }
                }
            }
        }
    }
}

#undef GB_C_TYPE
#undef GB_A_TYPE

