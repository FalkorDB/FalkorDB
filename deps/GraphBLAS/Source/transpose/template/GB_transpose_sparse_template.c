//------------------------------------------------------------------------------
// GB_transpose_sparse_template: C=op(cast(A')), transpose, typecast, & apply op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    #undef GBh_AB
    #ifdef GB_BIND_1ST
        // see discussion in GB_transpose_template.c
        GB_Bp_DECLARE (Ap, const) ; GB_Bp_PTR (Ap, A) ;
        GB_Bh_DECLARE (Ah, const) ; GB_Bh_PTR (Ah, A) ;
        GB_Bi_DECLARE (Ai, const) ; GB_Bi_PTR (Ai, A) ;
        #define GBh_AB(Ah,k) GBh_B(Ah,k)
    #else
        GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
        GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
        GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
        #define GBh_AB(Ah,k) GBh_A(Ah,k)
    #endif
    GB_Ci_DECLARE (Ci, ) ; GB_Ci_PTR (Ci, C) ;

    //--------------------------------------------------------------------------
    // C = A'
    //--------------------------------------------------------------------------

    if (nthreads == 1)
    {

        //----------------------------------------------------------------------
        // sequential method
        //----------------------------------------------------------------------

        // Cp and workspace are of type GB_W_TYPE
        GB_W_TYPE *restrict workspace = (GB_W_TYPE *) (Workspaces [0]) ;
        const int64_t anvec = A->nvec ;
        for (int64_t k = 0 ; k < anvec ; k++)
        {
            // iterate over the entries in A(:,j)
            int64_t j = GBh_AB (Ah, k) ;
            int64_t pA_start = GB_IGET (Ap, k) ;
            int64_t pA_end = GB_IGET (Ap, k+1) ;
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                // C(j,i) = A(i,j)
                int64_t i = GB_IGET (Ai, pA) ;
                GB_W_TYPE pC = workspace [i]++ ;
                // Ci [pC] = j ;
                GB_ISET (Ci, pC, j) ;
                #ifndef GB_ISO_TRANSPOSE
                // Cx [pC] = op (Ax [pA])
                GB_APPLY_OP (pC, pA) ;
                #endif
            }
        }

    }
    else if (nworkspaces == 1)
    {

        //----------------------------------------------------------------------
        // atomic method
        //----------------------------------------------------------------------

        GB_W_TYPE *restrict workspace = (GB_W_TYPE *) (Workspaces [0]) ;
        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            for (int64_t k = A_slice [tid] ; k < A_slice [tid+1] ; k++)
            {
                // iterate over the entries in A(:,j)
                int64_t j = GBh_AB (Ah, k) ;
                int64_t pA_start = GB_IGET (Ap, k) ;
                int64_t pA_end = GB_IGET (Ap, k+1) ;
                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                { 
                    // C(j,i) = A(i,j)
                    int64_t i = GB_IGET (Ai, pA) ;
                    // do this atomically:  pC = workspace [i]++
                    GB_W_TYPE pC ;
                    GB_ATOMIC_CAPTURE_INC (pC, workspace [i]) ;
                    // Ci [pC] = j ;
                    GB_ISET (Ci, pC, j) ;
                    #ifndef GB_ISO_TRANSPOSE
                    // Cx [pC] = op (Ax [pA])
                    GB_APPLY_OP (pC, pA) ;
                    #endif
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // non-atomic method
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            GB_W_TYPE *restrict workspace = (GB_W_TYPE *) (Workspaces [tid]) ;
            for (int64_t k = A_slice [tid] ; k < A_slice [tid+1] ; k++)
            {
                // iterate over the entries in A(:,j)
                int64_t j = GBh_AB (Ah, k) ;
                int64_t pA_start = GB_IGET (Ap, k) ;
                int64_t pA_end = GB_IGET (Ap, k+1) ;
                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                { 
                    // C(j,i) = A(i,j)
                    int64_t i = GB_IGET (Ai, pA) ;
                    GB_W_TYPE pC = workspace [i]++ ;
                    // Ci [pC] = j ;
                    GB_ISET (Ci, pC, j) ;
                    #ifndef GB_ISO_TRANSPOSE
                    // Cx [pC] = op (Ax [pA])
                    GB_APPLY_OP (pC, pA) ;
                    #endif
                }
            }
        }
    }
}

#undef GB_W_TYPE
#undef GB_ATOMIC_CAPTURE_INC
#undef GBh_AB

