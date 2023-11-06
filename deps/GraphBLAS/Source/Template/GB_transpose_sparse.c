//------------------------------------------------------------------------------
// GB_transpose_sparse: C=op(cast(A')), transpose, typecast, and apply op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //----------------------------------------------------------------------
    // A is sparse or hypersparse; C is sparse
    //----------------------------------------------------------------------

    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (GB_IS_SPARSE (C)) ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const int64_t anvec = A->nvec ;
    int64_t *restrict Ci = C->i ;

    if (nthreads == 1)
    {

        //------------------------------------------------------------------
        // sequential method
        //------------------------------------------------------------------

        int64_t *restrict workspace = Workspaces [0] ;
        for (int64_t k = 0 ; k < anvec ; k++)
        {
            // iterate over the entries in A(:,j)
            int64_t j = GBH_S (Ah, k) ;
            int64_t pA_start = Ap [k] ;
            int64_t pA_end = Ap [k+1] ;
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                // C(j,i) = A(i,j)
                int64_t i = Ai [pA] ;
                int64_t pC = workspace [i]++ ;
                Ci [pC] = j ;
                #ifndef GB_ISO_TRANSPOSE
                // Cx [pC] = op (Ax [pA])
                GB_APPLY_OP (pC, pA) ;
                #endif
            }
        }

    }
    else if (nworkspaces == 1)
    {

        //------------------------------------------------------------------
        // atomic method
        //------------------------------------------------------------------

        int64_t *restrict workspace = Workspaces [0] ;
        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            for (int64_t k = A_slice [tid] ; k < A_slice [tid+1] ; k++)
            {
                // iterate over the entries in A(:,j)
                int64_t j = GBH_S (Ah, k) ;
                int64_t pA_start = Ap [k] ;
                int64_t pA_end = Ap [k+1] ;
                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                { 
                    // C(j,i) = A(i,j)
                    int64_t i = Ai [pA] ;
                    // do this atomically:  pC = workspace [i]++
                    int64_t pC ;
                    GB_ATOMIC_CAPTURE_INC64 (pC, workspace [i]) ;
                    Ci [pC] = j ;
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

        //------------------------------------------------------------------
        // non-atomic method
        //------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            int64_t *restrict workspace = Workspaces [tid] ;
            for (int64_t k = A_slice [tid] ; k < A_slice [tid+1] ; k++)
            {
                // iterate over the entries in A(:,j)
                int64_t j = GBH_S (Ah, k) ;
                int64_t pA_start = Ap [k] ;
                int64_t pA_end = Ap [k+1] ;
                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                { 
                    // C(j,i) = A(i,j)
                    int64_t i = Ai [pA] ;
                    int64_t pC = workspace [i]++ ;
                    Ci [pC] = j ;
                    #ifndef GB_ISO_TRANSPOSE
                    // Cx [pC] = op (Ax [pA])
                    GB_APPLY_OP (pC, pA) ;
                    #endif
                }
            }
        }
    }
}

