//------------------------------------------------------------------------------
// GB_convert_sparse_to_hyper: convert a matrix from sparse to hyperspasre
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// On input, the matrix may have shallow A->p content; it is safely removed.
// On output, the matrix is always hypersparse (even if out of memory).  If the
// input matrix is non-hypersparse, it is given new A->p and A->h that are not
// shallow.  If the input matrix is already hypersparse, nothing is changed
// (and in that case A->p and A->h remain shallow on output if shallow on
// input). The A->x and A->i content is not changed; it remains in whatever
// shallow/non-shallow/iso property that it had on input).

// If an out-of-memory condition occurs, all content of the matrix is cleared.

// If the input matrix A is hypersparse, bitmap or full, it is unchanged.

#include "GB.h"

GrB_Info GB_convert_sparse_to_hyper // convert from sparse to hypersparse
(
    GrB_Matrix A,           // matrix to convert to hypersparse
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converting to hypersparse", GB0) ;
    int64_t anz = GB_nnz (A) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // convert A from sparse to hypersparse
    //--------------------------------------------------------------------------

    if (GB_IS_SPARSE (A))
    {

        //----------------------------------------------------------------------
        // determine the number of threads to use
        //----------------------------------------------------------------------

        int64_t n = A->vdim ;
        GB_BURBLE_N (n, "(sparse to hyper) ") ;
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int nthreads = GB_nthreads (n, chunk, nthreads_max) ;
        int ntasks = (nthreads == 1) ? 1 : (8 * nthreads) ;
        ntasks = GB_IMIN (ntasks, n) ;
        ntasks = GB_IMAX (ntasks, 1) ;

        //----------------------------------------------------------------------
        // count the number of non-empty vectors in A in each slice
        //----------------------------------------------------------------------

        ASSERT (A->nvec == A->plen && A->plen == n) ;

        const int64_t *restrict Ap_old = A->p ;
        size_t Ap_old_size = A->p_size ;
        bool Ap_old_shallow = A->p_shallow ;

        GB_WERK_DECLARE (Count, int64_t) ;
        GB_WERK_PUSH (Count, ntasks+1, int64_t) ;
        if (Count == NULL)
        { 
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {
            int64_t jstart, jend, my_nvec_nonempty = 0 ; ;
            GB_PARTITION (jstart, jend, n, tid, ntasks) ;
            for (int64_t j = jstart ; j < jend ; j++)
            { 
                if (Ap_old [j] < Ap_old [j+1]) my_nvec_nonempty++ ;
            }
            Count [tid] = my_nvec_nonempty ;
        }

        //----------------------------------------------------------------------
        // compute cumulative sum of Counts and nvec_nonempty
        //----------------------------------------------------------------------

        GB_cumsum (Count, ntasks, NULL, 1, NULL) ;
        int64_t nvec_nonempty = Count [ntasks] ;
        A->nvec_nonempty = nvec_nonempty ;

        //----------------------------------------------------------------------
        // allocate the new A->p and A->h
        //----------------------------------------------------------------------

        int64_t *restrict Ap_new = NULL ; size_t Ap_new_size = 0 ;
        int64_t *restrict Ah_new = NULL ; size_t Ah_new_size = 0 ;
        int64_t plen_new = (n == 1) ? 1 : nvec_nonempty ;
        Ap_new = GB_MALLOC (plen_new+1, int64_t, &Ap_new_size) ;
        Ah_new = GB_MALLOC (plen_new  , int64_t, &Ah_new_size) ;
        if (Ap_new == NULL || Ah_new == NULL)
        { 
            // out of memory
            GB_WERK_POP (Count, int64_t) ;
            GB_FREE (&Ap_new, Ap_new_size) ;
            GB_FREE (&Ah_new, Ah_new_size) ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // transplant the new A->p and A->h into the matrix
        //----------------------------------------------------------------------

        A->plen = plen_new ;
        A->nvec = nvec_nonempty ;
        A->p = Ap_new ; A->p_size = Ap_new_size ;
        A->h = Ah_new ; A->h_size = Ah_new_size ;
        A->p_shallow = false ;
        A->h_shallow = false ;

        //----------------------------------------------------------------------
        // construct the new hyperlist in the new A->p and A->h
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {
            int64_t jstart, jend, k = Count [tid] ;
            GB_PARTITION (jstart, jend, n, tid, ntasks) ;
            for (int64_t j = jstart ; j < jend ; j++)
            {
                if (Ap_old [j] < Ap_old [j+1])
                { 
                    // vector index j is the kth vector in the new Ah
                    Ap_new [k] = Ap_old [j] ;
                    Ah_new [k] = j ;
                    k++ ;
                }
            }
            ASSERT (k == Count [tid+1]) ;
        }

        Ap_new [nvec_nonempty] = anz ;
        A->magic = GB_MAGIC ;

        //----------------------------------------------------------------------
        // free workspace, and free the old A->p unless it's shallow
        //----------------------------------------------------------------------

        GB_WERK_POP (Count, int64_t) ;
        if (!Ap_old_shallow)
        { 
            GB_FREE (&Ap_old, Ap_old_size) ;
        }

        //----------------------------------------------------------------------
        // A is now hypersparse, but A->Y is not yet constructed
        //----------------------------------------------------------------------

        ASSERT (GB_IS_HYPERSPARSE (A)) ;
        ASSERT (A->Y == NULL && A->Y_shallow == false) ;
    }

    //--------------------------------------------------------------------------
    // A is now in hypersparse form (or left as full or bitmap)
    //--------------------------------------------------------------------------

    ASSERT (anz == GB_nnz (A)) ;
    ASSERT_MATRIX_OK (A, "A conv to hypersparse (or left full/bitmap)", GB0) ;
    ASSERT (!GB_IS_SPARSE (A)) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;
    return (GrB_SUCCESS) ;
}

