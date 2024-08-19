//------------------------------------------------------------------------------
// GB_convert_hyper_to_sparse: convert a matrix from hypersparse to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// On input, the matrix may have shallow A->p and A->h content; it is safely
// removed.  On output, the matrix is always non-hypersparse (even if out of
// memory).  If the input matrix is hypersparse, it is given a new A->p that is
// not shallow.  If the input matrix is already non-hypersparse, nothing is
// changed (and in that case A->p remains shallow on output if shallow on
// input). The A->x and A->i content is not changed; it remains in whatever
// shallow/non-shallow/iso property that it had on input).

// If an out-of-memory condition occurs, A is unchanged.

// If the input matrix A is sparse, bitmap or full, it is unchanged.

#include "GB.h"

GrB_Info GB_convert_hyper_to_sparse // convert hypersparse to sparse
(
    GrB_Matrix A,           // matrix to convert to non-hypersparse
    bool do_burble          // if true, then burble is allowed
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A being converted from hyper to sparse", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    if (!GB_IS_HYPERSPARSE (A))
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // convert A from hypersparse to sparse
    //--------------------------------------------------------------------------

    if (do_burble) GBURBLE ("(hyper to sparse) ") ;
    int64_t n = A->vdim ;
    int64_t anz = GB_nnz (A) ;

    if (n == 1)
    { 

        //----------------------------------------------------------------------
        // A is a single hypersparse vector
        //----------------------------------------------------------------------

        // This cannot fail, since no memory is allocated.  It must succeed if
        // A is a typecasted GrB_Vector, or else it will be returned to the
        // user as an invalid GrB_Vector.

        ASSERT (A->plen == 1) ;
        ASSERT (A->p_size >= 2 * sizeof (int64_t)) ;
        ASSERT (A->nvec == 0 || A->nvec == 1) ;
        if (A->nvec == 0)
        { 
            A->p [0] = 0 ;
            A->p [1] = 0 ;
            A->nvec = 1 ;
        }
        A->nvec_nonempty = (anz > 0) ? 1 : 0 ;

        // free A->h unless it is shallow
        if (!A->h_shallow)
        { 
            GB_FREE (&(A->h), A->h_size) ;
        }
        A->h = NULL ;
        A->h_size = 0 ;
        A->h_shallow = false ;
        GB_hyper_hash_free (A) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // determine the number of threads to use
        //----------------------------------------------------------------------

        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int nthreads = GB_nthreads (n, chunk, nthreads_max) ;
        int ntasks = (nthreads == 1) ? 1 : (8 * nthreads) ;
        ntasks = GB_IMIN (ntasks, n) ;
        ntasks = GB_IMAX (ntasks, 1) ;

        //----------------------------------------------------------------------
        // allocate the new Ap array, of size n+1
        //----------------------------------------------------------------------

        int64_t *restrict Ap_new = NULL ; size_t Ap_new_size = 0 ;
        Ap_new = GB_MALLOC (n+1, int64_t, &Ap_new_size) ;
        if (Ap_new == NULL)
        { 
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }

        #ifdef GB_DEBUG
        // to ensure all values of Ap_new are assigned below.
        for (int64_t j = 0 ; j <= n ; j++) Ap_new [j] = -99999 ;
        #endif

        //----------------------------------------------------------------------
        // get the old hyperlist
        //----------------------------------------------------------------------

        int64_t nvec = A->nvec ;            // # of vectors in Ah_old
        int64_t *restrict Ap_old = A->p ;   // size nvec+1
        int64_t *restrict Ah_old = A->h ;   // size nvec
        int64_t nvec_nonempty = 0 ;         // recompute A->nvec_nonempty

        //----------------------------------------------------------------------
        // construct the new vector pointers
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(+:nvec_nonempty)
        for (tid = 0 ; tid < ntasks ; tid++)
        {
            int64_t jstart, jend, my_nvec_nonempty = 0 ;
            GB_PARTITION (jstart, jend, n, tid, ntasks) ;
            ASSERT (0 <= jstart && jstart <= jend && jend <= n) ;

            // task tid computes Ap_new [jstart:jend-1] from Ap_old, Ah_old.

            // GB_SPLIT_BINARY_SEARCH of Ah_old [0..nvec-1] for jstart:
            // If found is true then Ah_old [k] == jstart.
            // If found is false, and nvec > 0 then
            //    Ah_old [0 ... k-1] < jstart <  Ah_old [k ... nvec-1]
            // Whether or not i is found, if nvec > 0
            //    Ah_old [0 ... k-1] < jstart <= Ah_old [k ... nvec-1]
            // If nvec == 0, then k == 0 and found will be false.  In this
            // case, jstart cannot be compared with any content of Ah_old,
            // since Ah_old is completely empty (Ah_old [0] is invalid).

            int64_t k = 0, pright = nvec-1 ;
            bool found ;
            GB_SPLIT_BINARY_SEARCH (jstart, Ah_old, k, pright, found) ;
            ASSERT (k >= 0 && k <= nvec) ;
            ASSERT (GB_IMPLIES (nvec == 0, !found && k == 0)) ;
            ASSERT (GB_IMPLIES (found, jstart == Ah_old [k])) ;
            ASSERT (GB_IMPLIES (!found && k < nvec, jstart < Ah_old [k])) ;

            // Let jk = Ah_old [k], jlast = Ah_old [k-1], and pk = Ah_old [k].
            // Then Ap_new [jlast+1:jk] must be set to pk.  This must be done
            // for all k = 0:nvec-1.  In addition, the last vector k=nvec-1
            // must be terminated by setting Ap_new [jk+1:n-1] to Ap_old [nvec].
            // A task owns the kth vector if jk is in jstart:jend-1, inclusive.
            // It counts all non-empty vectors that it owns.  However, the task
            // must also set Ap_new [...] = pk for any jlast+1:jk that overlaps
            // jstart:jend-1, even if it does not own that particular vector k.
            // This happens only at the tail end of jstart:jend-1. 

            int64_t jlast = (k == 0) ? (-1) : Ah_old [k-1] ;
            jlast = GB_IMAX (jstart-1, jlast) ;

            bool done = false ;

            for ( ; k <= nvec && !done ; k++)
            {

                //--------------------------------------------------------------
                // get the kth vector in Ah_old, which is vector index jk.
                //--------------------------------------------------------------

                int64_t jk = (k < nvec) ? Ah_old [k] : n ;
                int64_t pk = (k < nvec) ? Ap_old [k] : anz ;

                //--------------------------------------------------------------
                // determine if this task owns jk
                //--------------------------------------------------------------

                int64_t jfin ;
                if (jk >= jend)
                { 
                    // This is the last iteration for this task.  This task
                    // does not own the kth vector.  However, it does own the
                    // vector indices jlast+1:jend-1, and these vectors must
                    // be handled by this task.
                    jfin = jend - 1 ;
                    done = true ;
                }
                else
                { 
                    // This task owns the kth vector, which is vector index jk.
                    // Ap must be set to pk for all vector indices jlast+1:jk.
                    jfin = jk ;
                    ASSERT (k >= 0 && k < nvec && nvec > 0) ;
                    if (pk < Ap_old [k+1]) my_nvec_nonempty++ ;
                }

                //--------------------------------------------------------------
                // set Ap_new for this vector
                //--------------------------------------------------------------

                // Ap_new [jlast+1:jk] must be set to pk.  This tasks handles
                // the intersection of jlast+1:jk with jstart:jend-1.

                for (int64_t j = jlast+1 ; j <= jfin ; j++)
                { 
                    Ap_new [j] = pk ;
                }

                //--------------------------------------------------------------
                // keep track of the prior vector index
                //--------------------------------------------------------------

                jlast = jk ;
            }
            nvec_nonempty += my_nvec_nonempty ;

            //------------------------------------------------------------------
            // no task owns Ap_new [n] so it is set by the last task
            //------------------------------------------------------------------

            if (tid == ntasks-1)
            { 
                ASSERT (jend == n) ;
                Ap_new [n] = anz ;
            }
        }

        // free the old A->p, A->h, and A->H hyperlist content.
        // this clears A->nvec_nonempty so it must be restored below.
        GB_phy_free (A) ;

        // transplant the new vector pointers; matrix is no longer hypersparse
        A->p = Ap_new ; A->p_size = Ap_new_size ;
        A->h = NULL ;
        A->nvec = n ;
        A->nvec_nonempty = nvec_nonempty ;
        A->plen = n ;
        A->p_shallow = false ;
        A->h_shallow = false ;
        A->nvals = anz ;
    }

    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // A is now sparse
    //--------------------------------------------------------------------------

    ASSERT (anz == A->p [n]) ;
    ASSERT (anz == GB_nnz (A)) ;
    ASSERT_MATRIX_OK (A, "A converted to sparse", GB0) ;
    ASSERT (GB_IS_SPARSE (A)) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;
    return (GrB_SUCCESS) ;
}

