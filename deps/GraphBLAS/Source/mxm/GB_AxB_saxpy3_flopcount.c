//------------------------------------------------------------------------------
// GB_AxB_saxpy3_flopcount:  compute flops for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: possible: many variants possible (matrix sparsity formats)
// matrices: A, B, M; bool Mask_comp

// On input, A, B, and M (optional) are matrices for C=A*B, C<M>=A*B, or
// C<!M>=A*B.  The flop count for each B(:,j) is computed, and returned as a
// cumulative sum.  This function is CSR/CSC agnostic, but for simplicity of
// this description, assume A and B are both CSC matrices, so that ncols(A) ==
// nrows(B).  For both CSR and CSC, A->vdim == B->vlen holds.  A and/or B may
// be hypersparse, in any combination.

// Bflops has size (B->nvec)+1, for both standard and hypersparse B.  Let
// n=B->vdim be the column dimension of B (that is, B is m-by-n).

// If B is a standard CSC matrix then Bflops has size n+1 == B->nvec+1, and on
// output, Bflops [j] is the # of flops required to compute C (:, 0:j-1).  B->h
// is NULL, and is implicitly the vector 0:(n-1).

// If B is hypersparse, then let Bh = B->h.  Its size is B->nvec, and j = Bh
// [kk] is the (kk)th column in the data structure for B.  C will also be
// hypersparse, and only C(:,Bh) will be computed (C may have fewer non-empty
// columns than B).  On output, Bflops [kk] is the number of needed flops to
// compute C (:, Bh [0:kk-1]).

// In both cases, Bflops [0] = 0, and Bflops [B->nvec] = total number of flops.
// The size of Bflops is B->nvec+1 so that it has the same size as B->p.  The
// first entry of B->p and Bflops are both zero.  This allows B to be sliced
// either by # of entries in B (by slicing B->p) or by the flop count required
// (by slicing Bflops).

// This algorithm does not look at the values of M, A, or B, just their
// patterns.  The flop count of C=A*B, C<M>=A*B, or C<!M>=A*B is computed for a
// saxpy-based method; the work for A'*B for the dot product method is not
// computed.

// The algorithm scans all nonzeros in B.  It only scans at most the min and
// max (first and last) row indices in A and M (if M is present).  If A and M
// are not hypersparse, the time taken is O(nnz(B)+n).  If all matrices are
// hypersparse, the time is O(nnz(B)*log(h)) where h = max # of vectors present
// in A and M.  Assuming B is in standard (not hypersparse) form:

/*
    [m n] = size (B) ;
    Bflops = zeros (1,n+1) ;        % (set to zero in the caller)
    Mwork = 0 ;
    for each column j in B:
        if (B (:,j) is empty) continue ;
        mjnz = nnz (M (:,j))
        if (M is present, not complemented, and M (:,j) is empty) continue ;
        Bflops (j) = mjnz if M present and not dense, to scatter M(:,j)
        Mwork += mjnz
        for each k where B (k,j) is nonzero:
            aknz = nnz (A (:,k))
            if (aknz == 0) continue ;
            % numerical phase will compute: C(:,j)<#M(:,j)> += A(:,k)*B(k,j)
            % where #M is no mask, M, or !M.  This typically takes aknz flops,
            % or with a binary search if nnz(M(:,j)) << nnz(A(:,k)).
            Bflops (j) += aknz
        end
    end
*/ 

#include "mxm/GB_mxm.h"
#include "mxm/GB_AxB_saxpy3.h"

#define GB_FREE_ALL                         \
{                                           \
    GB_WERK_POP (Work, uint64_t) ;          \
    GB_WERK_POP (B_ek_slicing, int64_t) ;   \
}

GrB_Info GB_AxB_saxpy3_flopcount
(
    int64_t *Mwork,             // amount of work to handle the mask M
    int64_t *Bflops,            // size B->nvec+1
    const GrB_Matrix M,         // optional mask matrix
    const bool Mask_comp,       // if true, mask is complemented
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK_OR_NULL (M, "M for flop count A*B", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_PENDING (M)) ;

    ASSERT_MATRIX_OK (A, "A for flop count A*B", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    ASSERT_MATRIX_OK (B, "B for flop count A*B", GB0) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_PENDING (B)) ;

    ASSERT (A->vdim == B->vlen) ;
    ASSERT (Bflops != NULL) ;
    ASSERT (Mwork != NULL) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t bnvec = B->nvec ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    GB_memset (Bflops, 0, (bnvec+1) * sizeof (int64_t), nthreads_max) ;

    //--------------------------------------------------------------------------
    // get the mask, if present: any sparsity structure
    //--------------------------------------------------------------------------

    bool mask_is_M = (M != NULL && !Mask_comp) ;

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    void *Mh = NULL ;
    const void *M_Yp = NULL ;
    const void *M_Yi = NULL ;
    const void *M_Yx = NULL ;
    int64_t mnvec = 0 ;
    int64_t mvlen = 0 ;
    int64_t M_hash_bits = 0 ;
    bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    bool M_is_dense = false ;
    bool Mp_is_32 = false ;
    bool Mj_is_32 = false ;
    if (M != NULL)
    { 
        Mh = M->h ;
        Mp_is_32 = M->p_is_32 ;
        Mj_is_32 = M->j_is_32 ;
        mnvec = M->nvec ;
        mvlen = M->vlen ;
        M_is_dense = GB_IS_BITMAP (M) || GB_as_if_full (M) ;
        if (M->Y != NULL)
        { 
            // mask is present, and hypersparse
            ASSERT_MATRIX_OK (M->Y, "M->Y hyper_hash", GB0) ;
            M_Yp = M->Y->p ;
            M_Yi = M->Y->i ;
            M_Yx = M->Y->x ;
            M_hash_bits = M->Y->vdim - 1 ;
        }
    }

    //--------------------------------------------------------------------------
    // get A and B: any sparsity structure
    //--------------------------------------------------------------------------

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    void *Ah = A->h ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const void *A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
    const void *A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
    const void *A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;
    const bool Ap_is_32 = A->p_is_32 ;
    const bool Aj_is_32 = A->j_is_32 ;

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    GB_Bi_DECLARE_U (Bi, const) ; GB_Bi_PTR (Bi, B) ;
    const int8_t  *restrict Bb = B->b ;
    const int64_t bvlen = B->vlen ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (Work, uint64_t) ;
    GB_WERK_DECLARE (B_ek_slicing, int64_t) ;

    //--------------------------------------------------------------------------
    // construct the parallel tasks
    //--------------------------------------------------------------------------

    int B_ntasks, B_nthreads ;
    GB_SLICE_MATRIX (B, 64) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    GB_WERK_PUSH (Work, 2*B_ntasks, uint64_t) ;
    if (Work == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    uint64_t *restrict Wfirst = Work ;
    uint64_t *restrict Wlast  = Work + B_ntasks ;

    //--------------------------------------------------------------------------
    // compute flop counts for C=A*B, C<M>=A*B, or C<!M>=A*B
    //--------------------------------------------------------------------------

    int64_t total_Mwork = 0 ;
    int taskid ;
    #pragma omp parallel for num_threads(B_nthreads) schedule(dynamic,1) \
        reduction(+:total_Mwork)
    for (taskid = 0 ; taskid < B_ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kfirst = kfirst_Bslice [taskid] ;
        int64_t klast  = klast_Bslice  [taskid] ;
        Wfirst [taskid] = 0 ;
        Wlast  [taskid] = 0 ;
        int64_t task_Mwork = 0 ;

        //----------------------------------------------------------------------
        // count flops for vectors kfirst to klast of B
        //----------------------------------------------------------------------

        for (int64_t kk = kfirst ; kk <= klast ; kk++)
        {

            // nnz (B (:,j)), for all tasks
            int64_t bjnz = (Bp == NULL) ? bvlen :
                (GB_IGET (Bp, kk+1) - GB_IGET (Bp, kk)) ;
            // C(:,j) is empty if the entire vector B(:,j) is empty
            if (bjnz == 0) continue ;

            //------------------------------------------------------------------
            // find the part of B(:,j) to be computed by this task
            //------------------------------------------------------------------

            GB_GET_PA (pB, pB_end, taskid, kk, kfirst, klast, pstart_Bslice,
                GBp_B (Bp, kk, bvlen), GBp_B (Bp, kk+1, bvlen)) ;
            int64_t my_bjnz = pB_end - pB ;
            int64_t j = GBh_B (Bh, kk) ;

            //------------------------------------------------------------------
            // see if M(:,j) is present and non-empty
            //------------------------------------------------------------------

            // if M(:,j) is full, bitmap, or dense, do not add mjnz to bjflops
            // or task_MWork.

            int64_t bjflops = my_bjnz ; // account for scan of B(:,j) itself
            int64_t mjnz = 0 ;
            if (M != NULL && !M_is_dense)
            {
                // find M(:,j): only do this if M is sparse or hypersparse
                int64_t pM, pM_end ;
                if (M_is_hyper)
                { 
                    // M is hypersparse: find M(:,j) in the M->Y hyper_hash
                    GB_hyper_hash_lookup (Mp_is_32, Mj_is_32,
                        Mh, mnvec, Mp, M_Yp, M_Yi, M_Yx, M_hash_bits,
                        j, &pM, &pM_end) ;
                }
                else
                { 
                    // M is sparse
                    pM     = GB_IGET (Mp, j) ;
                    pM_end = GB_IGET (Mp, j+1) ;
                }
                mjnz = pM_end - pM ;
                // If M not complemented: C(:,j) is empty if M(:,j) is empty.
                if (mjnz == 0 && !Mask_comp) continue ;
                if (mjnz > 0)
                {
                    // M(:,j) not empty
                    if (pB == GBp_B (Bp, kk, bvlen))
                    { 
                        // this task owns the top part of B(:,j), so it can
                        // account for the work to access M(:,j), without the
                        // work being duplicated by other tasks working on
                        // B(:,j)
                        bjflops = mjnz ;
                        // keep track of total work spent examining the mask.
                        // If any B(:,j) is empty, M(:,j) can be ignored.  So
                        // total_Mwork will be <= nnz (M).
                        task_Mwork += mjnz ;
                    }
                }
            }
            int64_t mjnz_much = 64 * mjnz ;

            //------------------------------------------------------------------
            // count the flops to compute C(:,j)<#M(:,j)> = A*B(:,j)
            //------------------------------------------------------------------

            // where #M is either not present, M, or !M

            for ( ; pB < pB_end ; pB++)
            {
                // get B(k,j)
                uint64_t k = GBi_B (Bi, pB, bvlen) ;
                if (!GBb_B (Bb, pB)) continue ;

                // B(k,j) is nonzero

                // find A(:,k)
                int64_t pA, pA_end ;
                if (A_is_hyper)
                { 
                    // A is hypersparse: find A(:,k) in the A->Y hyper_hash
                    GB_hyper_hash_lookup (Ap_is_32, Aj_is_32,
                        Ah, anvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits,
                        k, &pA, &pA_end) ;
                }
                else
                { 
                    // A is sparse, bitmap, or full
                    pA     = GBp_A (Ap, k  , avlen) ;
                    pA_end = GBp_A (Ap, k+1, avlen) ;
                }

                // skip if A(:,k) empty
                const int64_t aknz = pA_end - pA ;
                if (aknz == 0) continue ;

                double bkjflops ;

                // skip if intersection of A(:,k) and M(:,j) is empty
                // and mask is not complemented (C<M>=A*B)
                if (mask_is_M)
                {
                    // A(:,k) is non-empty; get first and last index of A(:,k)
                    if (aknz > 256 && mjnz_much < aknz && mjnz < mvlen &&
                        aknz < avlen && !(A->jumbled))
                    { 
                        // scan M(:j), and do binary search for A(i,j)
                        bkjflops = mjnz * (1 + 4 * log2 ((double) aknz)) ;
                    }
                    else
                    { 
                        // scan A(:k), and lookup M(i,j)
                        bkjflops = aknz ;
                    }
                }
                else
                { 
                    // A(:,k)*B(k,j) requires aknz flops
                    bkjflops = aknz ;
                }

                // increment by flops for the single entry B(k,j)
                // C(:,j)<#M(:,j)> += A(:,k)*B(k,j).
                bjflops += bkjflops ;
            }

            //------------------------------------------------------------------
            // log the flops for B(:,j)
            //------------------------------------------------------------------

            if (kk == kfirst)
            { 
                Wfirst [taskid] = bjflops ;
            }
            else if (kk == klast)
            { 
                Wlast [taskid] = bjflops ;
            }
            else
            { 
                Bflops [kk] = bjflops ;
            }
        }

        // compute the total work to access the mask, which is <= nnz (M)
        total_Mwork += task_Mwork ;
    }

    //--------------------------------------------------------------------------
    // reduce the first and last vector of each slice
    //--------------------------------------------------------------------------

    int64_t kprior = -1 ;

    for (int taskid = 0 ; taskid < B_ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // sum up the partial flops that taskid computed for kfirst
        //----------------------------------------------------------------------

        int64_t kfirst = kfirst_Bslice [taskid] ;
        int64_t klast  = klast_Bslice  [taskid] ;

        if (kfirst <= klast)
        {
            int64_t pB = pstart_Bslice [taskid] ;
            int64_t pB_end = GBp_B (Bp, kfirst+1, bvlen) ;
            pB_end = GB_IMIN (pB_end, pstart_Bslice [taskid+1]) ;
            if (pB < pB_end)
            {
                if (kprior < kfirst)
                { 
                    // This task is the first one that did work on
                    // B(:,kfirst), so use it to start the reduction.
                    Bflops [kfirst] = Wfirst [taskid] ;
                }
                else
                { 
                    // subsequent task for B(:,kfirst)
                    Bflops [kfirst] += Wfirst [taskid] ;
                }
                kprior = kfirst ;
            }
        }

        //----------------------------------------------------------------------
        // sum up the partial flops that taskid computed for klast
        //----------------------------------------------------------------------

        if (kfirst < klast)
        {
            int64_t pB = GBp_B (Bp, klast, bvlen) ;
            int64_t pB_end = pstart_Bslice [taskid+1] ;
            if (pB < pB_end)
            {
                /* if */ ASSERT (kprior < klast) ;
                { 
                    // This task is the first one that did work on
                    // B(:,klast), so use it to start the reduction.
                    Bflops [klast] = Wlast [taskid] ;
                }
                /*
                else
                {
                    // If kfirst < klast and B(:,klast) is not empty,
                    // then this task is always the first one to do
                    // work on B(:,klast), so this case is never used.
                    ASSERT (GB_DEAD_CODE) ;
                    // subsequent task to work on B(:,klast)
                    Bflops [klast] += Wlast [taskid] ;
                }
                */
                kprior = klast ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // cumulative sum of Bflops
    //--------------------------------------------------------------------------

    // Bflops = cumsum ([0 Bflops]) ;
    ASSERT (Bflops [bnvec] == 0) ;
    GB_cumsum (Bflops, false, bnvec, NULL, B_nthreads, Werk) ;
    // Bflops [bnvec] is now the total flop count, including the time to
    // compute A*B and to handle the mask.  total_Mwork is part of this total
    // flop count, but is also returned separtely.

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    (*Mwork) = total_Mwork ;
    return (GrB_SUCCESS) ;
}

