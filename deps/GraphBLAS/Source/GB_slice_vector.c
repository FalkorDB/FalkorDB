//------------------------------------------------------------------------------
// GB_slice_vector:  slice a vector for GB_add, GB_emult, and GB_mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A(:,kA) and B(:,kB) are two long vectors that will be added with GB_add,
// GB_emult, or GB_mask, and the work to compute them needs to be split into
// multiple tasks.  They represent the same vector index j, for:

//      C(:,j) = A(:,j) +  B(:,j) in GB_add
//      C(:,j) = A(:,j) .* B(:,j) in GB_emult
//      C(:,j)<M(:,j)> = B(:,j) in GB_mask; A is passed in as the input C
//      union (A->h, B->h) in GB_add_phase0.

// The vector index j is not needed here.  The vectors kA and kB are not
// required, either; just the positions where the vectors appear in A and B
// (pA_start, pA_end, pB_start, and pB_end).

// The inputs Mi, Ai, and Bi must be sorted on input.

// This method finds i so that nnz (A (i:end,kA)) + nnz (B (i:end,kB)) is
// roughly equal to target_work.  The entries in A(i:end,kA) start at position
// pA in Ai and Ax, and the entries in B(i:end,kB) start at position pB in Bi
// and Bx.  Once the work is split, pM is found for M(i:end,kM), if the mask M
// is present.

// The lists Mi, Ai, and Bi can also be any sorted integer array.  This is used
// by GB_add_phase0 to construct the set union of A->h and B->h.  In this case,
// pA_start and pB_start are both zero, and pA_end and pB_end are A->nvec and
// B->nvec, respectively.

// If n = A->vlen = B->vlen, anz = nnz (A (:,kA)), and bnz = nnz (B (:,kB)),
// then the total time taken by this function is O(log(n)*(log(anz)+log(bnz))),
// or at most O((log(n)^2)).

// The input matrices M, A, and B are not present here, except for M->i,
// A->i, and B->i if they are sparse or hypersparse.  They cannot be jumbled.
// M, A, and B can have any sparsity structure.  If bitmap or full, their
// corresponding [A,B,M]->i arrays are NULL.

#include "GB.h"

void GB_slice_vector
(
    // output: return i, pA, and pB
    int64_t *p_i,                   // work starts at A(i,kA) and B(i,kB)
    int64_t *p_pM,                  // M(i:end,kM) starts at pM
    int64_t *p_pA,                  // A(i:end,kA) starts at pA
    int64_t *p_pB,                  // B(i:end,kB) starts at pB
    // input:
    const int64_t pM_start,         // M(:,kM) starts at pM_start in Mi,Mx
    const int64_t pM_end,           // M(:,kM) ends at pM_end-1 in Mi,Mx
    const int64_t *restrict Mi,  // indices of M (or NULL)
    const int64_t pA_start,         // A(:,kA) starts at pA_start in Ai,Ax
    const int64_t pA_end,           // A(:,kA) ends at pA_end-1 in Ai,Ax
    const int64_t *restrict Ai,  // indices of A (or NULL)
    const int64_t pB_start,         // B(:,kB) starts at pB_start in Bi,Bx
    const int64_t pB_end,           // B(:,kB) ends at pB_end-1 in Bi,Bx
    const int64_t *restrict Bi,  // indices of B (or NULL)
    const int64_t vlen,             // A->vlen and B->vlen
    const double target_work        // target work
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_pA != NULL && p_pB != NULL) ;

    //--------------------------------------------------------------------------
    // find i, pA, and pB for the start of this task
    //--------------------------------------------------------------------------

    // search for index i in the range ileft:iright, inclusive
    int64_t ileft  = 0 ;
    int64_t iright = vlen-1 ;
    int64_t i = 0 ;

    const int64_t aknz = pA_end - pA_start ;
    const int64_t bknz = pB_end - pB_start ;
    const int64_t mknz = pM_end - pM_start ;      // zero if M not present

    const bool a_empty = (aknz == 0) ;
    const bool b_empty = (bknz == 0) ;
    const bool m_empty = (mknz == 0) ;

    int64_t pM = (m_empty) ? -1 : pM_start ;
    int64_t pA = (a_empty) ? -1 : pA_start ;
    int64_t pB = (b_empty) ? -1 : pB_start ;

    while (ileft < iright)
    {

        //----------------------------------------------------------------------
        // find the index i in the middle of ileft:iright
        //----------------------------------------------------------------------

        i = (ileft + iright) >> 1 ;

        //----------------------------------------------------------------------
        // find where i appears in A(:,kA)
        //----------------------------------------------------------------------

        if (a_empty)
        { 
            // Ai is empty so i does not appear
            pA = -1 ;
        }
        else if (aknz == vlen)
        { 
            // A(:,kA) is dense (bitmap, full, or all entries present)
            // no need for a binary search
            pA = pA_start + i ;
            ASSERT (GBI (Ai, pA, vlen) == i) ;
        }
        else
        { 
            // Ai is an explicit integer list, Ai [pA_start:pA_end-1]
            ASSERT (aknz > 0) ;
            pA = pA_start ;
            bool afound ;
            int64_t apright = pA_end - 1 ;
            GB_SPLIT_BINARY_SEARCH (i, Ai, pA, apright, afound) ;
            ASSERT (GB_IMPLIES (afound, GBI (Ai, pA, vlen) == i)) ;
            ASSERT (pA_start <= pA && pA <= pA_end) ;
        }

        ASSERT (GB_IMPLIES (pA >  pA_start && pA < pA_end,
            (GBI (Ai, pA-1, vlen) < i))) ;
        ASSERT (GB_IMPLIES (pA >= pA_start && pA < pA_end,
            (GBI (Ai, pA, vlen) >= i ))) ;

        // Ai has been split.  If afound is false:
        //      Ai [pA_start : pA-1] < i
        //      Ai [pA : pA_end-1]   > i
        // If afound is true:
        //      Ai [pA_start : pA-1] < i
        //      Ai [pA : pA_end-1]  >= i
        //
        // in both cases, if i is chosen as the breakpoint, then the
        // subtask starts at index i, and position pA in Ai,Ax.

        // if A(:,kA) is empty, then pA is -1

        //----------------------------------------------------------------------
        // find where i appears in B(:,kB)
        //----------------------------------------------------------------------

        if (b_empty)
        { 
            // B(:,kB) is empty so i does not appear
            pB = -1 ;
        }
        else if (bknz == vlen)
        { 
            // B(:,kB) is dense (bitmap, full, or all entries present)
            // no need for a binary search
            pB = pB_start + i ;
            ASSERT (GBI (Bi, pB, vlen) == i) ;
        }
        else
        { 
            // B(:,kB) is sparse, and not empty
            ASSERT (bknz > 0) ;
            ASSERT (Bi != NULL) ;
            pB = pB_start ;
            bool bfound ;
            int64_t bpright = pB_end - 1 ;
            GB_SPLIT_BINARY_SEARCH (i, Bi, pB, bpright, bfound) ;
            ASSERT (pB_start <= pB && pB <= pB_end) ;
        }
        ASSERT (GB_IMPLIES (pB >  pB_start && pB < pB_end,
            (GBI (Bi, pB-1, vlen) < i))) ;
        ASSERT (GB_IMPLIES (pB >= pB_start && pB < pB_end,
            (GBI (Bi, pB, vlen) >= i ))) ;

        // Bi has been split.  If bfound is false:
        //      Bi [pB_start : pB-1] < i
        //      Bi [pB : pB_end-1]   > i
        // If bfound is true:
        //      Bi [pB_start : pB-1] < i
        //      Bi [pB : pB_end-1]  >= i
        //
        // in both cases, if i is chosen as the breakpoint, then the
        // subtask starts at index i, and position pB in Bi,Bx.

        // if B(:,kB) is empty, then pB is -1

        //----------------------------------------------------------------------
        // determine if the subtask is near the target task size
        //----------------------------------------------------------------------

        double work = (a_empty ? 0 : (pA_end - pA))
                    + (b_empty ? 0 : (pB_end - pB)) ;

        if (work < 0.9999 * target_work)
        { 

            //------------------------------------------------------------------
            // work is too low
            //------------------------------------------------------------------

            // work is too low, so i is too high.
            // Keep searching in the range (ileft:i), inclusive.

            iright = i ;

        }
        else if (work > 1.0001 * target_work)
        { 

            //------------------------------------------------------------------
            // work is too high
            //------------------------------------------------------------------

            // work is too high, so i is too low.
            // Keep searching in the range (i+1):iright, inclusive.

            ileft = i + 1 ;

        }
        else
        { 

            //------------------------------------------------------------------
            // work is about right; use this result.
            //------------------------------------------------------------------

            // return i, pA, and pB as the start of this task.
            ASSERT (0 <= i && i <= vlen) ;
            ASSERT (pA == -1 || (pA_start <= pA && pA <= pA_end)) ;
            ASSERT (pB == -1 || (pB_start <= pB && pB <= pB_end)) ;
            break ;
        }
    }

    //--------------------------------------------------------------------------
    // find where i appears in M(:,kM)
    //--------------------------------------------------------------------------

    if (m_empty)
    { 
        pM = -1 ;
    }
    else if (mknz == vlen)
    { 
        // M(:,kM) is dense (bitmap, full, or all entries present)
        // no need for a binary search
        pM = pM_start + i ;
        ASSERT (GBI (Mi, pM, vlen) == i) ;
    }
    else
    { 
        // M(:,kM) is sparse, and not empty
        ASSERT (mknz > 0) ;
        ASSERT (Mi != NULL) ;
        pM = pM_start ;
        bool mfound ;
        int64_t mpright = pM_end - 1 ;
        GB_SPLIT_BINARY_SEARCH (i, Mi, pM, mpright, mfound) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // pM, pA, and pB partition the three vectors M(:,j), A(:,j), and B(:,j),
    // or if any vector is empty, their p* pointer is -1.

    ASSERT (GB_IMPLIES ((pM >  pM_start && pM < pM_end),
        GBI (Mi, pM-1, vlen) <  i)) ;
    ASSERT (GB_IMPLIES ((pM >= pM_start && pM < pM_end),
        GBI (Mi, pM, vlen) >= i)) ;
    ASSERT (GB_IMPLIES ((pA >  pA_start && pA < pA_end),
        GBI (Ai, pA-1, vlen) <  i)) ;
    ASSERT (GB_IMPLIES ((pA >= pA_start && pA < pA_end),
        GBI (Ai, pA, vlen) >= i)) ;
    ASSERT (GB_IMPLIES ((pB >  pB_start && pB < pB_end),
        GBI (Bi, pB-1, vlen) <  i)) ;
    ASSERT (GB_IMPLIES ((pB >= pB_start && pB < pB_end),
        GBI (Bi, pB, vlen) >= i)) ;

    if (p_i != NULL)
    { 
        (*p_i)  = i ;
    }
    if (p_pM != NULL)
    { 
        (*p_pM) = pM ;
    }
    (*p_pA) = pA ;
    (*p_pB) = pB ;
}

