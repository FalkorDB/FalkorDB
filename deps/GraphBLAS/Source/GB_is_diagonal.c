//------------------------------------------------------------------------------
// GB_is_diagonal: check if A is a diagonal matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// Returns true if A is a square diagonal matrix, with all diagonal entries
// present.  All pending tuples are ignored.  Zombies are treated as entries.

#include "GB_mxm.h"

bool GB_is_diagonal             // true if A is diagonal
(
    const GrB_Matrix A          // input matrix to examine
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;
    ASSERT_MATRIX_OK (A, "A check diag", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // trivial cases
    //--------------------------------------------------------------------------

    int64_t n     = GB_NROWS (A) ;
    int64_t ncols = GB_NCOLS (A) ;

    if (n != ncols)
    { 
        // A is rectangular
        return (false) ;
    }

    if (GB_IS_BITMAP (A))
    { 
        // never treat bitmaps as diagonal
        return (false) ;
    }

    if (GB_IS_FULL (A))
    { 
        // A is full, and is diagonal only if 1-by-1, but always return
        // false so that GB_rowscale and GB_colscale are not used
        // by GB_reduce_to_vector.
        return (false) ;
    }

    int64_t anz  = GB_nnz (A) ;
    int64_t nvec = A->nvec ;

    if (n != anz || n != nvec)
    { 
        // A must have exactly n entries in n vectors.  A can be sparse or
        // hypersparse.  If hypersparse, all vectors must be present, so
        // Ap has size n+1 whether sparse or hypersparse.
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    // Break the work into lots of tasks so the early-exit can be exploited.

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (n, chunk, nthreads_max) ;
    int ntasks = (nthreads == 1) ? 1 : (256 * nthreads) ;
    ntasks = GB_IMIN (ntasks, n) ;
    ntasks = GB_IMAX (ntasks, 1) ;

    //--------------------------------------------------------------------------
    // examine each vector of A
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ai = A->i ;

    int diagonal = true ;

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // check for early exit
        //----------------------------------------------------------------------

        int diag = true ;
        { 
            GB_ATOMIC_READ
            diag = diagonal ;
        }
        if (!diag) continue ;

        //----------------------------------------------------------------------
        // check if vectors jstart:jend-1 are diagonal
        //----------------------------------------------------------------------

        int64_t jstart, jend ;
        GB_PARTITION (jstart, jend, n, tid, ntasks) ;
        for (int64_t j = jstart ; diag && j < jend ; j++)
        {
            int64_t p = Ap [j] ;
            int64_t ajnz = Ap [j+1] - p ;
            if (ajnz != 1)
            { 
                // A(:,j) must have exactly one entry
                diag = false ;
            }
            int64_t i = Ai [p] ;
            if (i != j)
            { 
                // the single entry must be A(i,i)
                diag = false ;
            }
        }

        //----------------------------------------------------------------------
        // early exit: tell all other tasks to halt
        //----------------------------------------------------------------------

        if (!diag)
        { 
            GB_ATOMIC_WRITE
            diagonal = false ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return ((bool) diagonal) ;
}

