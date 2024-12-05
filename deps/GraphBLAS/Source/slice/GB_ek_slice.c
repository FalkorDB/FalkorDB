//------------------------------------------------------------------------------
// GB_ek_slice: slice the entries and vectors of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Slice the entries of a matrix or vector into ntasks slices.

// The function is called GB_ek_slice because it first partitions the e entries
// into chunks of identical sizes, and then finds the first and last vector
// (k) for each chunk.

// Task t does entries pstart_slice [t] to pstart_slice [t+1]-1 inclusive and
// vectors kfirst_slice [t] to klast_slice [t].  The first and last vectors
// may be shared with prior slices and subsequent slices.

// On input, ntasks is the # of tasks requested.

// A can have any sparsity structure (sparse, hyper, bitmap, or full).
// A may be jumbled.

#include "slice/GB_ek_slice.h"
#include "slice/include/GB_search_for_vector.h"

//------------------------------------------------------------------------------
// GB_ek_slice_search: find the first and last vectors in a slice
//------------------------------------------------------------------------------

static inline void GB_ek_slice_search
(
    // input:
    int taskid,
    int ntasks,
    const int64_t *restrict pstart_slice,    // size ntasks+1
    const int64_t *restrict Ap,              // size anvec
    int64_t anvec,                           // # of vectors in A
    int64_t avlen,                           // vector length of A
    // output:
    int64_t *restrict kfirst_slice,          // size ntasks
    int64_t *restrict klast_slice            // size ntasks
)
{
    int64_t pfirst = pstart_slice [taskid] ;
    int64_t plast  = pstart_slice [taskid+1] - 1 ;

    // find the first vector of the slice for task taskid: the
    // vector that owns the entry Ai [pfirst] and Ax [pfirst].
    int64_t kfirst ;
    if (taskid == 0)
    { 
        kfirst = 0 ;
    }
    else
    { 
        kfirst = GB_search_for_vector (pfirst, Ap, 0, anvec, avlen) ;
    }

    // find the last vector of the slice for task taskid: the
    // vector that owns the entry Ai [plast] and Ax [plast].
    int64_t klast ;
    if (taskid == ntasks-1)
    { 
        klast = anvec - 1 ;
    }
    else if (pfirst > plast)
    { 
        // this task does no work
        klast = kfirst ;
    }
    else
    { 
        klast = GB_search_for_vector (plast, Ap, kfirst, anvec, avlen) ;
    }
    kfirst_slice [taskid] = kfirst ;
    klast_slice  [taskid] = klast ;
}

//------------------------------------------------------------------------------
// GB_ek_slice: slice the entries and vectors of a matrix
//------------------------------------------------------------------------------

GB_CALLBACK_EK_SLICE_PROTO (GB_ek_slice)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A_ek_slicing != NULL) ;
    ASSERT (ntasks >= 1) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    ASSERT (GB_JUMBLED_OK (A)) ;    // pattern of A is not accessed

    int64_t anvec = A->nvec ;
    int64_t avlen = A->vlen ;
    int64_t anz = GB_nnz_held (A) ;
    const int64_t *Ap = A->p ;      // NULL if bitmap or full

    //--------------------------------------------------------------------------
    // allocate result
    //--------------------------------------------------------------------------

    // kfirst_slice and klast_slice are size ntasks.
    // pstart_slice is size ntasks+1

    int64_t *restrict kfirst_slice = A_ek_slicing ;
    int64_t *restrict klast_slice  = A_ek_slicing + ntasks ;
    int64_t *restrict pstart_slice = A_ek_slicing + ntasks * 2 ;

    //--------------------------------------------------------------------------
    // quick return for empty matrices
    //--------------------------------------------------------------------------

    if (anz == 0)
    { 
        // construct a single empty task
        ASSERT (ntasks == 1) ;
        pstart_slice [0] = 0 ;
        pstart_slice [1] = 0 ;
        kfirst_slice [0] = -1 ;
        klast_slice  [0] = -2 ;
        return ;
    }

    //--------------------------------------------------------------------------
    // find the first and last entries in each slice
    //--------------------------------------------------------------------------

    // FUTURE: this can be done in parallel if there are many tasks
    GB_e_slice (pstart_slice, anz, ntasks) ;

    //--------------------------------------------------------------------------
    // find the first and last vectors in each slice
    //--------------------------------------------------------------------------

    // The first vector of the slice is the kth vector of A if
    // pstart_slice [taskid] is in the range Ap [k]...A[k+1]-1, and this
    // is vector is k = kfirst_slice [taskid].

    // The last vector of the slice is the kth vector of A if
    // pstart_slice [taskid+1]-1 is in the range Ap [k]...A[k+1]-1, and this
    // is vector is k = klast_slice [taskid].

    // FUTURE: this can be done in parallel if there are many tasks
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    { 
        GB_ek_slice_search (taskid, ntasks, pstart_slice, Ap, anvec, avlen,
            kfirst_slice, klast_slice) ;
    }

    ASSERT (kfirst_slice [0] == 0) ;
    ASSERT (klast_slice  [ntasks-1] == anvec-1) ;
}

