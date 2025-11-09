//------------------------------------------------------------------------------
// GB_ek_slice: slice the entries and vectors of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Slice the entries of a matrix or vector into A_ntasks slices.

// The function is called GB_ek_slice because it first partitions the e entries
// into chunks of identical sizes, and then finds the first and last vector
// (k) for each chunk.

// Task t does entries pstart_slice [t] to pstart_slice [t+1]-1 inclusive and
// vectors kfirst_slice [t] to klast_slice [t].  The first and last vectors
// may be shared with prior slices and subsequent slices.

// On input, A_ntasks is the # of tasks requested.

// A can have any sparsity structure (sparse, hyper, bitmap, or full).
// A may be jumbled.

#include "GB.h"
#include "slice/include/GB_search_for_vector.h"

//------------------------------------------------------------------------------
// GB_ek_slice_search: find the first and last vectors in a slice
//------------------------------------------------------------------------------

#define GB_ek_slice_search_TYPE   GB_ek_slice_search_32
#define GB_search_for_vector_TYPE GB_search_for_vector_32
#include "slice/factory/GB_ek_slice_search_template.c"

#define GB_ek_slice_search_TYPE   GB_ek_slice_search_64
#define GB_search_for_vector_TYPE GB_search_for_vector_64
#include "slice/factory/GB_ek_slice_search_template.c"

//------------------------------------------------------------------------------
// GB_ek_slice: slice the entries and vectors of a matrix
//------------------------------------------------------------------------------

//  void GB_ek_slice                    // slice a matrix
//  (
//      // output:
//      int64_t *restrict A_ek_slicing, // size 3*A_ntasks+1
//      // input:
//      GrB_Matrix A,                   // matrix to slice
//      int A_ntasks                    // # of tasks
//  ) ;

GB_CALLBACK_EK_SLICE_PROTO (GB_ek_slice)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A_ek_slicing != NULL) ;
    ASSERT (A_ntasks >= 1) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    ASSERT (GB_JUMBLED_OK (A)) ;    // pattern of A is not accessed

    int64_t anvec = A->nvec ;
    int64_t avlen = A->vlen ;
    int64_t anz = GB_nnz_held (A) ;
    const void *Ap = A->p ;         // NULL if bitmap or full
    bool Ap_is_32 = A->p_is_32 ;

    //--------------------------------------------------------------------------
    // allocate result
    //--------------------------------------------------------------------------

    // kfirst_slice and klast_slice are size A_ntasks.
    // pstart_slice is size A_ntasks+1

    int64_t *restrict kfirst_slice = A_ek_slicing ;
    int64_t *restrict klast_slice  = A_ek_slicing + A_ntasks ;
    int64_t *restrict pstart_slice = A_ek_slicing + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // quick return for empty matrices
    //--------------------------------------------------------------------------

    if (anz == 0)
    { 
        // construct a single empty task
        ASSERT (A_ntasks == 1) ;
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
    GB_e_slice (pstart_slice, anz, A_ntasks) ;

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
    if (Ap_is_32)
    {
        for (int taskid = 0 ; taskid < A_ntasks ; taskid++)
        { 
            // using GB_search_for_vector_32 (...):
            GB_ek_slice_search_32 (taskid, A_ntasks, pstart_slice, Ap,
                anvec, avlen, kfirst_slice, klast_slice) ;
        }
    }
    else
    {
        for (int taskid = 0 ; taskid < A_ntasks ; taskid++)
        { 
            // using GB_search_for_vector_64 (...):
            GB_ek_slice_search_64 (taskid, A_ntasks, pstart_slice, Ap,
                anvec, avlen, kfirst_slice, klast_slice) ;
        }
    }

    ASSERT (kfirst_slice [0] == 0) ;
    ASSERT (klast_slice  [A_ntasks-1] == anvec-1) ;
}

