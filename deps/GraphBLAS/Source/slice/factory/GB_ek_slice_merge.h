//------------------------------------------------------------------------------
// GB_ek_slice_merge.h: slice the entries and vectors of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_EK_SLICE_MERGE_H
#define GB_EK_SLICE_MERGE_H

//------------------------------------------------------------------------------
// GB_ek_slice_merge* methods
//------------------------------------------------------------------------------

// GB_ek_slice slices the entries of a matrix or vector into A_ntasks slices.
// Its prototype is in Source/callback:

//  void GB_ek_slice                    // slice a matrix
//  (
//      // output:
//      int64_t *restrict A_ek_slicing, // size 3*A_ntasks+1
//      // input:
//      GrB_Matrix A,                   // matrix to slice
//      int A_ntasks                    // # of tasks
//  ) ;

// Task t does entries pstart_slice [t] to pstart_slice [t+1]-1 and
// vectors kfirst_slice [t] to klast_slice [t].  The first and last vectors
// may be shared with prior slices and subsequent slices.

// On input, A_ntasks must be <= nnz (A), unless nnz (A) is zero.  In that
// case, A_ntasks must be 1.

// GB_ek_slice can optionally be followed by GB_ek_slice_merge1 and
// GB_ek_slice_merge2, defined below, to finalize the work on the output matrix
// C->p, for sparse select and emult methods.

//------------------------------------------------------------------------------
// GB_ek_slice_merge1: merge column counts for a matrix
//------------------------------------------------------------------------------

// The input matrix A has been sliced via GB_ek_slice, and scanned to compute
// the counts of entries in each vector of C in Cp, Wfirst, and Wlast.  This
// phase finalizes the column counts, Cp, merging the results of each task.

// On input, Cp [k] has been partially computed.  Task tid operators on vector
// kfirst = kfirst_Aslice [tid] to klast = klast_Aslice [tid].  If kfirst < k <
// klast, then Cp [k] is the total count of entries in C(:,k).  Otherwise, the
// counts are held in Wfirst and Wlast, and Cp [k] is zero (or uninititalized).
// Wfirst [tid] is the number of entries in C(:,kfirst) constructed by task
// tid, and Wlast [tid] is the number of entries in C(:,klast) constructed by
// task tid.

// This function sums up the entries computed for C(:,k) by all tasks, so that
// on output, Cp [k] is the total count of entries in C(:,k).

static inline void GB_ek_slice_merge1   // merge column counts for the matrix C
(
    // input/output:
    void *Cp,                           // column counts
    // input:
    const bool Cp_is_32,                // if true, Cp is 32-bit; else 64
    const uint64_t *restrict Wfirst,    // size A_ntasks
    const uint64_t *restrict Wlast,     // size A_ntasks
    const int64_t *A_ek_slicing,        // size 3*A_ntasks+1
    const int A_ntasks                  // # of tasks to slice A
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_IDECL (Cp, , u) ; GB_IPTR (Cp, Cp_is_32) ;
    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
//  const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // merge column counts
    //--------------------------------------------------------------------------

    int64_t kprior = -1 ;

    for (int tid = 0 ; tid < A_ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // sum up the partial result that thread tid computed for kfirst
        //----------------------------------------------------------------------

        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;

        if (kfirst <= klast)
        {
            uint64_t c = Wfirst [tid] ;
            if (kprior < kfirst)
            { 
                // This thread is the first one that did work on
                // A(:,kfirst), so use it to start the reduction.
                // Cp [kfirst] = Wfirst [tid] ;
            }
            else
            { 
                // Cp [kfirst] += Wfirst [tid] ;
                c += GB_IGET (Cp, kfirst) ;
            }
            // Cp [kfirst] = c ;
            GB_ISET (Cp, kfirst, c) ;
            kprior = kfirst ;
        }

        //----------------------------------------------------------------------
        // sum up the partial result that thread tid computed for klast
        //----------------------------------------------------------------------

        if (kfirst < klast)
        { 
            ASSERT (kprior < klast) ;
            // This thread is the first one that did work on
            // A(:,klast), so use it to start the reduction.
            // Cp [klast] = Wlast [tid] ;
            uint64_t c = Wlast [tid] ;
            GB_ISET (Cp, klast, c) ;
            kprior = klast ;
        }
    }
}


//------------------------------------------------------------------------------
// GB_ek_slice_merge2: merge final results for matrix C
//------------------------------------------------------------------------------

// Prior to calling this function, a method using GB_ek_slice to slice an input
// matrix A has computed the vector counts Cp, where Cp [k] is the number of
// entries in the kth vector of C on input to this function.

// The input matrix and the matrix C is sliced by kfirst_Aslice and
// klast_Aslice, where kfirst = kfirst_Aslice [tid] is the first vector in A
// and C computed by task tid, and klast = klast_Aslice [tid] is the last
// vector computed by task tid.  Tasks tid and tid+1 may cooperate on a single
// vector, however, where klast_Aslice [tid] may be the same as kfirst_Aslice
// [tid].  The method has also computed Wfirst [tid] and Wlast [tid] for each
// task id, tid.  Wfirst [tid] is the number of entries task tid contributes to
// C(:,kfirst), and Wlast [tid] is the number of entries task tid contributes
// to C(:,klast).

// Cp_kfirst [tid] is the position in C where task tid owns entries in
// C(:,kfirst), which is a cumulative sum of the entries computed in C(:,k) for
// all tasks that cooperate to compute that vector, starting at Cp [kfirst].
// There is no need to compute this for C(:,klast):  if kfirst < klast, then
// either task tid fully owns C(:,klast), or it is always the first task to
// operate on C(:,klast).  In both cases, task tid starts its computations at
// the top of C(:,klast), which can be found at Cp [klast].

static inline void GB_ek_slice_merge2   // merge final results for matrix C
(
    // output:
    uint64_t *restrict Cp_kfirst,       // size A_ntasks
    // input:
    const void *Cp,                     // size C->nvec+1
    const bool Cp_is_32,                // if true, Cp is 32-bit; else 64
    const uint64_t *restrict Wfirst,    // size A_ntasks
    const uint64_t *restrict Wlast,     // size A_ntasks
    const int64_t *A_ek_slicing,        // size 3*A_ntasks+1
    const int A_ntasks                  // # of tasks to slice A and construct C
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_IDECL (Cp, const, u) ; GB_IPTR (Cp, Cp_is_32) ;
    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
//  const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // determine the slice boundaries in the new C matrix
    //--------------------------------------------------------------------------

    int64_t kprior = -1 ;
    uint64_t pC = 0 ;

    for (int tid = 0 ; tid < A_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Aslice [tid] ;

        if (kprior < kfirst)
        { 
            // Task tid is the first one to do work on C(:,kfirst), so it
            // starts at Cp [kfirst], and it contributes Wfirst [tid] entries
            // to C(:,kfirst).
            pC = GB_IGET (Cp, kfirst) ;
            kprior = kfirst ;
        }

        // Task tid contributes Wfirst [tid] entries to C(:,kfirst)
        Cp_kfirst [tid] = pC ;
        pC += Wfirst [tid] ;

        int64_t klast = klast_Aslice [tid] ;
        if (kfirst < klast)
        { 
            // Task tid is the last to contribute to C(:,kfirst).
            ASSERT (pC == GB_IGET (Cp, kfirst+1)) ;
            // Task tid contributes the first Wlast [tid] entries to
            // C(:,klast), so the next task tid+1 starts at location Cp [klast]
            // + Wlast [tid], if its first vector is klast of this task.
            pC = GB_IGET (Cp, klast) + Wlast [tid] ;
            kprior = klast ;
        }
    }
}

#endif

