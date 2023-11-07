//------------------------------------------------------------------------------
// GB_ek_slice_merge2: merge final results for matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

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

// On output, Cp [0:cnvec] is overwritten with its cumulative sum.
// C_nvec_nonempty is the count of how many vectors in C are empty.
// Cp_kfirst [tid] is the position in C where task tid owns entries in
// C(:,kfirst), which is a cumulative sum of the entries computed in C(:,k) for
// all tasks that cooperate to compute that vector, starting at Cp [kfirst].
// There is no need to compute this for C(:,klast):  if kfirst < klast, then
// either task tid fully owns C(:,klast), or it is always the first task to
// operate on C(:,klast).  In both cases, task tid starts its computations at
// the top of C(:,klast), which can be found at Cp [klast].

#include "GB_ek_slice.h"

void GB_ek_slice_merge2     // merge final results for matrix C
(
    // output
    int64_t *C_nvec_nonempty,           // # of non-empty vectors in C
    int64_t *restrict Cp_kfirst,     // size ntasks
    // input/output
    int64_t *restrict Cp,            // size cnvec+1
    // input
    const int64_t cnvec,
    const int64_t *restrict Wfirst,          // size ntasks
    const int64_t *restrict Wlast,           // size ntasks
    const int64_t *A_ek_slicing,        // size 3*ntasks+1
    const int ntasks,                   // # of tasks used to construct C
    const int nthreads,                 // # of threads to use
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // Cp = cumsum (Cp)
    //--------------------------------------------------------------------------

    GB_cumsum (Cp, cnvec, C_nvec_nonempty, nthreads, Werk) ;

    //--------------------------------------------------------------------------
    // determine the slice boundaries in the new C matrix
    //--------------------------------------------------------------------------

    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + ntasks ;
//  const int64_t *restrict pstart_Aslice = A_ek_slicing + ntasks * 2 ;

    int64_t kprior = -1 ;
    int64_t pC = 0 ;

    for (int tid = 0 ; tid < ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Aslice [tid] ;

        if (kprior < kfirst)
        { 
            // Task tid is the first one to do work on C(:,kfirst), so it
            // starts at Cp [kfirst], and it contributes Wfirst [tid] entries
            // to C(:,kfirst).
            pC = Cp [kfirst] ;
            kprior = kfirst ;
        }

        // Task tid contributes Wfirst [tid] entries to C(:,kfirst)
        Cp_kfirst [tid] = pC ;
        pC += Wfirst [tid] ;

        int64_t klast = klast_Aslice [tid] ;
        if (kfirst < klast)
        { 
            // Task tid is the last to contribute to C(:,kfirst).
            ASSERT (pC == Cp [kfirst+1]) ;
            // Task tid contributes the first Wlast [tid] entries to
            // C(:,klast), so the next task tid+1 starts at location Cp [klast]
            // + Wlast [tid], if its first vector is klast of this task.
            pC = Cp [klast] + Wlast [tid] ;
            kprior = klast ;
        }
    }
}

