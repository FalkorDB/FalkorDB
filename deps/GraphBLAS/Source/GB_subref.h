//------------------------------------------------------------------------------
// GB_subref.h: definitions for GB_subref_* functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SUBREF_H
#define GB_SUBREF_H
#include "GB_ij.h"

GrB_Info GB_subref              // C = A(I,J): either symbolic or numeric
(
    // output
    GrB_Matrix C,               // output matrix, static header
    // input, not modified
    bool C_iso,                 // if true, return C as iso, regardless of A 
    const bool C_is_csc,        // requested format of C
    const GrB_Matrix A,
    const GrB_Index *I,         // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t ni,           // length of I, or special
    const GrB_Index *J,         // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t nj,           // length of J, or special
    const bool symbolic,        // if true, construct C as symbolic
    GB_Werk Werk
) ;

GrB_Info GB_subref_phase0
(
    // output
    int64_t *restrict *p_Ch,         // Ch = C->h hyperlist, or NULL standard
    size_t *p_Ch_size,
    int64_t *restrict *p_Ap_start,   // A(:,kA) starts at Ap_start [kC]
    size_t *p_Ap_start_size,
    int64_t *restrict *p_Ap_end,     // ... and ends at Ap_end [kC] - 1
    size_t *p_Ap_end_size,
    int64_t *p_Cnvec,       // # of vectors in C
    bool *p_need_qsort,     // true if C must be sorted
    int *p_Ikind,           // kind of I
    int64_t *p_nI,          // length of I
    int64_t Icolon [3],     // for GB_RANGE, GB_STRIDE
    int64_t *p_nJ,          // length of J
    // input, not modified
    const GrB_Matrix A,
    const GrB_Index *I,     // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t ni,       // length of I, or special
    const GrB_Index *J,     // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t nj,       // length of J, or special
//  const bool must_sort,   // true if C must be returned sorted
    GB_Werk Werk
) ;

GrB_Info GB_subref_slice    // phase 1 of GB_subref
(
    // output:
    GB_task_struct **p_TaskList,    // array of structs
    size_t *p_TaskList_size,        // size of TaskList
    int *p_ntasks,                  // # of tasks constructed
    int *p_nthreads,                // # of threads for subref operation
    bool *p_post_sort,              // true if a final post-sort is needed
    int64_t *restrict *p_Mark,      // for I inverse, if needed; size avlen
    size_t *p_Mark_size,
    int64_t *restrict *p_Inext,     // for I inverse, if needed; size nI
    size_t *p_Inext_size,
    int64_t *p_nduplicates,         // # of duplicates, if I inverse computed
    // from phase0:
    const int64_t *restrict Ap_start,   // location of A(imin:imax,kA)
    const int64_t *restrict Ap_end,
    const int64_t Cnvec,            // # of vectors of C
    const bool need_qsort,          // true if C must be sorted
    const int Ikind,                // GB_ALL, GB_RANGE, GB_STRIDE or GB_LIST
    const int64_t nI,               // length of I
    const int64_t Icolon [3],       // for GB_RANGE and GB_STRIDE
    // original input:
    const int64_t avlen,            // A->vlen
    const int64_t anz,              // nnz (A)
    const GrB_Index *I,
    GB_Werk Werk
) ;

GrB_Info GB_subref_phase2               // count nnz in each C(:,j)
(
    // computed by phase1:
    int64_t **Cp_handle,                // output of size Cnvec+1
    size_t *Cp_size_handle,
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C
    // tasks from phase0b:
    GB_task_struct *restrict TaskList,  // array of structs
    const int ntasks,                   // # of tasks
    const int nthreads,                 // # of threads to use
    const int64_t *Mark,                // for I inverse buckets, size A->vlen
    const int64_t *Inext,               // for I inverse buckets, size nI
    const int64_t nduplicates,          // # of duplicates, if I inverted
    // analysis from phase0:
    const int64_t *restrict Ap_start,
    const int64_t *restrict Ap_end,
    const int64_t Cnvec,
    const bool need_qsort,
    const int Ikind,
    const int64_t nI,
    const int64_t Icolon [3],
    // original input:
    const GrB_Matrix A,
    const GrB_Index *I,         // index list for C = A(I,J), or GrB_ALL, etc.
    const bool symbolic,
    GB_Werk Werk
) ;

GrB_Info GB_subref_phase3   // C=A(I,J)
(
    GrB_Matrix C,               // output matrix, static header
    // from phase1:
    int64_t **Cp_handle,        // vector pointers for C
    size_t Cp_size,
    const int64_t Cnvec_nonempty,       // # of non-empty vectors in C
    // from phase0b:
    const GB_task_struct *restrict TaskList,    // array of structs
    const int ntasks,                           // # of tasks
    const int nthreads,                         // # of threads to use
    const bool post_sort,               // true if post-sort needed
    const int64_t *Mark,                // for I inverse buckets, size A->vlen
    const int64_t *Inext,               // for I inverse buckets, size nI
    const int64_t nduplicates,          // # of duplicates, if I inverted
    // from phase0:
    int64_t **Ch_handle,
    size_t Ch_size,
    const int64_t *restrict Ap_start,
    const int64_t *restrict Ap_end,
    const int64_t Cnvec,
    const bool need_qsort,
    const int Ikind,
    const int64_t nI,
    const int64_t Icolon [3],
    const int64_t nJ,
    // from GB_subref:
    const bool C_iso,           // if true, C is iso
    const GB_void *cscalar,     // iso value of C
    // original input:
    const bool C_is_csc,        // format of output matrix C
    const GrB_Matrix A,
    const GrB_Index *I,
    const bool symbolic,
    GB_Werk Werk
) ;

GrB_Info GB_I_inverse           // invert the I list for C=A(I,:)
(
    const GrB_Index *I,         // list of indices, duplicates OK
    int64_t nI,                 // length of I
    int64_t avlen,              // length of the vectors of A
    // outputs:
    int64_t *restrict *p_Mark,  // head pointers for buckets, size avlen
    size_t *p_Mark_size,
    int64_t *restrict *p_Inext, // next pointers for buckets, size nI
    size_t *p_Inext_size,
    int64_t *p_ndupl,           // number of duplicate entries in I
    GB_Werk Werk
) ;

GrB_Info GB_bitmap_subref       // C = A(I,J): either symbolic or numeric
(
    // output
    GrB_Matrix C,               // output matrix, static header
    // input, not modified
    const bool C_iso,           // if true, C is iso
    const GB_void *cscalar,     // scalar value of C, if iso
    const bool C_is_csc,        // requested format of C
    const GrB_Matrix A,
    const GrB_Index *I,         // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t ni,           // length of I, or special
    const GrB_Index *J,         // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t nj,           // length of J, or special
    const bool symbolic,        // if true, construct C as symbolic
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subref_method: select a method for C(:,kC) = A(I,kA), for one vector of C
//------------------------------------------------------------------------------

// Determines the method used for to construct C(:,kC) = A(I,kA) for a
// single vector of C and A.

static inline int GB_subref_method  // return the method to use (1 to 12)
(
    // output
    int64_t *p_work,                // work required
    bool *p_this_needs_I_inverse,   // true if I needs to be inverted
    // input:
    const int64_t ajnz,             // nnz (A (:,j))
    const int64_t avlen,            // A->vlen
    const int Ikind,                // GB_ALL, GB_RANGE, GB_STRIDE, or GB_LIST
    const int64_t nI,               // length of I
    const bool I_inverse_ok,        // true if I is invertable 
    const bool need_qsort,          // true if C(:,k) requires sorting
    const int64_t iinc,             // increment for GB_STRIDE
    const int64_t nduplicates       // # of duplicates in I (zero if not known) 
)
{

    //--------------------------------------------------------------------------
    // initialize return values
    //--------------------------------------------------------------------------

    int method ;            // determined below
    bool this_needs_I_inverse = false ; // most methods do not need I inverse
    int64_t work ;          // most methods require O(nnz(A(:,j))) work

    //--------------------------------------------------------------------------
    // determine the method to use for C(:,j) = A (I,j)
    //--------------------------------------------------------------------------

    if (ajnz == avlen)
    {
        // A(:,j) is dense
        if (Ikind == GB_ALL)
        { 
            // Case 1: C(:,k) = A(:,j) are both dense
            method = 1 ;
            work = nI ;   // ajnz == avlen == nI
        }
        else
        { 
            // Case 2: C(:,k) = A(I,j), where A(:,j) is dense,
            // for Ikind == GB_RANGE, GB_STRIDE, or GB_LIST
            method = 2 ;
            work = nI ;
        }
    }
    else if (nI == 1)
    { 
        // Case 3: one index
        method = 3 ;
        work = 1 ;
    }
    else if (Ikind == GB_ALL)
    { 
        // Case 4: I is ":"
        method = 4 ;
        work = ajnz ;
    }
    else if (Ikind == GB_RANGE)
    { 
        // Case 5: C (:,k) = A (ibegin:iend,j)
        method = 5 ;
        work = ajnz ;
    }
    else if ((Ikind == GB_LIST && !I_inverse_ok) ||  // must do Case 6
        (64 * nI < ajnz))    // Case 6 faster
    { 
        // Case 6: nI not large; binary search of A(:,j) for each i in I
        method = 6 ;
        work = nI * 64 ;
    }
    else if (Ikind == GB_STRIDE)
    { 
        if (iinc >= 0)
        { 
            // Case 7: I = ibegin:iinc:iend with iinc >= 0
            method = 7 ;
            work = ajnz ;
        }
        else if (iinc < -1)
        { 
            // Case 8: I = ibegin:iinc:iend with iinc < =1
            method = 8 ;
            work = ajnz ;
        }
        else // iinc == -1
        { 
            // Case 9: I = ibegin:(-1):iend
            method = 9 ;
            work = ajnz ;
        }
    }
    else // Ikind == GB_LIST, and I inverse buckets will be used
    {
        // construct the I inverse buckets
        this_needs_I_inverse = true ;
        if (need_qsort)
        { 
            // Case 10: nI large, need qsort
            // duplicates are possible so cjnz > ajnz can hold.  If fine tasks
            // use this method, a post sort is needed when all tasks are done.
            method = 10 ;
            work = ajnz * 32 ;
        }
        else if (nduplicates > 0)
        { 
            // Case 11: nI large, no qsort, with duplicates
            // duplicates are possible so cjnz > ajnz can hold.  Note that the
            // # of duplicates is only known after I is inverted, which might
            // not yet be done.  In that case, nuplicates is assumed to be
            // zero, and Case 11 is assumed to be used instead.  This is
            // revised after I is inverted.
            method = 11 ;
            work = ajnz * 2 ;
        }
        else
        { 
            // Case 12: nI large, no qsort, no dupl
            method = 12 ;
            work = ajnz ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (p_work != NULL)
    { 
        (*p_work) = work ;
    }
    if (p_this_needs_I_inverse != NULL)
    { 
        (*p_this_needs_I_inverse) = this_needs_I_inverse ;
    }
    return (method) ;
}

#endif

