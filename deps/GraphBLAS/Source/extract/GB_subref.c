//------------------------------------------------------------------------------
// GB_subref: C = A(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C=A(I,J), either symbolic or numeric.  In a symbolic extraction, Cx [p] is
// not the value of A(i,j), but its position in Ai,Ax.  That is, pA = Cx [p]
// means that the entry at position p in C is the same as the entry in A at
// position pA.  In this case, Cx has a type of int64_t.

// Numeric extraction: C is iso if A is iso or C_iso is true on input

//      Sparse submatrix reference, C = A(I,J), extracting the values.  This is
//      an internal function called by GB_extract with symbolic==false, which
//      does the work of the user-callable GrB_*_extract methods.  It is also
//      called by GB_assign to extract the submask.  No pending tuples or
//      zombies appear in A.

// Symbolic extraction:  C is never iso

//      Sparse submatrix reference, C = A(I,J), extracting the pattern, not the
//      values.  For the symbolic case, this function is called only by
//      GB_subassign_symbolic.  Symbolic extraction creates a matrix C with the
//      same pattern (C->p and C->i) as numeric extraction, but with different
//      values, C->x.  For numeric extracion if C(inew,jnew) = A(i,j), the
//      value of A(i,j) is copied into C(i,j).  For symbolic extraction, its
//      *pointer* is copied into C(i,j).  Suppose an entry A(i,j) is held in
//      Ai [pa] and Ax [pa], and it appears in the output matrix C in Ci [pc]
//      and Cx [pc].  Then the two methods differ as follows:

//          this is the same:

//          i = Ai [pa] ;           // index i of entry A(i,j)
//          aij = Ax [pa] ;         // value of the entry A(i,j)
//          Ci [pc] = inew ;        // index inew of C(inew,jnew)

//          this is different:

//          Cx [pc] = aij ;         // for numeric extraction
//          Cx [pc] = pa ;          // for symbolic extraction

//      This function is called with symbolic==true by only by
//      GB_subassign_symbolic, which uses it to extract the pattern of C(I,J),
//      for the submatrix assignment C(I,J)=A.  In this case, this function
//      needs to deal with zombie entries.  GB_subassign_symbolic uses this
//      function on its C matrix, which is called A here because it is not
//      modified here.

//      Reading a zombie entry:  A zombie entry A(i,j) has been marked by
//      GB_ZOMBIE on its index.  The value of a zombie is not important, just
//      its presence in the pattern.  All zombies have been marked with
//      GB_ZOMBIE so that i < 0, and all regular entries are not marked as
//      zombies (i >= 0).  Zombies are entries that have been marked for
//      deletion but have not been removed from the matrix yet, since it's more
//      efficient to delete zombies all at once rather than one at a time.

//      The symbolic case is zombie-agnostic, in the sense that it does not
//      delete them.  It treats them like regular entries.  However, their
//      normal index must be used, not their GB_ZOMBIE'd indices.  The output
//      matrix C contains all non-zombie indices, and its references to zombies
//      and regular entries are identical.  Zombies in A are dealt with later.
//      They cannot be detected in the output C matrix, but they can be
//      detected in A.  Since pa = Cx [pc] holds the position of the entry in
//      A, the entry is a zombie if Ai [pa] has been marked with GB_ZOMBIE.

//      For symbolic extractionm, pending tuples can appear in the input matrix
//      A.  These are ignored.

#define GB_FREE_WORKSPACE                       \
{                                               \
    GB_FREE_MEMORY (&TaskList, TaskList_size) ; \
    GB_FREE_MEMORY (&Ap_start, Ap_start_size) ; \
    GB_FREE_MEMORY (&Ap_end, Ap_end_size) ;     \
    GB_FREE_MEMORY (&Cwork, Cwork_size) ;       \
    GB_Matrix_free (&R) ;                       \
}

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE_MEMORY (&Cp, Cp_size) ;     \
    GB_FREE_MEMORY (&Ch, Ch_size) ;     \
    GB_phybix_free (C) ;                \
    GB_FREE_WORKSPACE ;                 \
}

#include "extract/GB_subref.h"

GrB_Info GB_subref              // C = A(I,J): either symbolic or numeric
(
    // output
    GrB_Matrix C,               // output matrix, static header
    // input, not modified
    bool C_iso,                 // if true, return C as iso, regardless of A
    const bool C_is_csc,        // requested format of C
    const GrB_Matrix A,
    const void *I,              // index list for C = A(I,J), or GrB_ALL, etc.
    const bool I_is_32,         // if true, I is 32-bit; else 64-bit
    const int64_t ni,           // length of I, or special
    const void *J,              // index list for C = A(I,J), or GrB_ALL, etc.
    const bool J_is_32,         // if true, I is 32-bit; else 64-bit
    const int64_t nj,           // length of J, or special
    const bool symbolic,        // if true, construct C as symbolic
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->header_size == 0 || GBNSTATIC)) ;
    ASSERT_MATRIX_OK (A, "A for C=A(I,J) subref", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;    // A is sorted, below, if jumbled on input
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // determine the type of C
    //--------------------------------------------------------------------------

    GrB_Type ctype ;
    if (symbolic)
    {
        // select the integer type for symbolic subref, based on nnz (A)
        int64_t anz = GB_nnz_held (A) ;
        ctype = (anz <= UINT32_MAX) ? GrB_UINT32 : GrB_UINT64 ;
    }
    else
    {
        // for all other cases, C is given the same type as A.
        ctype = A->type ;
    }

    //--------------------------------------------------------------------------
    // check if C is iso and get its iso value
    //--------------------------------------------------------------------------

    size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    memset (cscalar, 0, csize) ;
    if (symbolic)
    { 
        // symbolic extraction never results in an iso matrix
        C_iso = false ;
        ASSERT (GB_ZOMBIES_OK (A)) ;
    }
    else
    {
        // determine if C is iso and get the iso scalar
        if (A->iso)
        { 
            memcpy (cscalar, A->x, csize) ;
            C_iso = true ;
        }
        ASSERT (!GB_ZOMBIES (A)) ;
    }

    if (C_iso)
    { 
        GBURBLE ("(iso subref) ") ;
    }

    //--------------------------------------------------------------------------
    // handle bitmap and full cases
    //--------------------------------------------------------------------------

    if (GB_IS_BITMAP (A) || GB_IS_FULL (A))
    { 
        // C is constructed with same sparsity as A (bitmap or full)
        return (GB_bitmap_subref (C, ctype, C_iso, cscalar, C_is_csc, A,
            I, I_is_32, ni, J, J_is_32, nj, symbolic, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // C = A(I,J) where C and A are both sparse or hypersparse
    //--------------------------------------------------------------------------

    void *Cp       = NULL ; size_t Cp_size = 0 ;
    void *Ch       = NULL ; size_t Ch_size = 0 ;
    void *Ap_start = NULL ; size_t Ap_start_size = 0 ;
    void *Ap_end   = NULL ; size_t Ap_end_size = 0 ;
    uint64_t *Cwork = NULL ; size_t Cwork_size = 0 ;
    GB_task_struct *TaskList = NULL ; size_t TaskList_size = 0 ;
    int64_t Cnvec = 0, nI = 0, nJ, Icolon [3], Cnvec_nonempty ;
    bool post_sort, need_qsort, Cp_is_32, Cj_is_32, Ci_is_32 ;
    int Ikind, ntasks, nthreads ;
    GrB_Matrix R = NULL ;

    //--------------------------------------------------------------------------
    // ensure A is unjumbled
    //--------------------------------------------------------------------------

    // Ensure input matrix is not jumbled.  Zombies are OK.
    // Pending tuples are OK (and ignored) for symbolic extraction.
    // GB_subref_phase0 may build the hyper_hash.
    GB_UNJUMBLE (A) ;
    ASSERT (!GB_JUMBLED (A)) ;

    //--------------------------------------------------------------------------
    // phase0: find vectors for C=A(I,J), and I,J properties
    //--------------------------------------------------------------------------

    GB_OK (GB_subref_phase0 (
        // computed by phase0:
        &Ch, &Cj_is_32, &Ci_is_32, &Ch_size, &Ap_start, &Ap_start_size,
        &Ap_end, &Ap_end_size, &Cnvec, &need_qsort, &Ikind, &nI, Icolon, &nJ,
        // original input:
        A, I, I_is_32, ni, J, J_is_32, nj, Werk)) ;

    //--------------------------------------------------------------------------
    // phase1: split C=A(I,J) into tasks for phase2 and phase3
    //--------------------------------------------------------------------------

    // Cwork is allocated; it either becomes Cp, or it is freed by phase2
    // This phase also inverts I if needed.

    GB_OK (GB_subref_slice (
        // computed by phase1:
        &TaskList, &TaskList_size, &ntasks, &nthreads, &post_sort,
        &R, &Cwork, &Cwork_size,
        // computed by phase0:
        Ap_start, Ap_end, Cnvec, need_qsort, Ikind, nI, Icolon,
        // original input:
        A->vlen, GB_nnz (A), A->p_is_32, I, I_is_32, Werk)) ;

    //--------------------------------------------------------------------------
    // phase2: count the number of entries in each vector of C
    //--------------------------------------------------------------------------

    GB_OK (GB_subref_phase2 (
        // computed by phase2:
        &Cp, &Cp_is_32, &Cp_size, &Cnvec_nonempty,
        // computed by phase1:
        TaskList, ntasks, nthreads, R, &Cwork, Cwork_size,
        // computed by phase0:
        Ap_start, Ap_end, Cnvec, need_qsort, Ikind, nI, Icolon, nJ,
        // original input:
        A, I, I_is_32, symbolic, Werk)) ;

    //--------------------------------------------------------------------------
    // phase3: compute the entries (indices and values) in each vector of C
    //--------------------------------------------------------------------------

    GB_OK (GB_subref_phase3 (
        // computed by phase3:
        C,
        // from phase2:
        &Cp, Cp_is_32, Cp_size, Cnvec_nonempty,
        // from phase1:
        TaskList, ntasks, nthreads, post_sort, R,
        // from phase0:
        &Ch, Cj_is_32, Ci_is_32, Ch_size, Ap_start, Ap_end, Cnvec, need_qsort,
        Ikind, nI, Icolon, nJ,
        // from GB_subref, above:
        ctype, C_iso, cscalar,
        // original input:
        C_is_csc, A, I, I_is_32, symbolic, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    // Cp and Ch have been imported into C->p and C->h, or freed if phase3
    // fails.  Either way, Cp and Ch are set to NULL so that they cannot be
    // freed here (except by freeing C itself).

    // free workspace
    GB_FREE_WORKSPACE ;

    // C can be returned jumbled, even if A is not jumbled
    ASSERT_MATRIX_OK (C, "C output for C=A(I,J)", GB0) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    return (GrB_SUCCESS) ;
}

