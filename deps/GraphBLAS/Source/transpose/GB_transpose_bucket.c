//------------------------------------------------------------------------------
// GB_transpose_bucket: transpose and optionally typecast and/or apply operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A' or op(A').  Optionally typecasts from A->type to the new type ctype,
// and/or optionally applies a unary operator.

// If an operator z=op(x) is provided, the type of z must be the same as the
// type of C.  The type of A must be compatible with the type of of x (A is
// typecasted into the type of x).  These conditions must be checked in the
// caller.

// This function is agnostic for the CSR/CSC format of C and A.  C_is_csc is
// defined by the caller and assigned to C->is_csc, but otherwise unused.
// A->is_csc is ignored.

// The input can be hypersparse or non-hypersparse.  The output C is always
// non-hypersparse, and never shallow.  On input, C is a static header.

// If A is m-by-n in CSC format, with e nonzeros, the time and memory taken is
// O(m+n+e) if A is non-hypersparse, or O(m+e) if hypersparse.  This is fine if
// most rows and columns of A are non-empty, but can be very costly if A or A'
// is hypersparse.  In particular, if A is a non-hypersparse column vector with
// m >> e, the time and memory is O(m), which can be huge.  Thus, for
// hypersparse matrices, or for very sparse matrices, the qsort method should
// be used instead (see GB_transpose).

// This method is parallel, but not highly scalable.  At most O(e/m) threads
// are used.

#include "transpose/GB_transpose.h"

#define GB_FREE_WORKSPACE                                               \
{                                                                       \
    if (Workspaces != NULL && Workspaces_size != NULL)                  \
    {                                                                   \
        for (int tid = 0 ; tid < nworkspaces ; tid++)                   \
        {                                                               \
            GB_FREE_MEMORY (&(Workspaces [tid]), Workspaces_size [tid]) ; \
        }                                                               \
    }                                                                   \
    GB_WERK_POP (A_slice, int64_t) ;                                    \
    GB_WERK_POP (Workspaces_size, size_t) ;                             \
    GB_WERK_POP (Workspaces, void *) ;                                  \
}

#define GB_FREE_ALL                                                     \
{                                                                       \
    GB_phybix_free (C) ;                                                \
    GB_FREE_WORKSPACE ;                                                 \
}

GrB_Info GB_transpose_bucket    // bucket transpose; typecast and apply op
(
    GrB_Matrix C,               // output matrix (static header)
    const GB_iso_code C_code_iso,   // iso code for C
    const GrB_Type ctype,       // type of output matrix C
    const bool C_is_csc,        // format of output matrix C
    const GrB_Matrix A,         // input matrix
        // no operator is applied if op is NULL
        const GB_Operator op,       // unary/idxunop/binop to apply
        const GrB_Scalar scalar,    // scalar to bind to binary operator
        bool binop_bind1st,         // if true, binop(x,A) else binop(A,y)
    const int nworkspaces,      // # of workspaces to use
    const int nthreads,         // # of threads to use
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C is an empty header and not yet allocated
    ASSERT (C != NULL && (C->header_size == 0 || GBNSTATIC)) ;
    ASSERT_TYPE_OK (ctype, "ctype for transpose", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for transpose_bucket", GB0) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;

    // if op is NULL, then no operator is applied

    // This method is only be used when A is sparse or hypersparse.
    // The full and bitmap cases are handled in GB_transpose.
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;

    GB_WERK_DECLARE (A_slice, int64_t) ;            // size nthreads+1
    GB_WERK_DECLARE (Workspaces, void *) ;          // size nworkspaces
    GB_WERK_DECLARE (Workspaces_size, size_t) ;     // size nworkspaces

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    GrB_Info info ;
    int64_t anz = GB_nnz (A) ;
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    // # of threads to use in the O(avlen) loops below
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nth = GB_nthreads (avlen, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // allocate C: always sparse
    //--------------------------------------------------------------------------

    // The bucket transpose only works when C is sparse.
    // A can be sparse or hypersparse.

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new matrix
    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_SPARSE, anz, avdim, avlen, Werk) ;

    // C->p is allocated but not initialized.
    bool C_iso = (C_code_iso != GB_NON_ISO) ;
    GB_OK (GB_new_bix (&C, // sparse, existing header
        ctype, avdim, avlen, GB_ph_malloc, C_is_csc, GxB_SPARSE, true,
        A->hyper_switch, avlen, anz, true, C_iso,
        Cp_is_32, Cj_is_32, Ci_is_32)) ;

    C->nvals = anz ;
    size_t cpsize = (Cp_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    GB_WERK_PUSH (Workspaces, nworkspaces, void *) ;
    GB_WERK_PUSH (Workspaces_size, nworkspaces, size_t) ;
    if (Workspaces == NULL || Workspaces_size == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    bool ok = true ;
    for (int tid = 0 ; tid < nworkspaces ; tid++)
    { 
        // each workspace has the same size integer as Cp
        Workspaces [tid] = GB_MALLOC_MEMORY (avlen + 1, cpsize,
            &Workspaces_size [tid]) ;
        ok = ok && (Workspaces [tid] != NULL) ;
    }

    if (!ok)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // phase1: symbolic analysis
    //--------------------------------------------------------------------------

    // slice the A matrix, perfectly balanced for one task per thread
    GB_WERK_PUSH (A_slice, nthreads + 1, int64_t) ;
    if (A_slice == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_p_slice (A_slice, Ap, A->p_is_32, A->nvec, nthreads, true) ;

    // sum up the row counts and find C->p
    if (Cp_is_32)
    {
        #define GB_Cp_TYPE uint32_t
        #include "transpose/factory/GB_transpose_bucket_template.c"
    }
    else
    {
        #define GB_Cp_TYPE uint64_t
        #include "transpose/factory/GB_transpose_bucket_template.c"
    }

    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // phase2: transpose A into C
    //--------------------------------------------------------------------------

    // transpose both the pattern and the values
    if (op == NULL)
    { 
        // do not apply an operator; optional typecast to C->type
        GB_OK (GB_transpose_ix (C, A, Workspaces, A_slice, nworkspaces,
            nthreads)) ;
    }
    else
    { 
        // apply an operator, C has type op->ztype
        GB_OK (GB_transpose_op (C, C_code_iso, op, scalar, binop_bind1st, A,
            Workspaces, A_slice, nworkspaces, nthreads)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (C, "C transpose of A", GB0) ;
    ASSERT (C->h == NULL) ;
    return (GrB_SUCCESS) ;
}

