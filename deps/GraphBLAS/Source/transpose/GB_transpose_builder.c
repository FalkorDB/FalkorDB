//------------------------------------------------------------------------------
// GB_transpose_builder: C=A' or C=op(A'), with typecasting, via builder
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// CALLS:     GB_builder

// Transpose a matrix, C=A', using the GB_builder method, and optionally apply
// a unary operator and/or typecast the values.  The result is returned in the
// T = (*Thandle) output, which later transplanted into the C matrix by the
// caller, GB_transpose.

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_FREE_MEMORY (&iwork, iwork_mem) ;    \
    GB_FREE_MEMORY (&jwork, jwork_mem) ;    \
    GB_FREE_MEMORY (&Swork, Swork_mem) ;    \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORKSPACE ;                     \
    GB_Matrix_free (Thandle) ;              \
}

#include "transpose/GB_transpose.h"
#include "builder/GB_build.h"
#include "apply/GB_apply.h"
#include "extractTuples/GB_extractTuples.h"

//------------------------------------------------------------------------------
// GB_transpose_builder
//------------------------------------------------------------------------------

GrB_Info GB_transpose_builder       // T=A', T=(ctype)A' or T=op(A')
(
    GrB_Matrix *Thandle,        // output matrix T, header allocated on input
    GrB_Type ctype,             // desired type of T
    const bool C_is_csc,        // desired CSR/CSC format of C and T
    const bool C_iso,           // true if C (and T) is iso
    const GB_iso_code C_code_iso,   // iso code for C and T
    const GrB_Matrix A,         // input matrix; C == A if done in place
    const bool in_place,        // true if C and A are the same matrix
        // no operator is applied if op is NULL
        const GB_Operator op,       // unary/idxunop/binop to apply
        const GrB_Scalar scalar,    // scalar to bind to binary operator
        bool binop_bind1st,         // if true, binop(x,A) else binop(A,y)
        bool flipij,                // if true, flip i,j for user idxunop
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (Thandle != NULL) ;
    GrB_Matrix T = (*Thandle) ;     // just the header of T is given on input
    ASSERT (T != NULL) ;
    int memlane = GB_memlane (T->header_mem) ;
    uint64_t mem = GB_mem (memlane, 0) ;

    void *iwork = NULL ; uint64_t iwork_mem = mem ;
    void *jwork = NULL ; uint64_t jwork_mem = mem ;
    GB_void *Swork = NULL ; uint64_t Swork_mem = mem ;

    GrB_Type atype = A->type ;
    int64_t anz = GB_nnz (A) ;
    bool Ap_is_32 = A->p_is_32 ;
    bool Aj_is_32 = A->j_is_32 ;
    bool Ai_is_32 = A->i_is_32 ;
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;
    float A_hyper_switch = A->hyper_switch ;

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;

//  size_t apsize = (Ap_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t ajsize = (Aj_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t aisize = (Ai_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

    size_t csize = ctype->size ;

    //--------------------------------------------------------------------------
    // allocate and create iwork
    //--------------------------------------------------------------------------

    // allocate iwork of size anz
    iwork = GB_MALLOC_MEMORY (anz, ajsize, &iwork_mem) ;
    if (iwork == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // Construct the "row" indices of C, which are "column" indices of
    // A.  This array becomes the permanent T->i on output.
    GB_OK (GB_extract_vector_list (iwork, Aj_is_32, A, Werk)) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix and additional space (jwork and Swork)
    //------------------------------------------------------------------

    // T is created using the requested integers of C.
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_HYPERSPARSE, anz, avdim, avlen, Werk) ;

    // initialize the header of T, with no content,
    // and initialize the type and dimension of T.

    info = GB_new (Thandle, // hyper, existing header
        ctype, avdim, avlen, GB_ph_null, C_is_csc,
        GxB_HYPERSPARSE, A_hyper_switch, 0,
        Cp_is_32, Cj_is_32, Ci_is_32, memlane) ;
    ASSERT (info == GrB_SUCCESS) ;

    // if in_place, the prior A->p and A->h can now be freed
    if (in_place)
    { 
        if (!A->p_shallow) GB_FREE_MEMORY (&A->p, A->p_mem) ;
        if (!A->h_shallow) GB_FREE_MEMORY (&A->h, A->h_mem) ;
    }

    GB_void *S_input = NULL ;

    // for the GB_builder method, if the transpose is done in-place and
    // A->i is not shallow, A->i can be used and then freed.
    // Otherwise, A->i is not modified at all.
    bool ok = true ;
    bool recycle_Ai = (in_place && !A->i_shallow) ;
    if (!recycle_Ai)
    { 
        // allocate jwork of size anz
        jwork = GB_MALLOC_MEMORY (anz, aisize, &jwork_mem) ;
        ok = ok && (jwork != NULL) ;
    }

    if (op != NULL && !C_iso)
    { 
        Swork = (GB_void *) GB_XALLOC_MEMORY (false, C_iso, anz, csize,
            &Swork_mem) ;
        ok = ok && (Swork != NULL) ;
    }

    if (!ok)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //------------------------------------------------------------------
    // construct jwork and Swork
    //------------------------------------------------------------------

    // "row" indices of A become "column" indices of C
    if (recycle_Ai)
    { 
        // A->i is used as workspace for the "column" indices of C.
        // jwork is A->i, and is freed by GB_builder.
        jwork = A->i ;
        jwork_mem = A->i_mem ;
        A->i = NULL ;
        ASSERT (in_place) ;
    }
    else
    { 
        // copy A->i into jwork, making a deep copy.  jwork is freed by
        // GB_builder.  A->i is not modified, even if out of memory.
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        GB_memcpy (jwork, A->i, anz * aisize, nthreads_max) ;
    }

    // numerical values: apply the op, typecast, or make shallow copy
    GrB_Type stype ;
    GB_void sscalar [GB_VLA(csize)] ;
    if (C_iso)
    { 
        // apply the op to the iso scalar
        GB_unop_iso (sscalar, ctype, C_code_iso, op, A, scalar) ;
        S_input = sscalar ;     // S_input is used instead of Swork
        Swork = NULL ;
        stype = ctype ;
    }
    else if (op != NULL)
    { 
        // Swork = op (A)
        info = GB_apply_op (Swork, ctype, C_code_iso, op, scalar,
            binop_bind1st, flipij, A, Werk) ;
        ASSERT (info == GrB_SUCCESS) ;
        // GB_builder will not need to typecast Swork to T->x, and it
        // may choose to transplant it into T->x
        S_input = NULL ;        // Swork is used instead of S_input
        stype = ctype ;
    }
    else
    { 
        // GB_builder will typecast S_input from atype to ctype if
        // needed.  S_input is a shallow copy of Ax, and must not be
        // modified.
        ASSERT (!C_iso) ;
        ASSERT (!A->iso) ;
        S_input = (GB_void *) A->x ; // S_input is used instead of Swork
        Swork = NULL ;
        stype = atype ;
    }

    //------------------------------------------------------------------
    // build the matrix: T = (ctype) A' or op ((xtype) A')
    //------------------------------------------------------------------

    // internally, jwork is freed and then T->x is allocated, so the
    // total memory usage is anz * max (csize, sizeof(aisize)).  T is
    // always hypersparse.  Either T, Swork, and S_input are all iso,
    // or all non-iso, depending on C_iso.

    GB_OK (GB_builder (
        T,          // create T using an existing header
        ctype,      // T is of type ctype
        avdim,      // T->vlen = A->vdim, always > 1
        avlen,      // T->vdim = A->vlen, always > 1
        C_is_csc,   // T has the same CSR/CSC format as C
        &iwork,     // iwork_handle, becomes T->i on output
        &iwork_mem,
        &jwork,     // jwork_handle, freed on output
        &jwork_mem,
        &Swork,     // Swork_handle, freed on output
        &Swork_mem,
        false,      // tuples are not sorted on input
        true,       // tuples have no duplicates
        anz,        // size of iwork, jwork, and Swork
        true,       // is_matrix: unused
        NULL, NULL, // original I,J indices: not used here
        S_input,    // array of values of type stype, not modified
        C_iso,      // iso property of T is the same as C->iso
        anz,        // number of tuples
        NULL,       // no dup operator needed (input has no duplicates)
        stype,      // type of S_input or Swork
        false,      // no burble (already burbled above)
        Werk,
        Aj_is_32, Ai_is_32, // integer sizes of iwork and jwork
        Cp_is_32, Cj_is_32, Ci_is_32  // integer sizes for T 
    )) ;

    //------------------------------------------------------------------
    // return result
    //------------------------------------------------------------------

    // GB_builder always frees jwork, and either frees iwork or
    // transplants it in to T->i and sets iwork to NULL.  So iwork and
    // jwork are always NULL on output.  GB_builder does not modify
    // S_input.
    ASSERT (iwork == NULL && jwork == NULL && Swork == NULL) ;
    ASSERT (!GB_JUMBLED (T)) ;
    return (GrB_SUCCESS) ;
}

