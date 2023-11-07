//------------------------------------------------------------------------------
// GB_kroner: Kronecker product, C = kron (A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// C = kron(A,B) where op determines the binary multiplier to use.  The type of
// A and B are compatible with the x and y inputs of z=op(x,y), but can be
// different.  The type of C is the type of z.  C is hypersparse if either A
// or B are hypersparse.

// FUTURE: this would be faster with built-in types and operators.

// FUTURE: at most one thread is used for each vector of C=kron(A,B).  The
// matrix C is normally very large, but if both A and B are n-by-1, then C is
// n^2-by-1 and only a single thread is used.  A better method for this case
// would construct vectors of C in parallel.

// FUTURE: each vector C(:,k) takes O(nnz(C(:,k))) work, but this is not
// accounted for in the parallel load-balancing.

#define GB_FREE_WORKSPACE       \
{                               \
    GB_Matrix_free (&Awork) ;   \
    GB_Matrix_free (&Bwork) ;   \
}

#define GB_FREE_ALL         \
{                           \
    GB_FREE_WORKSPACE ;     \
    GB_phybix_free (C) ;    \
}

#include "GB_kron.h"
#include "GB_emult.h"

GrB_Info GB_kroner                  // C = kron (A,B)
(
    GrB_Matrix C,                   // output matrix
    const bool C_is_csc,            // desired format of C
    const GrB_BinaryOp op,          // multiply operator
    const GrB_Matrix A_in,          // input matrix
    bool A_is_pattern,              // true if values of A are not used
    const GrB_Matrix B_in,          // input matrix
    bool B_is_pattern,              // true if values of B are not used
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    struct GB_Matrix_opaque Awork_header, Bwork_header ;
    GrB_Matrix Awork = NULL, Bwork = NULL ;

    ASSERT_MATRIX_OK (A_in, "A_in for kron (A,B)", GB0) ;
    ASSERT_MATRIX_OK (B_in, "B_in for kron (A,B)", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for kron (A,B)", GB0) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (A_in) ;
    GB_MATRIX_WAIT (B_in) ;

    //--------------------------------------------------------------------------
    // bitmap case: create sparse copies of A and B if they are bitmap
    //--------------------------------------------------------------------------

    GrB_Matrix A = A_in ;
    if (GB_IS_BITMAP (A))
    { 
        GBURBLE ("A:") ;
        // set Awork->iso = A->iso     OK: no need for burble
        GB_CLEAR_STATIC_HEADER (Awork, &Awork_header) ;
        GB_OK (GB_dup_worker (&Awork, A->iso, A, true, NULL)) ;
        ASSERT_MATRIX_OK (Awork, "dup Awork for kron (A,B)", GB0) ;
        GB_OK (GB_convert_bitmap_to_sparse (Awork, Werk)) ;
        ASSERT_MATRIX_OK (Awork, "to sparse, Awork for kron (A,B)", GB0) ;
        A = Awork ;
    }

    GrB_Matrix B = B_in ;
    if (GB_IS_BITMAP (B))
    { 
        GBURBLE ("B:") ;
        // set Bwork->iso = B->iso     OK: no need for burble
        GB_CLEAR_STATIC_HEADER (Bwork, &Bwork_header) ;
        GB_OK (GB_dup_worker (&Bwork, B->iso, B, true, NULL)) ;
        ASSERT_MATRIX_OK (Bwork, "dup Bwork for kron (A,B)", GB0) ;
        GB_OK (GB_convert_bitmap_to_sparse (Bwork, Werk)) ;
        ASSERT_MATRIX_OK (Bwork, "to sparse, Bwork for kron (A,B)", GB0) ;
        B = Bwork ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const GB_void *restrict Ax = A_is_pattern ? NULL : ((GB_void *) A->x) ;
    const int64_t asize = A->type->size ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    int64_t anvec = A->nvec ;
    int64_t anz = GB_nnz (A) ;

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Bi = B->i ;
    const GB_void *restrict Bx = B_is_pattern ? NULL : ((GB_void *) B->x) ;
    const int64_t bsize = B->type->size ;
    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    int64_t bnvec = B->nvec ;
    int64_t bnz = GB_nnz (B) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    double work = ((double) anz) * ((double) bnz)
                + (((double) anvec) * ((double) bnvec)) ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (work, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    GrB_Type ctype = op->ztype ;
    const size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_emult_iso (cscalar, ctype, A, B, op) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    // C has the same type as z for the multiply operator, z=op(x,y)

    GrB_Index cvlen, cvdim, cnzmax, cnvec ;

    bool ok = GB_int64_multiply (&cvlen, avlen, bvlen) ;
    ok = ok & GB_int64_multiply (&cvdim, avdim, bvdim) ;
    ok = ok & GB_int64_multiply (&cnzmax, anz, bnz) ;
    ok = ok & GB_int64_multiply (&cnvec, anvec, bnvec) ;
    ASSERT (ok) ;

    if (C_iso)
    { 
        // the values of A and B are no longer needed if C is iso
        GBURBLE ("(iso kron) ") ;
        A_is_pattern = true ;
        B_is_pattern = true ;
    }

    // C is hypersparse if either A or B are hypersparse.  It is never bitmap.
    bool C_is_hyper = (cvdim > 1) && (Ah != NULL || Bh != NULL) ;
    bool C_is_full = GB_as_if_full (A) && GB_as_if_full (B) ;
    int sparsity = C_is_full ? GxB_FULL :
        ((C_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE) ;

    // set C->iso = C_iso   OK
    GB_OK (GB_new_bix (&C, // full, sparse, or hyper; existing header
        ctype, (int64_t) cvlen, (int64_t) cvdim, GB_Ap_malloc, C_is_csc,
        sparsity, true, B->hyper_switch, cnvec, cnzmax, true, C_iso)) ;

    //--------------------------------------------------------------------------
    // get C and the operator
    //--------------------------------------------------------------------------

    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ch = C->h ;
    int64_t *restrict Ci = C->i ;
    GB_void *restrict Cx = (GB_void *) C->x ;
    int64_t *restrict Cx_int64 = NULL ;
    int32_t *restrict Cx_int32 = NULL ;

    GxB_binary_function fmult = op->binop_function ;
    GB_Opcode opcode = op->opcode ;
    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
    GB_cast_function cast_A = NULL, cast_B = NULL ;
    if (!A_is_pattern)
    { 
        cast_A = GB_cast_factory (op->xtype->code, A->type->code) ;
    }
    if (!B_is_pattern)
    { 
        cast_B = GB_cast_factory (op->ytype->code, B->type->code) ;
    }

    int64_t offset = 0 ;
    bool depends_on_j = false ;
    if (op_is_positional)
    { 
        offset = GB_positional_offset (opcode, NULL, &depends_on_j) ;
        Cx_int64 = (int64_t *) Cx ;
        Cx_int32 = (int32_t *) Cx ;
    }
    bool is64 = (ctype == GrB_INT64) ;

    //--------------------------------------------------------------------------
    // compute the column counts of C, and C->h if C is hypersparse
    //--------------------------------------------------------------------------

    int64_t kC ;

    if (!C_is_full)
    { 
        // C is sparse or hypersparse
        #pragma omp parallel for num_threads(nthreads) schedule(guided)
        for (kC = 0 ; kC < cnvec ; kC++)
        {
            const int64_t kA = kC / bnvec ;
            const int64_t kB = kC % bnvec ;

            // get A(:,jA), the (kA)th vector of A
            const int64_t jA = GBH (Ah, kA) ;
            const int64_t aknz = (Ap == NULL) ? avlen : (Ap [kA+1] - Ap [kA]) ;
            // get B(:,jB), the (kB)th vector of B
            const int64_t jB = GBH (Bh, kB) ;
            const int64_t bknz = (Bp == NULL) ? bvlen : (Bp [kB+1] - Bp [kB]) ;
            // determine # entries in C(:,jC), the (kC)th vector of C
            // int64_t kC = kA * bnvec + kB ;
            Cp [kC] = aknz * bknz ;
            if (C_is_hyper)
            { 
                Ch [kC] = jA * bvdim + jB ;
            }
        }

        GB_cumsum (Cp, cnvec, &(C->nvec_nonempty), nthreads, Werk) ;
        C->nvals = Cp [cnvec] ;
        if (C_is_hyper) C->nvec = cnvec ;
    }

    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // C = kron (A,B) where C is iso and full
    //--------------------------------------------------------------------------

    if (C_iso)
    {
        // Cx [0] = cscalar = op (A,B)
        memcpy (C->x, cscalar, csize) ;
        if (C_is_full)
        { 
            // no more work to do if C is iso and full
            ASSERT_MATRIX_OK (C, "C=kron(A,B), iso full", GB0) ;
            GB_FREE_WORKSPACE ;
            return (GrB_SUCCESS) ;
        }
    }

    //--------------------------------------------------------------------------
    // C = kron (A,B)
    //--------------------------------------------------------------------------

    const bool A_iso = A->iso ;
    const bool B_iso = B->iso ;

    #pragma omp parallel for num_threads(nthreads) schedule(guided)
    for (kC = 0 ; kC < cnvec ; kC++)
    {
        int64_t kA = kC / bnvec ;
        int64_t kB = kC % bnvec ;

        // get B(:,jB), the (kB)th vector of B
        int64_t jB = GBH (Bh, kB) ;
        int64_t pB_start = GBP (Bp, kB, bvlen) ;
        int64_t pB_end   = GBP (Bp, kB+1, bvlen) ;
        int64_t bknz = pB_start - pB_end ;
        if (bknz == 0) continue ;
        GB_void bwork [GB_VLA(bsize)] ;
        if (!B_is_pattern && B_iso)
        { 
            cast_B (bwork, Bx, bsize) ;
        }

        // get C(:,jC), the (kC)th vector of C
        // int64_t kC = kA * bnvec + kB ;
        int64_t pC = GBP (Cp, kC, cvlen) ;

        // get A(:,jA), the (kA)th vector of A
        int64_t jA = GBH (Ah, kA) ;
        int64_t pA_start = GBP (Ap, kA, avlen) ;
        int64_t pA_end   = GBP (Ap, kA+1, avlen) ;
        GB_void awork [GB_VLA(asize)] ;
        if (!A_is_pattern && A_iso)
        { 
            cast_A (awork, Ax, asize) ;
        }

        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
        {
            // awork = A(iA,jA), typecasted to op->xtype
            int64_t iA = GBI (Ai, pA, avlen) ;
            int64_t iAblock = iA * bvlen ;
            if (!A_is_pattern && !A_iso)
            { 
                cast_A (awork, Ax + (pA*asize), asize) ;
            }
            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
            {
                // bwork = B(iB,jB), typecasted to op->ytype
                int64_t iB = GBI (Bi, pB, bvlen) ;
                if (!B_is_pattern && !B_iso)
                { 
                    cast_B (bwork, Bx +(pB*bsize), bsize) ;
                }
                // C(iC,jC) = A(iA,jA) * B(iB,jB)
                if (!C_is_full)
                { 
                    int64_t iC = iAblock + iB ;
                    Ci [pC] = iC ;
                }
                if (op_is_positional)
                {
                    // positional binary operator
                    switch (opcode)
                    {
                        case GB_FIRSTI_binop_code   : 
                            // z = first_i(A(iA,jA),y) == iA
                        case GB_FIRSTI1_binop_code  : 
                            // z = first_i1(A(iA,jA),y) == iA+1
                            if (is64)
                            { 
                                Cx_int64 [pC] = iA + offset ;
                            }
                            else
                            { 
                                Cx_int32 [pC] = (int32_t) (iA + offset) ;
                            }
                            break ;
                        case GB_FIRSTJ_binop_code   : 
                            // z = first_j(A(iA,jA),y) == jA
                        case GB_FIRSTJ1_binop_code  : 
                            // z = first_j1(A(iA,jA),y) == jA+1
                            if (is64)
                            { 
                                Cx_int64 [pC] = jA + offset ;
                            }
                            else
                            { 
                                Cx_int32 [pC] = (int32_t) (jA + offset) ;
                            }
                            break ;
                        case GB_SECONDI_binop_code  : 
                            // z = second_i(x,B(iB,jB)) == iB
                        case GB_SECONDI1_binop_code : 
                            // z = second_i1(x,B(iB,jB)) == iB+1
                            if (is64)
                            { 
                                Cx_int64 [pC] = iB + offset ;
                            }
                            else
                            { 
                                Cx_int32 [pC] = (int32_t) (iB + offset) ;
                            }
                            break ;
                        case GB_SECONDJ_binop_code  : 
                            // z = second_j(x,B(iB,jB)) == jB
                        case GB_SECONDJ1_binop_code : 
                            // z = second_j1(x,B(iB,jB)) == jB+1
                            if (is64)
                            { 
                                Cx_int64 [pC] = jB + offset ;
                            }
                            else
                            { 
                                Cx_int32 [pC] = (int32_t) (jB + offset) ;
                            }
                            break ;
                        default: ;
                    }
                }
                else if (!C_iso)
                { 
                    // standard binary operator
                    fmult (Cx +(pC*csize), awork, bwork) ;
                }
                pC++ ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    GB_OK (GB_hypermatrix_prune (C, Werk)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C=kron(A,B)", GB0) ;
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

