//------------------------------------------------------------------------------
// GB_kroner: Kronecker product, C = kron (A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = kron(A,B) where op determines the binary multiplier to use.  The type of
// C is the ztype of the operator.  C is hypersparse if either A or B are
// hypersparse, full if both A and B are full, or sparse otherwise.  C is never
// constructed as bitmap.

#define GB_FREE_WORKSPACE       \
{                               \
    GB_Matrix_free (&Awork) ;   \
    GB_Matrix_free (&Bwork) ;   \
}

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORKSPACE ;         \
    GB_phybix_free (C) ;        \
}

#include "kronecker/GB_kron.h"
#include "emult/GB_emult.h"
#include "slice/include/GB_search_for_vector.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GB_kroner                  // C = kron (A,B)
(
    GrB_Matrix C,                   // output matrix
    const bool C_is_csc,            // desired format of C
    const GrB_BinaryOp op,          // multiply operator
    const bool flipij,              // if true, i and j are flipped: z=(x,y,j,i)
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
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    const int64_t anvec = A->nvec ;
    const int64_t anz = GB_nnz (A) ;

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Bi = B->i ;
    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    const int64_t bnvec = B->nvec ;
    const int64_t bnz = GB_nnz (B) ;

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
    int C_sparsity = C_is_full ? GxB_FULL :
        ((C_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE) ;

    // set C->iso = C_iso   OK
    GB_OK (GB_new_bix (&C, // full, sparse, or hyper; existing header
        ctype, (int64_t) cvlen, (int64_t) cvdim, GB_Ap_malloc, C_is_csc,
        C_sparsity, true, B->hyper_switch, cnvec, cnzmax, true, C_iso)) ;

    //--------------------------------------------------------------------------
    // compute the column counts of C: Cp and Ch if C is hypersparse
    //--------------------------------------------------------------------------

    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ch = C->h ;

    if (!C_is_full)
    { 
        // C is sparse or hypersparse
        int64_t kC ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
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
    // C = kron (A,B) where C is iso and/or full full
    //--------------------------------------------------------------------------

    if (C_iso)
    { 
        // C->x [0] = cscalar = op (A,B)
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
    // quick return if C is empty
    //--------------------------------------------------------------------------

    int64_t cnz = GB_nnz (C) ;
    if (cnz == 0)
    { 
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // C = kron (A,B)
    //--------------------------------------------------------------------------

    // via the JIT kernel
    info = GB_kroner_jit (C, op, flipij, A, B, nthreads) ;

    if (info == GrB_NO_VALUE)
    { 
        // via the generic kernel
        #define GB_A_TYPE GB_void
        #define GB_B_TYPE GB_void
        #define GB_C_TYPE GB_void
        #define GB_A_ISO A_iso
        #define GB_B_ISO B_iso
        #define GB_C_ISO C_iso
        const bool A_iso = A->iso ;
        const bool B_iso = B->iso ;
        const int64_t asize = A->type->size ;
        const int64_t bsize = B->type->size ;

        GxB_binary_function fmult = op->binop_function ;
        GxB_index_binary_function fmult_idx = op->idxbinop_function ;
        const void *theta = op->theta ;
        GB_cast_function cast_A = NULL, cast_B = NULL ;
        if (!A_is_pattern)
        { 
            cast_A = GB_cast_factory (op->xtype->code, A->type->code) ;
        }
        if (!B_is_pattern)
        { 
            cast_B = GB_cast_factory (op->ytype->code, B->type->code) ;
        }

        #define GB_C_IS_FULL C_is_full

        #define GB_DECLAREA(a) GB_void a [GB_VLA(asize)]
        #define GB_DECLAREB(b) GB_void b [GB_VLA(bsize)]

        #define GB_GETA(a,Ax,p,iso)                         \
        {                                                   \
            if (!A_is_pattern)                              \
            {                                               \
                cast_A (a, Ax + (p)*asize, asize) ;         \
            }                                               \
        }

        #define GB_GETB(b,Bx,p,iso)                         \
        {                                                   \
            if (!B_is_pattern)                              \
            {                                               \
                cast_B (b, Bx + (p)*bsize, bsize) ;         \
            }                                               \
        }

        #define GB_KRONECKER_OP(Cx,pC,a,ix,jx,b,iy,jy)      \
        {                                                   \
            if (fmult != NULL)                              \
            {                                               \
                /* standard binary operator */              \
                fmult (Cx +(pC)*csize, a, b) ;              \
            }                                               \
            else                                            \
            {                                               \
                /* index binary operator */                 \
                if (flipij)                                 \
                {                                           \
                    fmult_idx (Cx +(pC)*csize,              \
                        a, jx, ix, b, jy, iy, theta) ;      \
                }                                           \
                else                                        \
                {                                           \
                    fmult_idx (Cx +(pC)*csize,              \
                        a, ix, jx, b, iy, jy, theta) ;      \
                }                                           \
            }                                               \
        }

        #define GB_GENERIC
        #include "ewise/include/GB_ewise_shared_definitions.h"
        #include "kronecker/template/GB_kroner_template.c"
        info = GrB_SUCCESS ;
    }

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    { 
        GB_OK (GB_hypermatrix_prune (C, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C=kron(A,B)", GB0) ;
    GB_FREE_WORKSPACE ;
    return (info) ;
}

