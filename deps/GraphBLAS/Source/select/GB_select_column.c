//------------------------------------------------------------------------------
// GB_select_column: apply a select COL* operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The column selectors can be done in a single pass.
// C->iso and A->iso are identical.

#include "select/GB_select.h"
#include "transpose/GB_transpose.h"
#include "jitifyer/GB_stringify.h"

#define GB_FREE_ALL                         \
{                                           \
    GB_phybix_free (C) ;                    \
}

GrB_Info GB_select_column
(
    GrB_Matrix C,
    const GrB_IndexUnaryOp op,
    GrB_Matrix A,
    int64_t ithunk,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for GB_select_column", GB0) ;
    ASSERT_MATRIX_OK (A, "A for select column", GB0_Z) ;
    GB_Opcode opcode = op->opcode ;
    ASSERT (opcode == GB_COLINDEX_idxunop_code ||
            opcode == GB_COLLE_idxunop_code ||
            opcode == GB_COLGT_idxunop_code) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (C != NULL && (C->header_size == 0 || GBNSTATIC)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;

    const GB_void *restrict Ax = (GB_void *) A->x ;
    int64_t anvec = A->nvec ;
    bool A_jumbled = A->jumbled ;
    bool A_is_hyper = (Ah != NULL) ;
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;
    const bool A_iso = A->iso ;
    const size_t asize = A->type->size ;

    GB_Type_code Ap_code = A->p_is_32 ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code Ah_code = A->j_is_32 ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code Ai_code = A->i_is_32 ? GB_INT32_code  : GB_INT64_code ;

    //--------------------------------------------------------------------------
    // determine number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nth = nthreads_max ;

    //--------------------------------------------------------------------------
    // find column j in A
    //--------------------------------------------------------------------------

    int64_t j = (opcode == GB_COLINDEX_idxunop_code) ? (-ithunk) : ithunk ;

    int64_t k = 0 ;
    bool found ;
    if (j < 0)
    { 
        // j is outside the range of columns of A
        k = 0 ;
        found = false ;
    }
    else if (j >= avdim)
    { 
        // j is outside the range of columns of A
        k = anvec ;
        found = false ;
    }
    else if (A_is_hyper)
    { 
        // find the column j in the hyperlist of A
        // future:: use hyper_hash if present
        int64_t kright = anvec-1 ;
        found = GB_split_binary_search (j, Ah, A->j_is_32, &k, &kright) ;
        // if found is true the Ah [k] == j
        // if found is false, then Ah [0..k-1] < j and Ah [k..anvec-1] > j
    }
    else
    { 
        // j appears as the jth column in A; found is always true
        k = j ;
        found = true ;
    }

    //--------------------------------------------------------------------------
    // determine the # of entries and # of vectors in C
    //--------------------------------------------------------------------------

    int64_t pstart = GB_IGET (Ap, k) ;
    int64_t pend = found ? GB_IGET (Ap, k+1) : pstart ;
    int64_t ajnz = pend - pstart ;
    int64_t cnz, cnvec ;
    int64_t anz = A->nvals ;
    ASSERT (A->nvals == GB_IGET (Ap, anvec)) ;

    if (opcode == GB_COLINDEX_idxunop_code)
    { 
        // COLINDEX: delete column j:  C = A (:, [0:j-1 j+1:end])
        cnz = anz - ajnz ;
        cnvec = (A_is_hyper && found) ? (anvec-1) : anvec ;
    }
    else if (opcode == GB_COLLE_idxunop_code)
    { 
        // COLLE: C = A (:, 0:j)
        cnz = pend ;
        cnvec = (A_is_hyper) ? (found ? (k+1) : k) : anvec ;
    }
    else // (opcode == GB_COLGT_idxunop_code)
    { 
        // COLGT: C = A (:, j+1:end)
        cnz = anz - pend ;
        cnvec = anvec - ((A_is_hyper) ? (found ? (k+1) : k) : 0) ;
    }

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new matrix
    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_AUTO_SPARSITY, cnz, avlen, avdim, Werk) ;

    if (cnz == anz)
    { 
        // C is the same as A: return it a pure shallow copy
        return (GB_shallow_copy (C, true, A, Werk)) ;
    }
    else if (cnz == 0)
    { 
        // return C as empty
        return (GB_new (&C, // auto (sparse or hyper), existing header
            A->type, avlen, avdim, GB_ph_calloc, true,
            GxB_AUTO_SPARSITY, GB_Global_hyper_switch_get ( ), 1,
            Cp_is_32, Cj_is_32, Ci_is_32)) ;
    }

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    int csparsity = (A_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE ;
    GB_OK (GB_new_bix (&C, // sparse or hyper (from A), existing header
        A->type, avlen, avdim, GB_ph_malloc, true, csparsity, false,
        A->hyper_switch, cnvec, cnz, true, A_iso,
        Cp_is_32, Cj_is_32, Ci_is_32)) ;

    ASSERT (Cp_is_32 == C->p_is_32) ;
    ASSERT (Cj_is_32 == C->j_is_32) ;
    ASSERT (Ci_is_32 == C->i_is_32) ;

    Cp_is_32 = C->p_is_32 ;
    Cj_is_32 = C->j_is_32 ;
    Ci_is_32 = C->i_is_32 ;

    ASSERT (info == GrB_SUCCESS) ;
    int nth2 = GB_nthreads (cnvec, chunk, nth) ;

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, ) ; GB_Ch_PTR (Ch, C) ;
    GB_Ci_DECLARE (Ci, ) ; GB_Ci_PTR (Ci, C) ;

    GB_Type_code Cp_code = Cp_is_32 ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code Ch_code = Cj_is_32 ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code Ci_code = Ci_is_32 ? GB_INT32_code  : GB_INT64_code ;
    size_t cpsize = Cp_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    GB_void *restrict Cx = (GB_void *) C->x ;
    int64_t kk ;

    //--------------------------------------------------------------------------
    // construct C
    //--------------------------------------------------------------------------

    if (A_iso)
    { 
        // Cx [0] = Ax [0]
        memcpy (Cx, Ax, asize) ;
    }

    if (opcode == GB_COLINDEX_idxunop_code)
    {

        //----------------------------------------------------------------------
        // COLINDEX: delete the column j
        //----------------------------------------------------------------------

        if (A_is_hyper)
        { 
            ASSERT (found) ;

            // Cp [0:k-1] = Ap [0:k-1]
            GB_cast_int (Cp, Cp_code, Ap, Ap_code, k, nth) ;

            // Cp [k:cnvec] = Ap [k+1:anvec] - ajnz
            #pragma omp parallel for num_threads(nth2)
            for (kk = k ; kk <= cnvec ; kk++)
            { 
                // Cp [kk] = Ap [kk+1] - ajnz ;
                int64_t p = GB_IGET (Ap, kk+1) - ajnz ;
                GB_ISET (Cp, kk, p) ;
            }

            // Ch [0:k-1] = Ah [0:k-1]
            GB_cast_int (Ch, Ch_code, Ah, Ah_code, k, nth) ;

            // Ch [k:cnvec-1] = Ah [k+1:anvec-1]
            GB_cast_int (GB_IADDR (Ch, k  ), Ch_code,
                         GB_IADDR (Ah, k+1), Ah_code, cnvec - k, nth) ;

        }
        else
        { 

            // Cp [0:k] = Ap [0:k]
            GB_cast_int (Cp, Cp_code, Ap, Ap_code, k+1, nth) ;

            // Cp [k+1:anvec] = Ap [k+1:anvec] - ajnz
            #pragma omp parallel for num_threads(nth2)
            for (kk = k+1 ; kk <= cnvec ; kk++)
            { 
                // Cp [kk] = Ap [kk] - ajnz ;
                int64_t p = GB_IGET (Ap, kk) - ajnz ;
                GB_ISET (Cp, kk, p) ;
            }
        }

        // Ci [0:pstart-1] = Ai [0:pstart-1]
        GB_cast_int (Ci, Ci_code, Ai, Ai_code, pstart, nth) ;

        // Ci [pstart:cnz-1] = Ai [pend:anz-1]
        GB_cast_int (GB_IADDR (Ci, pstart), Ci_code,
                     GB_IADDR (Ai, pend  ), Ai_code, cnz - pstart, nth) ;

        if (!A_iso)
        { 
            // Cx [0:pstart-1] = Ax [0:pstart-1]
            GB_memcpy (Cx, Ax, pstart * asize, nth) ;

            // Cx [pstart:cnz-1] = Ax [pend:anz-1]
            GB_memcpy (Cx + pstart * asize, Ax + pend * asize,
                (cnz - pstart) * asize, nth) ;
        }

    }
    else if (opcode == GB_COLLE_idxunop_code)
    {

        //----------------------------------------------------------------------
        // COLLE: C = A (:, 0:j)
        //----------------------------------------------------------------------

        if (A_is_hyper)
        { 
            // Cp [0:cnvec] = Ap [0:cnvec]
            GB_cast_int (Cp, Cp_code, Ap, Ap_code, cnvec+1, nth) ;

            // Ch [0:cnvec-1] = Ah [0:cnvec-1]
            GB_cast_int (Ch, Ch_code, Ah, Ah_code, cnvec, nth) ;
        }
        else
        {
            // Cp [0:k+1] = Ap [0:k+1]
            ASSERT (found) ;
            GB_cast_int (Cp, Cp_code, Ap, Ap_code, k+2, nth) ;

            // Cp [k+2:cnvec] = cnz
            #pragma omp parallel for num_threads(nth2)
            for (kk = k+2 ; kk <= cnvec ; kk++)
            { 
                // Cp [kk] = cnz ;
                GB_ISET (Cp, kk, cnz) ;
            }
        }

        // Ci [0:cnz-1] = Ai [0:cnz-1]
        GB_cast_int (Ci, Ci_code, Ai, Ai_code, cnz, nth) ;

        if (!A_iso)
        { 
            // Cx [0:cnz-1] = Ax [0:cnz-1]
            GB_memcpy (Cx, Ax, cnz * asize, nth) ;
        }

    }
    else // (opcode == GB_COLGT_idxunop_code)
    {

        //----------------------------------------------------------------------
        // COLGT: C = A (:, j+1:end)
        //----------------------------------------------------------------------

        if (A_is_hyper)
        { 
            // Cp [0:cnvec] = Ap [k+found:anvec] - pend
            #pragma omp parallel for num_threads(nth2)
            for (kk = 0 ; kk <= cnvec ; kk++)
            { 
                // Cp [kk] = Ap [kk + k + found] - pend ;
                int64_t p = GB_IGET (Ap, kk + k + found) - pend ;
                GB_ISET (Cp, kk, p) ;
            }

            // Ch [0:cnvec-1] = Ah [k+found:anvec-1]
            GB_cast_int (Ch, Ch_code, GB_IADDR (Ah, k+found), Ah_code,
                cnvec, nth) ;

        }
        else
        {

            ASSERT (found) ;

            // Cp [0:k] = 0
            GB_memset (Cp, 0, (k+1) * cpsize, nth) ;

            // Cp [k+1:cnvec] = Ap [k+1:cnvec] - pend
            #pragma omp parallel for num_threads(nth2)
            for (kk = k+1 ; kk <= cnvec ; kk++)
            { 
                // Cp [kk] = Ap [kk] - pend ;
                int64_t p = GB_IGET (Ap, kk) - pend ;
                GB_ISET (Cp, kk, p) ;
            }
        }

        // Ci [0:cnz-1] = Ai [pend:anz-1]
        GB_cast_int (Ci, Ci_code, GB_IADDR (Ai, pend), Ai_code, cnz, nth) ;

        if (!A_iso)
        { 
            // Cx [0:cnz-1] = Ax [pend:anz-1]
            GB_memcpy (Cx, Ax + pend * asize, cnz * asize, nth) ;
        }
    }

    //--------------------------------------------------------------------------
    // finalize the matrix, free workspace, and return result
    //--------------------------------------------------------------------------

    C->nvec = cnvec ;
    C->magic = GB_MAGIC ;
    C->jumbled = A_jumbled ;    // C is jumbled if A is jumbled
    C->nvals = GB_IGET (Cp, cnvec) ;
    GB_nvec_nonempty_set (C, GB_nvec_nonempty (C)) ;
    ASSERT_MATRIX_OK (C, "C output for GB_select_column", GB0) ;
    return (GrB_SUCCESS) ;
}

