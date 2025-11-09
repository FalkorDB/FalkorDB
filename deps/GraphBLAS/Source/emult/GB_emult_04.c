//------------------------------------------------------------------------------
// GB_emult_04: C<M>= A.*B, M sparse/hyper, A and B bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M>= A.*B, M sparse/hyper, A and B bitmap/full.  C has the same sparsity
// structure as M, and its pattern is a subset of M.  M is not complemented.

            //      ------------------------------------------
            //      C       <M>=        A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      bitmap          bitmap  (method: 04)
            //      sparse  sparse      bitmap          full    (method: 04)
            //      sparse  sparse      full            bitmap  (method: 04)
            //      sparse  sparse      full            full    (method: 04)

// TODO: this function can also do eWiseAdd, just as easily.
// Just change the "&&" to "||" in the GB_emult_04_template. 
// If A and B are both full, eadd and emult are identical.

#include "ewise/GB_ewise.h"
#include "emult/GB_emult.h"
#include "binaryop/GB_binop.h"
#include "jitifyer/GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "FactoryKernels/GB_ew__include.h"
#endif
#include "slice/factory/GB_ek_slice_merge.h"

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_WERK_POP (Work, uint64_t) ;          \
    GB_WERK_POP (M_ek_slicing, int64_t) ;   \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORKSPACE ;                     \
    GB_phybix_free (C) ;                    \
}

GrB_Info GB_emult_04        // C<M>=A.*B, M sparse/hyper, A and B bitmap/full
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // sparse/hyper, not NULL
    const bool Mask_struct, // if true, use the only structure of M
    bool *mask_applied,     // if true, the mask was applied
    const GrB_Matrix A,     // input A matrix (bitmap/full)
    const GrB_Matrix B,     // input B matrix (bitmap/full)
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    const bool flipij,      // if true, i,j must be flipped
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->header_size == 0 || GBNSTATIC)) ;

    ASSERT_MATRIX_OK (M, "M for emult_04", GB0) ;
    ASSERT_MATRIX_OK (A, "A for emult_04", GB0) ;
    ASSERT_MATRIX_OK (B, "B for emult_04", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for emult_04", GB0) ;

    ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
    ASSERT (!GB_PENDING (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B)) ;

    int C_sparsity = GB_sparsity (M) ;

    GBURBLE ("emult_04:(%s<%s>=%s.*%s) ",
        GB_sparsity_char (C_sparsity),
        GB_sparsity_char_matrix (M),
        GB_sparsity_char_matrix (A),
        GB_sparsity_char_matrix (B)) ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (Work, uint64_t) ;
    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;

    //--------------------------------------------------------------------------
    // get M, A, and B
    //--------------------------------------------------------------------------

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mh_DECLARE (Mh, const) ; GB_Mh_PTR (Mh, M) ;
    GB_Mi_DECLARE (Mi, const) ; GB_Mi_PTR (Mi, M) ;

    const GB_M_TYPE *restrict Mx = (Mask_struct) ? NULL : (GB_M_TYPE *) M->x ;
    const int64_t vlen = M->vlen ;
    const int64_t vdim = M->vdim ;
    const int64_t nvec = M->nvec ;
    const int64_t mnz = GB_nnz (M) ;
    const size_t  msize = M->type->size ;

    const int8_t *restrict Ab = A->b ;
    const int8_t *restrict Bb = B->b ;

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    const size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_emult_iso (cscalar, ctype, A, B, op) ;

    //--------------------------------------------------------------------------
    // allocate C->p and C->h
    //--------------------------------------------------------------------------

    GB_OK (GB_new (&C, // sparse or hyper (same as M), existing header
        ctype, vlen, vdim, GB_ph_calloc, C_is_csc,
        C_sparsity, M->hyper_switch, nvec,
        M->p_is_32, M->j_is_32, M->i_is_32)) ;

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ci_DECLARE (Ci, ) ; GB_Ci_PTR (Ci, C) ;
    bool Cp_is_32 = C->p_is_32 ;
    bool Cj_is_32 = C->j_is_32 ;
    ASSERT (Cp_is_32 == M->p_is_32) ;
    ASSERT (Cj_is_32 == M->j_is_32) ;
    ASSERT (C->i_is_32 == M->i_is_32) ;

    //--------------------------------------------------------------------------
    // slice the mask matrix M
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int M_ntasks, M_nthreads ;
    GB_SLICE_MATRIX (M, 8) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    GB_WERK_PUSH (Work, 3*M_ntasks, uint64_t) ;
    if (Work == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    uint64_t *restrict Wfirst    = Work ;
    uint64_t *restrict Wlast     = Work + M_ntasks ;
    uint64_t *restrict Cp_kfirst = Work + M_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // count entries in C
    //--------------------------------------------------------------------------

    // This phase is very similar to GB_select_entry_phase1_template.c.

    // TODO: if M is structural and A and B are both full, then C has exactly
    // the same pattern as M, the first phase can be skipped.

    int tid ;
    #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < M_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Mslice [tid] ;
        int64_t klast  = klast_Mslice  [tid] ;
        Wfirst [tid] = 0 ;
        Wlast  [tid] = 0 ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // count the entries in C(:,j)
            int64_t j = GBh_M (Mh, k) ;
            int64_t pstart = j * vlen ;     // start of A(:,j) and B(:,j)
            GB_GET_PA (pM, pM_end, tid, k, kfirst, klast, pstart_Mslice,
                GB_IGET (Mp, k), GB_IGET (Mp, k+1)) ;
            int64_t cjnz = 0 ;
            for ( ; pM < pM_end ; pM++)
            { 
                bool mij = GB_MCAST (Mx, pM, msize) ;
                if (mij)
                {
                    int64_t i = GB_IGET (Mi, pM) ;
                    cjnz += (GBb_A (Ab, pstart + i) && GBb_B (Bb, pstart + i)) ;
                }
            }
            if (k == kfirst)
            { 
                Wfirst [tid] = cjnz ;
            }
            else if (k == klast)
            { 
                Wlast [tid] = cjnz ;
            }
            else
            { 
                GB_ISET (Cp, k, cjnz) ;     // Cp [k] = cjnz ; 
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize Cp, cumulative sum of Cp and compute Cp_kfirst
    //--------------------------------------------------------------------------

    GB_ek_slice_merge1 (Cp, Cp_is_32, Wfirst, Wlast, M_ek_slicing, M_ntasks) ;
    int64_t nvec_nonempty ;
    GB_cumsum (Cp, Cp_is_32, nvec, &nvec_nonempty, M_nthreads, Werk) ;
    GB_nvec_nonempty_set (C, nvec_nonempty) ;
    GB_ek_slice_merge2 (Cp_kfirst, Cp, Cp_is_32,
        Wfirst, Wlast, M_ek_slicing, M_ntasks) ;

    //--------------------------------------------------------------------------
    // allocate C->i and C->x
    //--------------------------------------------------------------------------

    int64_t cnz = GB_IGET (Cp, nvec) ;
    GB_OK (GB_bix_alloc (C, cnz, GxB_SPARSE, false, true, C_iso)) ;

    //--------------------------------------------------------------------------
    // copy pattern into C
    //--------------------------------------------------------------------------

    // FUTURE: could make this components of C shallow instead

    size_t cjsize = Cj_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    if (GB_IS_HYPERSPARSE (M))
    { 
        // copy M->h into C->h
        GB_memcpy (C->h, Mh, nvec * cjsize, M_nthreads) ;
    }

    C->nvec = nvec ;
    C->jumbled = M->jumbled ;
    C->nvals = cnz ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // get the opcode
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    bool op_is_builtin_positional =
        GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (opcode) ;
    bool op_is_index_binop = GB_IS_INDEXBINARYOP_CODE (opcode) ;
    bool op_is_positional = op_is_builtin_positional || op_is_index_binop ;
    bool op_is_first  = (opcode == GB_FIRST_binop_code) ;
    bool op_is_second = (opcode == GB_SECOND_binop_code) ;
    bool op_is_pair   = (opcode == GB_PAIR_binop_code) ;
    GB_Type_code ccode = ctype->code ;

    //--------------------------------------------------------------------------
    // check if the values of A and/or B are ignored
    //--------------------------------------------------------------------------

    // With C = ewisemult (A,B), only the intersection of A and B is used.
    // If op is SECOND or PAIR, the values of A are never accessed.
    // If op is FIRST  or PAIR, the values of B are never accessed.
    // If op is PAIR, the values of A and B are never accessed.
    // Contrast with ewiseadd.

    // A is passed as x, and B as y, in z = op(x,y)
    bool A_is_pattern = op_is_second || op_is_pair || op_is_builtin_positional ;
    bool B_is_pattern = op_is_first  || op_is_pair || op_is_builtin_positional ;

    //--------------------------------------------------------------------------
    // using a built-in binary operator (except for positional operators)
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;

    if (C_iso)
    { 

        //----------------------------------------------------------------------
        // via the iso kernel
        //----------------------------------------------------------------------

        // Cx [0] = cscalar = op (A,B)
        GB_BURBLE_MATRIX (C, "(iso emult) ") ;
        memcpy (C->x, cscalar, csize) ;

        // pattern of C = set intersection of pattern of A and B
        #define GB_ISO_EMULT
        #include "emult/template/GB_emult_04_template.c"
        info = GrB_SUCCESS ;

    }
    else
    {

        //----------------------------------------------------------------------
        // via the factory kernel
        //----------------------------------------------------------------------

        #ifndef GBCOMPACT
        GB_IF_FACTORY_KERNELS_ENABLED
        { 

            //------------------------------------------------------------------
            // define the worker for the switch factory
            //------------------------------------------------------------------

            #define GB_AemultB_04(mult,xname) GB (_AemultB_04_ ## mult ## xname)

            #define GB_BINOP_WORKER(mult,xname)                             \
            {                                                               \
                info = GB_AemultB_04(mult,xname) (C, M, Mask_struct, A, B,  \
                    Cp_kfirst, M_ek_slicing, M_ntasks, M_nthreads) ;        \
            }                                                               \
            break ;

            //------------------------------------------------------------------
            // launch the switch factory
            //------------------------------------------------------------------

            GB_Type_code xcode, ycode, zcode ;
            if (!op_is_positional &&
                GB_binop_builtin (A->type, A_is_pattern, B->type, B_is_pattern,
                op, false, &opcode, &xcode, &ycode, &zcode) && ccode == zcode)
            { 
                #define GB_NO_PAIR
                #include "binaryop/factory/GB_binop_factory.c"
            }
        }
        #endif
    }

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        info = GB_emult_04_jit (C, C_sparsity, M, Mask_struct, op, flipij,
            A, B, Cp_kfirst, M_ek_slicing, M_ntasks, M_nthreads) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        GB_BURBLE_MATRIX (C, "(generic emult_04: %s) ", op->name) ;
        info = GB_emult_generic (C, op, flipij, NULL, 0, 0,
            NULL, NULL, NULL, C_sparsity, GB_EMULT_METHOD4, Cp_kfirst,
            M_ek_slicing, M_ntasks, M_nthreads, NULL, 0, 0, NULL, 0, 0,
            M, Mask_struct, false, A, B) ;
    }

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    if (info != GrB_SUCCESS)
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    GB_OK (GB_hyper_prune (C, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (C, "C output for emult_04", GB0) ;
    (*mask_applied) = true ;
    return (GrB_SUCCESS) ;
}

