//------------------------------------------------------------------------------
// GB_emult_02: C = A.*B where A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

// C = A.*B where A is sparse/hyper and B is bitmap/full constructs C with
// the same sparsity structure as A.

// When no mask is present, or the mask is applied later, this method handles
// the following cases:

        //      ------------------------------------------
        //      C       =           A       .*      B
        //      ------------------------------------------
        //      sparse  .           sparse          bitmap
        //      sparse  .           sparse          full  

// If M is sparse/hyper and complemented, it is not passed here:

        //      ------------------------------------------
        //      C       <!M>=       A       .*      B
        //      ------------------------------------------
        //      sparse  sparse      sparse          bitmap  (mask later)
        //      sparse  sparse      sparse          full    (mask later)

// If M is present, it is bitmap/full:

        //      ------------------------------------------
        //      C      <M> =        A       .*      B
        //      ------------------------------------------
        //      sparse  bitmap      sparse          bitmap
        //      sparse  bitmap      sparse          full  

        //      ------------------------------------------
        //      C      <M> =        A       .*      B
        //      ------------------------------------------
        //      sparse  full        sparse          bitmap
        //      sparse  full        sparse          full  

        //      ------------------------------------------
        //      C      <!M> =        A       .*      B
        //      ------------------------------------------
        //      sparse  bitmap      sparse          bitmap
        //      sparse  bitmap      sparse          full  

        //      ------------------------------------------
        //      C      <!M> =        A       .*      B
        //      ------------------------------------------
        //      sparse  full        sparse          bitmap
        //      sparse  full        sparse          full  

#include "GB_ewise.h"
#include "GB_emult.h"
#include "GB_binop.h"
#include "GB_unused.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_ew__include.h"
#endif

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_WERK_POP (Work, int64_t) ;           \
    GB_WERK_POP (A_ek_slicing, int64_t) ;   \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORKSPACE ;                     \
    GB_phybix_free (C) ;                    \
}

GrB_Info GB_emult_02        // C=A.*B when A is sparse/hyper, B bitmap/full
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask, unused if NULL
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use !M
    const GrB_Matrix A,     // input A matrix (sparse/hyper)
    const GrB_Matrix B,     // input B matrix (bitmap/full)
    GrB_BinaryOp op,        // op to perform C = op (A,B)
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    ASSERT_MATRIX_OK_OR_NULL (M, "M for emult_02", GB0) ;
    ASSERT_MATRIX_OK (A, "A for emult_02", GB0) ;
    ASSERT_MATRIX_OK (B, "B for emult_02", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for emult_02", GB0) ;
    ASSERT_TYPE_OK (ctype, "ctype for emult_02", GB0) ;

    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B)) ;
    ASSERT (M == NULL || GB_IS_BITMAP (M) || GB_IS_FULL (M)) ;

    int C_sparsity = GB_sparsity (A) ;

    if (M == NULL)
    { 
        GBURBLE ("emult_02:(%s=%s.*%s)",
            GB_sparsity_char (C_sparsity),
            GB_sparsity_char_matrix (A),
            GB_sparsity_char_matrix (B)) ;
    }
    else
    { 
        GBURBLE ("emult_02:(%s<%s%s%s>=%s.*%s) ",
            GB_sparsity_char (C_sparsity),
            Mask_comp ? "!" : "",
            GB_sparsity_char_matrix (M),
            Mask_struct ? ",struct" : "",
            GB_sparsity_char_matrix (A),
            GB_sparsity_char_matrix (B)) ;
    }

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (Work, int64_t) ;
    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;

    //--------------------------------------------------------------------------
    // get M, A, and B
    //--------------------------------------------------------------------------

    const int8_t  *restrict Mb = (M == NULL) ? NULL : M->b ;
    const GB_M_TYPE *restrict Mx = (M == NULL || Mask_struct) ? NULL :
        (const GB_M_TYPE *) M->x ;
    const size_t msize = (M == NULL) ? 0 : M->type->size ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const int64_t vlen = A->vlen ;
    const int64_t vdim = A->vdim ;
    const int64_t nvec = A->nvec ;
    const int64_t anz = GB_nnz (A) ;

    const int8_t *restrict Bb = B->b ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    const size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_emult_iso (cscalar, ctype, A, B, op) ;

    //--------------------------------------------------------------------------
    // allocate C->p and C->h
    //--------------------------------------------------------------------------

    GB_OK (GB_new (&C, // sparse or hyper (same as A), existing header
        ctype, vlen, vdim, GB_Ap_calloc, C_is_csc,
        C_sparsity, A->hyper_switch, nvec)) ;
    int64_t *restrict Cp = C->p ;

    //--------------------------------------------------------------------------
    // slice the input matrix A
    //--------------------------------------------------------------------------

    int A_nthreads, A_ntasks ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    GB_SLICE_MATRIX (A, 8) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    GB_WERK_PUSH (Work, 3*A_ntasks, int64_t) ;
    if (Work == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    int64_t *restrict Wfirst    = Work ;
    int64_t *restrict Wlast     = Work + A_ntasks ;
    int64_t *restrict Cp_kfirst = Work + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // phase1: count entries in C and allocate C->i and C->x
    //--------------------------------------------------------------------------

    GB_OK (GB_emult_02_phase1 (C, C_iso, M, Mask_struct, Mask_comp, A, B,
        A_ek_slicing, A_ntasks, A_nthreads, Wfirst, Wlast, Cp_kfirst, Werk)) ;

    //--------------------------------------------------------------------------
    // get the opcode for phase2
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
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
    bool A_is_pattern = op_is_second || op_is_pair || op_is_positional ;
    bool B_is_pattern = op_is_first  || op_is_pair || op_is_positional ;

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
        #include "GB_emult_02_template.c"
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

            #define GB_AemultB_02(mult,xname) GB (_AemultB_02_ ## mult ## xname)

            #define GB_BINOP_WORKER(mult,xname)                         \
            {                                                           \
                info = GB_AemultB_02(mult,xname) (C,                    \
                    M, Mask_struct, Mask_comp, A, B,                    \
                    Cp_kfirst, A_ek_slicing, A_ntasks, A_nthreads) ;    \
            }                                                           \
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
                #include "GB_binop_factory.c"
            }
        }
        #endif
    }

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        info = GB_emult_02_jit (C, C_sparsity, M, Mask_struct,
            Mask_comp, op, A, B, Cp_kfirst, A_ek_slicing, A_ntasks,
            A_nthreads) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        GB_BURBLE_MATRIX (C, "(generic emult_02: %s) ", op->name) ;
        info = GB_emult_generic (C, op, NULL, 0, 0,
            NULL, NULL, NULL, C_sparsity, GB_EMULT_METHOD2, Cp_kfirst,
            NULL, 0, 0, A_ek_slicing, A_ntasks, A_nthreads, NULL, 0, 0,
            M, Mask_struct, Mask_comp, A, B) ;
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

    GB_OK (GB_hypermatrix_prune (C, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (C, "C output for emult_02", GB0) ;
    return (GrB_SUCCESS) ;
}

