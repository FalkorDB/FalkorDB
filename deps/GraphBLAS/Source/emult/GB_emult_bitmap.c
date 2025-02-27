//------------------------------------------------------------------------------
// GB_emult_bitmap: C = A.*B, C<M>=A.*B, or C<!M>=A.*B when C is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_EMULT_METHOD5 : 

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      bitmap  .           bitmap          bitmap  (method: 5)
            //      bitmap  .           bitmap          full    (method: 5)
            //      bitmap  .           full            bitmap  (method: 5)

// GB_EMULT_METHOD6 : 

            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      bitmap  sparse      bitmap          bitmap  (method: 6)
            //      bitmap  sparse      bitmap          full    (method: 6)
            //      bitmap  sparse      full            bitmap  (method: 6)

// GB_EMULT_METHOD7 : 

            //      ------------------------------------------
            //      C      <M> =        A       .*      B
            //      ------------------------------------------
            //      bitmap  bitmap      bitmap          bitmap  (method: 7)
            //      bitmap  bitmap      bitmap          full    (method: 7)
            //      bitmap  bitmap      full            bitmap  (method: 7)
            //      bitmap  full        bitmap          bitmap  (method: 7)
            //      bitmap  full        bitmap          full    (method: 7)
            //      bitmap  full        full            bitmap  (method: 7)
            //      ------------------------------------------
            //      C      <!M> =       A       .*      B
            //      ------------------------------------------
            //      bitmap  bitmap      bitmap          bitmap  (method: 7)
            //      bitmap  bitmap      bitmap          full    (method: 7)
            //      bitmap  bitmap      full            bitmap  (method: 7)
            //      bitmap  full        bitmap          bitmap  (method: 7)
            //      bitmap  full        bitmap          full    (method: 7)
            //      bitmap  full        full            bitmap  (method: 7)

            // For methods 5, 6, and 7, C is constructed as bitmap.
            // Both A and B are bitmap/full.  M is either not present,
            // complemented, or not complemented and bitmap/full.  The
            // case when M is not complemented and sparse/hyper is handled
            // by method 100, which constructs C as sparse/hyper (the same
            // structure as M), not bitmap.

// TODO: if C is bitmap on input and C_sparsity is GxB_BITMAP, then C=A.*B,
// C<M>=A.*B and C<M>+=A.*B can all be done in-place.

#include "ewise/GB_ewise.h"
#include "emult/GB_emult.h"
#include "binaryop/GB_binop.h"
#include "jitifyer/GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "FactoryKernels/GB_ew__include.h"
#endif

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_WERK_POP (M_ek_slicing, int64_t) ;   \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORKSPACE ;                     \
    GB_phybix_free (C) ;                    \
}

GrB_Info GB_emult_bitmap    // C=A.*B, C<M>=A.*B, or C<!M>=A.*B
(
    GrB_Matrix C,           // output matrix, static header
    const int ewise_method,
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask, unused if NULL
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use !M
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

    ASSERT_MATRIX_OK (A, "A for bitmap emult ", GB0) ;
    ASSERT_MATRIX_OK (B, "B for bitmap emult ", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for bitmap emult ", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for bitmap emult ", GB0) ;

    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B)) ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;
    int M_ntasks = 0 ; int M_nthreads = 0 ;

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    // M can be jumbled
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (M) ;

    GBURBLE ("emult_bitmap:(B<%s%s%s>=%s.*%s) ",
        Mask_comp ? "!" : "",
        GB_sparsity_char_matrix (M),
        Mask_struct ? ",struct" : "",
        GB_sparsity_char_matrix (A),
        GB_sparsity_char_matrix (B)) ;

    //--------------------------------------------------------------------------
    // determine how many threads to use
    //--------------------------------------------------------------------------

    int64_t cnz = GB_nnz_full (A) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int C_nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;

    // slice the M matrix for Method6
    if (ewise_method == GB_EMULT_METHOD6)
    { 
        GB_SLICE_MATRIX (M, 8) ;
    }

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

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    const size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_emult_iso (cscalar, ctype, A, B, op) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    // allocate the result C (but do not allocate C->p or C->h)
    GB_OK (GB_new_bix (&C, // bitmap, existing header
        ctype, A->vlen, A->vdim, GB_ph_null, C_is_csc,
        GxB_BITMAP, true, A->hyper_switch, -1, cnz, true, C_iso,
        /* OK: */ false, false, false)) ;

    C->magic = GB_MAGIC ;
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
        GB_BURBLE_MATRIX (C, "(iso bitmap emult) ") ;
        memcpy (C->x, cscalar, csize) ;

        // pattern of C = set intersection of pattern of A and B
        #define GB_ISO_EMULT
        #include "emult/template/GB_emult_bitmap_template.c"
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

            #define GB_AemultB_bitmap(mult,xname) \
                GB (_AemultB_bitmap_ ## mult ## xname)

            #define GB_BINOP_WORKER(mult,xname)                             \
            {                                                               \
                info = GB_AemultB_bitmap(mult,xname) (C, M,                 \
                    Mask_struct, Mask_comp, A, B,                           \
                    M_ek_slicing, M_ntasks, M_nthreads, C_nthreads) ;       \
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
        info = GB_emult_bitmap_jit (C, M, Mask_struct,
            Mask_comp, op, flipij, A, B, M_ek_slicing, M_ntasks, M_nthreads,
            C_nthreads) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        GB_BURBLE_MATRIX (C, "(generic bitmap emult: %s) ", op->name) ;
        info = GB_emult_generic (C, op, flipij, NULL, 0, C_nthreads,
            NULL, NULL, NULL, GxB_BITMAP, ewise_method, NULL,
            M_ek_slicing, M_ntasks, M_nthreads, NULL, 0, 0, NULL, 0, 0,
            M, Mask_struct, Mask_comp, A, B) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (info != GrB_SUCCESS)
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (C, "C output for emult_bitmap", GB0) ;
    (*mask_applied) = (M != NULL) ;
    return (GrB_SUCCESS) ;
}

