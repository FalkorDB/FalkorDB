//------------------------------------------------------------------------------
// GB_emult_08_phase2: C=A.*B, C<M>=A.*B, or C<!M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_emult_08_phase2 computes C=A.*B, C<M>=A.*B, or C<!M>=A.*B.  It is
// preceded first by GB_emult_08_phase0, which finds the list of vectors of
// C to compute (Ch, either NULL, or a shallow copy of A->h, B->h, or M->h) and
// their location in M, A, and B (C_to_[MAB]).  Next, GB_emult_08_phase1 counts
// the entries in each vector C(:,j) and computes Cp.

// GB_emult_08_phase2 computes the pattern and values of each vector of C(:,j),
// entirely in parallel.

// C is sparse or hypersparse; M, A, and B can be have any sparsity structure.
// If M is sparse or hypersparse, and complemented, however, then it is not
// applied and not passed to this function.  It is applied later, as determined
// by GB_emult_sparsity.

// This function either frees Cp or transplants it into C, as C->p.  Either
// way, the caller must not free it.

#include "ewise/GB_ewise.h"
#include "emult/GB_emult.h"
#include "binaryop/GB_binop.h"
#include "jitifyer/GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "FactoryKernels/GB_ew__include.h"
#endif

#define GB_FREE_ALL             \
{                               \
    GB_phybix_free (C) ;        \
}

GrB_Info GB_emult_08_phase2             // C=A.*B or C<M>=A.*B
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    const bool flipij,      // if true, i,j must be flipped
    // from phase1:
    void **Cp_handle,       // vector pointers for C
    size_t Cp_size,
    const int64_t Cnvec_nonempty,       // # of non-empty vectors in C
    // tasks from phase1a:
    const GB_task_struct *restrict TaskList, // array of structs
    const int C_ntasks,                         // # of tasks
    const int C_nthreads,                       // # of threads to use
    // analysis from phase0:
    const int64_t Cnvec,
    const void *Ch,
    size_t Ch_size,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const bool Cp_is_32,
    const bool Cj_is_32,
    const bool Ci_is_32,
    const int C_sparsity,
    // from GB_emult_sparsity:
    const int ewise_method,
    // original input:
    const GrB_Matrix M,             // optional mask, may be NULL
    const bool Mask_struct,         // if true, use the only structure of M
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL && (C->header_size == 0 || GBNSTATIC)) ;

    ASSERT_BINARYOP_OK (op, "op for emult phase2", GB0) ;
    ASSERT_MATRIX_OK (A, "A for emult 08 phase2", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    ASSERT_MATRIX_OK (B, "B for emult 08 phase2", GB0) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;
    ASSERT (!GB_PENDING (B)) ;

    ASSERT_MATRIX_OK_OR_NULL (M, "M for 08 emult phase2", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (!GB_JUMBLED (M)) ;
    ASSERT (!GB_PENDING (M)) ;

    ASSERT (A->vdim == B->vdim) ;

    ASSERT (Cp_handle != NULL) ;

    GB_MDECL (Cp, , u) ;
    Cp = (*Cp_handle) ;
    GB_IPTR (Cp, Cp_is_32) ;

    //--------------------------------------------------------------------------
    // get the opcode
    //--------------------------------------------------------------------------

    bool C_is_hyper = (C_sparsity == GxB_HYPERSPARSE) ;
    ASSERT (C_is_hyper || (C_sparsity == GxB_SPARSE)) ;
    ASSERT (Cp != NULL) ;
    ASSERT (C_is_hyper == (Ch != NULL)) ;

    GB_Opcode opcode = op->opcode ;
    bool op_is_builtin_positional =
        GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (opcode) ;
    bool op_is_index_binop = GB_IS_INDEXBINARYOP_CODE (opcode) ;
    bool op_is_positional = op_is_builtin_positional || op_is_index_binop ;
    bool op_is_first  = (opcode == GB_FIRST_binop_code) ;
    bool op_is_second = (opcode == GB_SECOND_binop_code) ;
    bool op_is_pair   = (opcode == GB_PAIR_binop_code) ;

    ASSERT (GB_Type_compatible (ctype, op->ztype)) ;
    ASSERT (GB_IMPLIES (!(op_is_second || op_is_pair
        || op_is_builtin_positional),
        GB_Type_compatible (A->type, op->xtype))) ;
    ASSERT (GB_IMPLIES (!(op_is_first || op_is_pair
        || op_is_builtin_positional),
        GB_Type_compatible (B->type, op->ytype))) ;

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    const size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_emult_iso (cscalar, ctype, A, B, op) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    int64_t cnz = GB_IGET (Cp, Cnvec) ;

    // allocate the result C (but do not allocate C->p or C->h)
    GrB_Info info = GB_new_bix (&C, // sparse/hyper, existing header
        ctype, A->vlen, A->vdim, GB_ph_null, C_is_csc,
        C_sparsity, true, A->hyper_switch, Cnvec, cnz, true, C_iso,
        Cp_is_32, Cj_is_32, Ci_is_32) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory; caller must free C_to_M, C_to_A, C_to_B
        // Ch must not be freed since Ch is always shallow
        GB_FREE_MEMORY (Cp_handle, Cp_size) ;
        return (info) ;
    }

    ASSERT (C->p_is_32 == Cp_is_32) ;
    ASSERT (C->j_is_32 == Cj_is_32) ;
    ASSERT (C->i_is_32 == Ci_is_32) ;

    // transplant Cp into C as the vector pointers, from GB_emult_08_phase1
//  C->nvec_nonempty = Cnvec_nonempty ;
    GB_nvec_nonempty_set (C, Cnvec_nonempty) ;
    C->p = Cp ; C->p_size = Cp_size ;
    C->nvals = cnz ;
    (*Cp_handle) = NULL ;

    // add Ch as the hypersparse list for C, from GB_emult_08_phase0
    if (C_is_hyper)
    { 
        // C->h is currently shallow; a copy is made at the end
        C->h = (void *) Ch ; C->h_size = Ch_size ;
        C->h_shallow = true ;
        C->nvec = Cnvec ;
    }

    // Cp has been transplanted into C; so it is not freed here
    ASSERT ((*Cp_handle) == NULL) ;
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
        #include "emult/template/GB_emult_08_template.c"
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

            #define GB_AemultB_08(mult,xname) GB (_AemultB_08_ ## mult ## xname)

            #define GB_BINOP_WORKER(mult,xname)                             \
            {                                                               \
                info = GB_AemultB_08(mult,xname) (C, M,                     \
                    Mask_struct, Mask_comp, A, B, C_to_M, C_to_A, C_to_B,   \
                    TaskList, C_ntasks, C_nthreads) ;                       \
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
        info = GB_emult_08_jit (C, C_sparsity, M, Mask_struct, Mask_comp,
            op, flipij, A, B, C_to_M, C_to_A, C_to_B, TaskList, C_ntasks,
            C_nthreads) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        GB_BURBLE_MATRIX (C, "(generic emult: %s) ", op->name) ;
        info = GB_emult_generic (C, op, flipij, TaskList, C_ntasks, C_nthreads,
            C_to_M, C_to_A, C_to_B, C_sparsity, ewise_method, NULL,
            NULL, 0, 0, NULL, 0, 0, NULL, 0, 0,
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

    ASSERT_MATRIX_OK (C, "C before hyper prune for emult 08 phase2", GB0) ;
    GB_OK (GB_hyper_prune (C, Werk)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C output for emult 08 phase2", GB0) ;
    return (GrB_SUCCESS) ;
}

