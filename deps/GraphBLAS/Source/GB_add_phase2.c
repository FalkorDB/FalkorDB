//------------------------------------------------------------------------------
// GB_add_phase2: C=A+B or C<M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

// GB_add_phase2 computes C=A+B, C<M>=A+B, or C<!M>A+B.  It is preceded first
// by GB_add_phase0, which computes the list of vectors of C to compute (Ch)
// and their location in A and B (C_to_[AB]).  Next, GB_add_phase1 counts the
// entries in each vector C(:,j) and computes Cp.

// GB_add_phase2 computes the pattern and values of each vector of C(:,j),
// entirely in parallel.

// C, M, A, and B can be standard sparse or hypersparse, as determined by
// GB_add_phase0.  The mask can be either: not present, or present and
// not complemented.  The complemented mask is handled in most cases,
// except when C, M, A, and B are all sparse or hypersparse.

// This function either frees Cp and Ch, or transplants then into C, as C->p
// and C->h.  Either way, the caller must not free them.

// op may be NULL.  In this case, the intersection of A and B must be empty.
// This is used by GB_wait only, for merging the pending tuple matrix T into A.
// In this case, C is always sparse or hypersparse, not bitmap or full.

#include "GB_add.h"
#include "GB_binop.h"
#include "GB_unused.h"
#include "GB_ek_slice.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_ew__include.h"
#endif

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_WERK_POP (B_ek_slicing, int64_t) ;   \
    GB_WERK_POP (A_ek_slicing, int64_t) ;   \
    GB_WERK_POP (M_ek_slicing, int64_t) ;   \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                 \
{                                   \
    GB_FREE_WORKSPACE ;             \
    GB_phybix_free (C) ;            \
}

GrB_Info GB_add_phase2      // C=A+B, C<M>=A+B, or C<!M>=A+B
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    const bool A_and_B_are_disjoint,    // if true, then A and B are disjoint
    // from phase1:
    int64_t **Cp_handle,    // vector pointers for C
    size_t Cp_size,
    const int64_t Cnvec_nonempty,   // # of non-empty vectors in C
    // tasks from phase1a:
    const GB_task_struct *restrict TaskList,    // array of structs
    const int C_ntasks,         // # of tasks
    const int C_nthreads,       // # of threads to use
    // analysis from phase0:
    const int64_t Cnvec,
    int64_t **Ch_handle,
    size_t Ch_size,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const bool Ch_is_Mh,        // if true, then Ch == M->h
    const int C_sparsity,
    // original input:
    const GrB_Matrix M,         // optional mask, may be NULL
    const bool Mask_struct,     // if true, use the only structure of M
    const bool Mask_comp,       // if true, use !M
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool is_eWiseUnion,   // if true, eWiseUnion, else eWiseAdd
    const GrB_Scalar alpha,     // alpha and beta ignored for eWiseAdd,
    const GrB_Scalar beta,      // nonempty scalars for GxB_eWiseUnion
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;
    ASSERT_BINARYOP_OK (op, "op for add phase2", GB0) ;
    ASSERT_MATRIX_OK (A, "A for add phase2", GB0) ;
    ASSERT_MATRIX_OK (B, "B for add phase2", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for add phase2", GB0) ;
    ASSERT (A->vdim == B->vdim) ;

    ASSERT (!GB_JUMBLED (M)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_JUMBLED (B)) ;

    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;
    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    GB_WERK_DECLARE (B_ek_slicing, int64_t) ;

    ASSERT (Cp_handle != NULL) ;
    ASSERT (Ch_handle != NULL) ;
    int64_t *restrict Cp = (*Cp_handle) ;
    int64_t *restrict Ch = (*Ch_handle) ;

    //--------------------------------------------------------------------------
    // get the opcode
    //--------------------------------------------------------------------------

    bool C_is_hyper = (C_sparsity == GxB_HYPERSPARSE) ;
    bool C_is_sparse_or_hyper = (C_sparsity == GxB_SPARSE) || C_is_hyper ;
    ASSERT (C_is_sparse_or_hyper == (Cp != NULL)) ;
    ASSERT (C_is_hyper == (Ch != NULL)) ;

    GB_Opcode opcode = op->opcode ;
    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
    bool op_is_first  = (opcode == GB_FIRST_binop_code) ;
    bool op_is_second = (opcode == GB_SECOND_binop_code) ;
    bool op_is_pair   = (opcode == GB_PAIR_binop_code) ;

#ifdef GB_DEBUG
    // assert that the op is compatible with A, B, and C
    if (!(GB_IS_FULL (A) && GB_IS_FULL (B)))
    {
        // eWiseMult uses GB_add when A and B are both as-if-full,
        // and in this case, the entries of A and B are never typecasted
        // directly to C.
        ASSERT (GB_Type_compatible (ctype, A->type)) ;
        ASSERT (GB_Type_compatible (ctype, B->type)) ;
    }
    ASSERT (GB_Type_compatible (ctype, op->ztype)) ;
    ASSERT (GB_IMPLIES (!(op_is_second || op_is_pair || op_is_positional),
            GB_Type_compatible (A->type, op->xtype))) ;
    ASSERT (GB_IMPLIES (!(op_is_first  || op_is_pair || op_is_positional),
            GB_Type_compatible (B->type, op->ytype))) ;
#endif

    //--------------------------------------------------------------------------
    // get the typecasting functions
    //--------------------------------------------------------------------------

    size_t asize, bsize, xsize, ysize, zsize ;
    GB_cast_function cast_A_to_C = NULL, cast_B_to_C = NULL ;
    GB_cast_function cast_A_to_X, cast_B_to_Y, cast_Z_to_C ;
    const size_t csize = ctype->size ;
    GB_Type_code ccode = ctype->code ;

    if (A_and_B_are_disjoint)
    { 
        // GB_wait: GB_SECOND_[type] operator with no typecasting
        ASSERT (!is_eWiseUnion) ;
        asize = csize ;
        bsize = csize ;
        xsize = csize ;
        ysize = csize ;
        zsize = csize ;
        cast_A_to_X = GB_copy_user_user ;
        cast_B_to_Y = GB_copy_user_user ;
        cast_A_to_C = GB_copy_user_user ;
        cast_B_to_C = GB_copy_user_user ;
        cast_Z_to_C = GB_copy_user_user ;
    }
    else
    {
        // normal case, with optional typecasting
        asize = A->type->size ;
        bsize = B->type->size ;

        if (op_is_second || op_is_pair || op_is_positional)
        { 
            // the op does not depend on the value of A(i,j)
            xsize = 1 ;
            cast_A_to_X = NULL ;
        }
        else
        { 
            xsize = op->xtype->size ;
            cast_A_to_X = GB_cast_factory (op->xtype->code, A->type->code) ;
        }

        if (op_is_first || op_is_pair || op_is_positional)
        { 
            // the op does not depend on the value of B(i,j)
            ysize = 1 ;
            cast_B_to_Y = NULL ;
        }
        else
        { 
            ysize = op->ytype->size ;
            cast_B_to_Y = GB_cast_factory (op->ytype->code, B->type->code) ;
        }

        zsize = op->ztype->size ;
        if (!is_eWiseUnion)
        { 
            // typecasting for eWiseAdd only
            cast_A_to_C = GB_cast_factory (ccode, A->type->code) ;
            cast_B_to_C = GB_cast_factory (ccode, B->type->code) ;
        }
        cast_Z_to_C = GB_cast_factory (ccode, op->ztype->code) ;
    }

    //--------------------------------------------------------------------------
    // cast the alpha and beta scalars, if present
    //--------------------------------------------------------------------------

    GB_void alpha_scalar [GB_VLA(xsize)] ;
    GB_void beta_scalar  [GB_VLA(ysize)] ;
    if (is_eWiseUnion)
    { 
        // alpha_scalar = (xtype) alpha
        ASSERT (alpha != NULL) ;
        GB_cast_scalar (alpha_scalar, op->xtype->code, alpha->x, 
            alpha->type->code, alpha->type->size) ;
        // beta_scalar = (ytype) beta
        ASSERT (beta != NULL) ;
        GB_cast_scalar (beta_scalar, op->ytype->code, beta->x,
            beta->type->code, beta->type->size) ;
    }

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_add_iso (cscalar, ctype, A, alpha_scalar,
        B, beta_scalar, op, A_and_B_are_disjoint, is_eWiseUnion) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C: hypersparse, sparse, bitmap, or full
    //--------------------------------------------------------------------------

    // C is hypersparse if both A and B are (contrast with GrB_Matrix_emult),
    // or if M is present, not complemented, and hypersparse.
    // C acquires the same hyperatio as A.

    int64_t cnz = (C_is_sparse_or_hyper) ? (Cp [Cnvec]) : GB_nnz_full (A) ;

    // allocate the result C (but do not allocate C->p or C->h)
    // set C->iso = C_iso   OK
    GrB_Info info = GB_new_bix (&C, // any sparsity, existing header
        ctype, A->vlen, A->vdim, GB_Ap_null, C_is_csc,
        C_sparsity, true, A->hyper_switch, Cnvec, cnz, true, C_iso) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory; caller must free C_to_M, C_to_A, C_to_B
        GB_FREE_ALL ;
        GB_FREE (Cp_handle, Cp_size) ;
        GB_FREE (Ch_handle, Ch_size) ;
        return (info) ;
    }

    // add Cp as the vector pointers for C, from GB_add_phase1
    if (C_is_sparse_or_hyper)
    { 
        C->nvec_nonempty = Cnvec_nonempty ;
        C->p = (int64_t *) Cp ; C->p_size = Cp_size ;
        (*Cp_handle) = NULL ;
        C->nvals = cnz ;
    }

    // add Ch as the hypersparse list for C, from GB_add_phase0
    if (C_is_hyper)
    { 
        C->h = (int64_t *) Ch ; C->h_size = Ch_size ;
        C->nvec = Cnvec ;
        (*Ch_handle) = NULL ;
    }

    // now Cp and Ch have been transplanted into C
    ASSERT ((*Cp_handle) == NULL) ;
    ASSERT ((*Ch_handle) == NULL) ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // slice M, A, and B if needed
    //--------------------------------------------------------------------------

    int M_nthreads = 0, M_ntasks = 0 ;
    int A_nthreads = 0, A_ntasks = 0 ;
    int B_nthreads = 0, B_ntasks = 0 ;

    if (!C_is_sparse_or_hyper)
    {
        // if C is bitmap/full, then each matrix M, A, and B needs to be sliced
        // if they are sparse/hyper
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        if (M != NULL && (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)))
        { 
            GB_SLICE_MATRIX (M, 8) ;
        }
        if (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A))
        { 
            GB_SLICE_MATRIX (A, 8) ;
        }
        if (GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B))
        { 
            GB_SLICE_MATRIX (B, 8) ;
        }
    }

    //--------------------------------------------------------------------------
    // for the "easy mask" condition:
    //--------------------------------------------------------------------------

    bool M_is_A = GB_all_aliased (M, A) ;
    bool M_is_B = GB_all_aliased (M, B) ;

    //--------------------------------------------------------------------------
    // using a built-in binary operator (except for positional operators)
    //--------------------------------------------------------------------------

    #include "GB_ewise_shared_definitions.h"
    #define GB_ADD_PHASE 2

    info = GrB_NO_VALUE ;

    if (C_iso)
    { 

        //----------------------------------------------------------------------
        // via the iso kernel
        //----------------------------------------------------------------------

        // Cx [0] = cscalar = op (A,B)
        GB_BURBLE_MATRIX (C, "(iso add) ") ;
        memcpy (C->x, cscalar, csize) ;

        // pattern of C = set union of pattern of A and B.
        // eWiseAdd and eWiseUnion are identical since no numerical values
        // are used, and the operator is not used.
        #define GB_ISO_ADD
        #define GB_IS_EWISEUNION 0
        #include "GB_add_template.c"
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
            GB_Type_code xcode, ycode, zcode ;
            if (!op_is_positional &&
                GB_binop_builtin (A->type, false, B->type, false,
                op, false, &opcode, &xcode, &ycode, &zcode) && ccode == zcode)
            { 

                #define GB_AaddB(mult,xname) GB (_AaddB_ ## mult ## xname)
                #define GB_AunionB(mult,xname) GB (_AunionB_ ## mult ## xname)

                if (is_eWiseUnion)
                { 

                    //----------------------------------------------------------
                    // define the worker for the switch factory
                    //----------------------------------------------------------

                    #define GB_BINOP_WORKER(mult,xname)                 \
                        info = GB_AunionB(mult,xname) (C, C_sparsity,   \
                            M, Mask_struct, Mask_comp, A, B,            \
                            alpha_scalar, beta_scalar,                  \
                            Ch_is_Mh, C_to_M, C_to_A, C_to_B,           \
                            TaskList, C_ntasks, C_nthreads,             \
                            M_ek_slicing, M_nthreads, M_ntasks,         \
                            A_ek_slicing, A_nthreads, A_ntasks,         \
                            B_ek_slicing, B_nthreads, B_ntasks) ;       \
                        break ;

                    //----------------------------------------------------------
                    // launch the switch factory
                    //----------------------------------------------------------

                    // eWiseUnion is like emult: the pair results in C being iso
                    #define GB_NO_PAIR
                    #include "GB_binop_factory.c"

                }
                else
                { 

                    //----------------------------------------------------------
                    // define the worker for the switch factory
                    //----------------------------------------------------------

                    #undef  GB_BINOP_WORKER
                    #define GB_BINOP_WORKER(mult,xname)                 \
                        info = GB_AaddB(mult,xname) (C, C_sparsity,     \
                            M, Mask_struct, Mask_comp, A, B,            \
                            Ch_is_Mh, C_to_M, C_to_A, C_to_B,           \
                            TaskList, C_ntasks, C_nthreads,             \
                            M_ek_slicing, M_nthreads, M_ntasks,         \
                            A_ek_slicing, A_nthreads, A_ntasks,         \
                            B_ek_slicing, B_nthreads, B_ntasks) ;       \
                        break ;

                    //----------------------------------------------------------
                    // launch the switch factory
                    //----------------------------------------------------------

                    #include "GB_binop_factory.c"
                }
            }
        }
        #endif
    }

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    {
        if (is_eWiseUnion)
        { 
            info = GB_union_jit (C, C_sparsity, M, Mask_struct,
                Mask_comp, op, A, B, alpha_scalar, beta_scalar,
                Ch_is_Mh, C_to_M, C_to_A, C_to_B, 
                TaskList, C_ntasks, C_nthreads,
                M_ek_slicing, M_nthreads, M_ntasks,
                A_ek_slicing, A_nthreads, A_ntasks,
                B_ek_slicing, B_nthreads, B_ntasks) ;
        }
        else
        { 
            info = GB_add_jit (C, C_sparsity, M, Mask_struct,
                Mask_comp, op, A, B,
                Ch_is_Mh, C_to_M, C_to_A, C_to_B, 
                TaskList, C_ntasks, C_nthreads,
                M_ek_slicing, M_nthreads, M_ntasks,
                A_ek_slicing, A_nthreads, A_ntasks,
                B_ek_slicing, B_nthreads, B_ntasks) ;
        }
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    {

        #include "GB_generic.h"
        GB_BURBLE_MATRIX (C, "(generic add: %s) ", op->name) ;

        // C(i,j) = (ctype) A(i,j), located in Ax [pA]
        #undef  GB_COPY_A_to_C 
        #define GB_COPY_A_to_C(Cx,pC,Ax,pA,A_iso)                             \
        cast_A_to_C (Cx +((pC)*csize), Ax +((A_iso) ? 0: (pA)*asize), asize) ;

        // C(i,j) = (ctype) B(i,j), located in Bx [pB]
        #undef  GB_COPY_B_to_C
        #define GB_COPY_B_to_C(Cx,pC,Bx,pB,B_iso)                             \
        cast_B_to_C (Cx +((pC)*csize), Bx +((B_iso) ? 0: (pB)*bsize), bsize) ;

        // declare aij as xtype
        #undef  GB_DECLAREA
        #define GB_DECLAREA(aij)                                            \
            GB_void aij [GB_VLA(xsize)] ;

        // aij = (xtype) A(i,j), located in Ax [pA]
        #undef  GB_GETA
        #define GB_GETA(aij,Ax,pA,A_iso)                                    \
            if (cast_A_to_X != NULL)                                        \
            {                                                               \
                cast_A_to_X (aij, Ax +((A_iso) ? 0:(pA)*asize), asize) ;    \
            }

        // declare bij as ytype
        #undef  GB_DECLAREB
        #define GB_DECLAREB(bij)                                            \
            GB_void bij [GB_VLA(ysize)] ;

        // bij = (ytype) B(i,j), located in Bx [pB]
        #undef  GB_GETB
        #define GB_GETB(bij,Bx,pB,B_iso)                                    \
            if (cast_B_to_Y != NULL)                                        \
            {                                                               \
                cast_B_to_Y (bij, Bx +((B_iso) ? 0:(pB)*bsize), bsize) ;    \
            }

        // C (i,j) = (ctype) z
        #undef  GB_PUTC
        #define GB_PUTC(z, Cx, p) cast_Z_to_C (Cx +((p)*csize), &z, csize)

        if (op_is_positional)
        {

            //------------------------------------------------------------------
            // C(i,j) = positional_op (aij, bij)
            //------------------------------------------------------------------

            // z = op (aij, bij)
            #undef  GB_BINOP
            #define GB_BINOP(z,x,y,i,j)                             \
                z = (positional_is_i ? (i):(j)) + offset

            #define GB_POSITIONAL_OP
            const bool positional_is_i = 
                (opcode == GB_FIRSTI_binop_code)    ||
                (opcode == GB_FIRSTI1_binop_code)   ||
                (opcode == GB_SECONDI_binop_code)   ||
                (opcode == GB_SECONDI1_binop_code) ;
            const int64_t offset = GB_positional_offset (opcode, NULL, NULL) ;
            if (op->ztype == GrB_INT64)
            {

                // C(i,j) = positional_op (aij, bij)
                #undef  GB_EWISEOP
                #define GB_EWISEOP(Cx, p, aij, bij, i, j)               \
                {                                                       \
                    int64_t z ;                                         \
                    GB_BINOP (z, , , i, j) ;                            \
                    GB_PUTC (z, Cx, p) ;                                \
                }

                if (is_eWiseUnion)
                { 
                    #define GB_IS_EWISEUNION 1
                    #include "GB_add_template.c"
                }
                else
                { 
                    #define GB_IS_EWISEUNION 0
                    #include "GB_add_template.c"
                }
            }
            else
            {

                // C(i,j) = positional_op (aij, bij)
                #undef  GB_EWISEOP
                #define GB_EWISEOP(Cx, p, aij, bij, i, j)               \
                {                                                       \
                    int64_t z ;                                         \
                    GB_BINOP (z, , , i, j) ;                            \
                    int32_t z32 = (int32_t) z ;                         \
                    GB_PUTC (z32, Cx, p) ;                              \
                }

                if (is_eWiseUnion)
                { 
                    #define GB_IS_EWISEUNION 1
                    #include "GB_add_template.c"
                }
                else
                { 
                    #define GB_IS_EWISEUNION 0
                    #include "GB_add_template.c"
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // standard binary operator
            //------------------------------------------------------------------

            #undef GB_POSITIONAL_OP
            GxB_binary_function fadd = op->binop_function ;

            // The binary op is not used if fadd is null since in that case
            // the intersection of A and B is empty

            // z = op (aij, bij)
            #undef  GB_BINOP
            #define GB_BINOP(z, aij, bij, i, j)             \
                ASSERT (fadd != NULL) ;                     \
                fadd (z, aij, bij) ;

            // C(i,j) = (ctype) (A(i,j) + B(i,j))
            #undef  GB_EWISEOP
            #define GB_EWISEOP(Cx, p, aij, bij, i, j)       \
            {                                               \
                GB_void z [GB_VLA(zsize)] ;                 \
                GB_BINOP (z, aij, bij, i, j) ;              \
                GB_PUTC (z, Cx, p) ;                        \
            }

            if (is_eWiseUnion)
            { 
                #define GB_IS_EWISEUNION 1
                #include "GB_add_template.c"
            }
            else
            { 
                #define GB_IS_EWISEUNION 0
                #include "GB_add_template.c"
            }
        }
        info = GrB_SUCCESS ;
    }

    if (info != GrB_SUCCESS)
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    GB_OK (GB_hypermatrix_prune (C, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    // caller must free C_to_M, C_to_A, and C_to_B, but not Cp or Ch
    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (C, "C output for add phase2", GB0) ;
    return (GrB_SUCCESS) ;
}

