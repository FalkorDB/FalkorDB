//------------------------------------------------------------------------------
// GB_subassign_23: C += A where C is full and A is any matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 23: C += A, where C is full

// M:           NULL
// Mask_comp:   false
// Mask_struct: ignored
// C_replace:   false
// accum:       present
// A:           matrix
// S:           none

// The type of C must match the type of x and z for the accum function, since
// C(i,j) = accum (C(i,j), A(i,j)) is handled.  The generic case here can
// typecast A(i,j) but not C(i,j).  The case for typecasting of C is handled by
// Method 04.

// C and A can have any sparsity structure, but C must be as-if-full.

#include "binaryop/GB_binop.h"
#include "assign/GB_subassign_dense.h"
#include "jitifyer/GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "FactoryKernels/GB_aop__include.h"
#endif
#include "include/GB_unused.h"
#define GB_GENERIC
#define GB_SCALAR_ASSIGN 0
#include "assign/include/GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

GrB_Info GB_subassign_23      // C += A; C is full
(
    GrB_Matrix C,                   // input/output matrix
    const GrB_Matrix A,             // input matrix
    const GrB_BinaryOp accum,       // operator to apply
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix S = NULL ;           // not constructed
    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A

    ASSERT_MATRIX_OK (C, "C for C+=A", GB0) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (GB_IS_FULL (C)) ;

    ASSERT_MATRIX_OK (A, "A for C+=A", GB0) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT_BINARYOP_OK (accum, "accum for C+=A", GB0) ;
    ASSERT (!GB_OP_IS_POSITIONAL (accum)) ;
    ASSERT (A->vlen == C->vlen) ;
    ASSERT (A->vdim == C->vdim) ;

    //--------------------------------------------------------------------------
    // get the operator
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    if (accum->opcode == GB_FIRST_binop_code || C->iso)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    // C = accum (C,A) will be computed
    ASSERT (!C->iso) ;
    // TODO: the types of C, Z, and X need not match for the JIT kernel
    ASSERT (C->type == accum->ztype) ;
    ASSERT (C->type == accum->xtype) ;
    ASSERT (GB_Type_compatible (A->type, accum->ytype)) ;

    //--------------------------------------------------------------------------
    // via the factory kernel
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;

    #ifndef GBCOMPACT
    GB_IF_FACTORY_KERNELS_ENABLED
    { 

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_sub23(accum,xname) GB (_subassign_23_ ## accum ## xname)
        #define GB_BINOP_WORKER(accum,xname)                    \
        {                                                       \
            info = GB_sub23 (accum,xname) (C, A, Werk) ;        \
        }                                                       \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        GB_Opcode opcode ;
        GB_Type_code xcode, ycode, zcode ;
        // C = C + A so A must cast to the Y input of the accum operator.  To
        // use the factory kernel, A->type and accum->ytype must be identical.
        if (/* C->type == accum->ztype && C->type == accum->xtype && */
            GB_binop_builtin (C->type, false, A->type, false,
            accum, false, &opcode, &xcode, &ycode, &zcode))
        { 
            // accumulate sparse matrix into full matrix with built-in operator
            #define GB_NO_FIRST
            #define GB_XTYPE_AND_ZTYPE_MUST_MATCH
            #include "binaryop/factory/GB_binop_factory.c"
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        info = GB_subassign_jit (C,
            /* C_replace: */ false,
            /* I, ni, nI, Ikind, Icolon: */ NULL, false, 0, 0, GB_ALL, NULL,
            /* J, nj, nJ, Jkind, Jcolon: */ NULL, false, 0, 0, GB_ALL, NULL,
            /* M: */ NULL,
            /* Mask_comp: */ false,
            /* Mask_struct: */ true,
            /* accum: */ accum,
            /* A: */ A,
            /* scalar, scalar_type: */ NULL, NULL,
            /* S: */ NULL,
            GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_23, "subassign_23",
            Werk) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        #include "generic/GB_generic.h"
        GB_BURBLE_MATRIX (A, "(generic C+=A) ") ;

        GxB_binary_function faccum = accum->binop_function ;

        size_t csize = C->type->size ;
        size_t asize = A->type->size ;
        size_t ysize = accum->ytype->size ;
        // A is typecasted to y
        GB_cast_function cast_A_to_Y ;
        cast_A_to_Y = GB_cast_factory (accum->ytype->code, A->type->code) ;

        // get the iso value of A
        GB_DECLAREY (ywork) ;
        if (A->iso)
        {
            // ywork = (ytype) Ax [0]
            cast_A_to_Y (ywork, A->x, asize) ;
        }

        #define C_iso false

        #undef  GB_ACCUMULATE_aij
        #define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork,C_iso)        \
        {                                                               \
            /* Cx [pC] += (ytype) Ax [A_iso ? 0 : pA] */                \
            if (A_iso)                                                  \
            {                                                           \
                faccum (Cx +((pC)*csize), Cx +((pC)*csize), ywork) ;    \
            }                                                           \
            else                                                        \
            {                                                           \
                GB_DECLAREY (ywork) ;                                   \
                cast_A_to_Y (ywork, Ax +((pA)*asize), asize) ;          \
                faccum (Cx +((pC)*csize), Cx +((pC)*csize), ywork) ;    \
            }                                                           \
        }

        #include "assign/template/GB_subassign_23_template.c"
        info = GrB_SUCCESS ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    {
        ASSERT_MATRIX_OK (C, "C+=A output", GB0) ;
    }
    return (info) ;
}

