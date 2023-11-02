//------------------------------------------------------------------------------
// GB_subassign_22: C += scalar where C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

// Method 22: C += scalar, where C is full

// M:           NULL
// Mask_comp:   false
// Mask_struct: ignored
// C_replace:   false
// accum:       present
// A:           scalar
// S:           none

// C += scalar where C is full

#include "GB_subassign_dense.h"
#include "GB_assign_shared_definitions.h"
#include "GB_binop.h"
#include "GB_unused.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_aop__include.h"
#endif

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

GrB_Info GB_subassign_22      // C += scalar where C is full
(
    GrB_Matrix C,                   // input/output matrix
    const void *scalar,             // input scalar
    const GrB_Type scalar_type,           // type of the input scalar
    const GrB_BinaryOp accum,       // operator to apply
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for C+=scalar", GB0) ;
    ASSERT (GB_IS_FULL (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_ZOMBIES (C)) ;

    ASSERT (scalar != NULL) ;
    ASSERT_TYPE_OK (scalar_type, "scalar_type for C+=scalar", GB0) ;
    ASSERT_BINARYOP_OK (accum, "accum for C+=scalar", GB0) ;
    ASSERT (!GB_OP_IS_POSITIONAL (accum)) ;

    //--------------------------------------------------------------------------
    // get the operator
    //--------------------------------------------------------------------------

    if (accum->opcode == GB_FIRST_binop_code || C->iso)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    // C = accum (C,scalar) will be computed
    ASSERT (C->type == accum->ztype) ;
    ASSERT (C->type == accum->xtype) ;
    ASSERT (GB_Type_compatible (scalar_type, accum->ytype)) ;

    //--------------------------------------------------------------------------
    // typecast the scalar into the same type as the y input of the binary op
    //--------------------------------------------------------------------------

    int64_t csize = C->type->size ;
    size_t ysize = accum->ytype->size ;
    GB_cast_function 
        cast_A_to_Y = GB_cast_factory (accum->ytype->code, scalar_type->code) ;
    GB_DECLAREY (ywork) ;
    cast_A_to_Y (ywork, scalar, scalar_type->size) ;

    // Since no pending tuples will be created, and no entries will be copied
    // from the scalar into C, only the typecased ywork scalar is needed.  The
    // scalar and its type (scalar_type) can now be ignored.  Only ctype and
    // cwork need to be considered.

    //--------------------------------------------------------------------------
    // via the factory kernel
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    info = GrB_NO_VALUE ;

    #ifndef GBCOMPACT
    GB_IF_FACTORY_KERNELS_ENABLED
    { 

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_sub22(accum,xname) GB (_subassign_22_ ## accum ## xname)

        #define GB_BINOP_WORKER(accum,xname)                                \
        {                                                                   \
            info = GB_sub22 (accum,xname) (C, ywork) ;                      \
        }                                                                   \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        GB_Opcode opcode ;
        GB_Type_code xcode, ycode, zcode ;
        // C = C + scalar where the scalar ywork already matches the Y input of
        // the accum op.  The original scalar_type of the scalar can be ignored.
        if (/* (C->type == accum->ztype) && (C->type == accum->xtype) && */
            GB_binop_builtin (C->type, false, accum->ytype, false,
            accum, false, &opcode, &xcode, &ycode, &zcode))
        { 
            // accumulate sparse matrix into full matrix with built-in operator
            #include "GB_binop_factory.c"
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    // pass (ywork, accum->ytype) in place of (scalar, scalar_type)
    if (info == GrB_NO_VALUE)
    { 
        info = GB_subassign_jit (C,
            /* C_replace: */ false,
            /* I, ni, nI, Ikind, Icolon: */ NULL, 0, 0, GB_ALL, NULL,
            /* J, nj, nJ, Jkind, Jcolon: */ NULL, 0, 0, GB_ALL, NULL,
            /* M: */ NULL,
            /* Mask_comp: */ false,
            /* Mask_struct: */ true,
            /* accum: */ accum,
            /* A: */ NULL,
            /* scalar, scalar_type: */ ywork, accum->ytype,
            GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_22, "subassign_22",
            Werk) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        #include "GB_generic.h"
        GB_BURBLE_MATRIX (C, "(generic C(:,:)+=x assign) ") ;

        GxB_binary_function faccum = accum->binop_function ;

        // C(i,j) = C(i,j) + y
        #undef  GB_ACCUMULATE_scalar
        #define GB_ACCUMULATE_scalar(Cx,pC,ywork)           \
            faccum (Cx +((pC)*csize), Cx +((pC)*csize), ywork)

        #include "GB_subassign_22_template.c"
        info = GrB_SUCCESS ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    {
        ASSERT_MATRIX_OK (C, "C+=scalar output", GB0) ;
    }
    return (info) ;
}

