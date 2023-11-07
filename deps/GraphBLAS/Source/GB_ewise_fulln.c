//------------------------------------------------------------------------------
// GB_ewise_fulln: C = A+B where A and B are full, C is anything
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

// C can have any sparsity on input; it becomes a full non-iso matrix on output.
// C can have pending work, which is discarded.

#include "GB_ewise.h"
#include "GB_binop.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_ew__include.h"
#endif

#define GB_FREE_ALL ;

GrB_Info GB_ewise_fulln      // C = A+B
(
    GrB_Matrix C,                   // input/output matrix
    const GrB_BinaryOp op,          // must not be a positional op
    const GrB_Matrix A,
    const GrB_Matrix B
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    ASSERT_MATRIX_OK (C, "C for full C=A+B", GB0) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;    // C is entirely overwritten by A+B
    ASSERT (GB_PENDING_OK (C)) ;

    ASSERT_MATRIX_OK (A, "A for full C=A+B", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    ASSERT_MATRIX_OK (B, "B for full C=A+B", GB0) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;
    ASSERT (!GB_PENDING (B)) ;

    ASSERT (GB_IS_FULL (A)) ;
    ASSERT (GB_IS_FULL (B)) ;

    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (!GB_IS_BITMAP (B)) ;

    ASSERT (!A->iso) ;
    ASSERT (!B->iso) ;

    ASSERT_BINARYOP_OK (op, "op for full C=A+B", GB0) ;
    ASSERT (!GB_OP_IS_POSITIONAL (op)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t anz = GB_nnz (A) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (2 * anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // if C not already full, allocate it as full
    //--------------------------------------------------------------------------

    // clear prior content and create C as a full matrix.  Keep the same type
    // and CSR/CSC for C.  Allocate the values of C but do not initialize them.

    if (!GB_as_if_full (C))
    { 
        // free the content of C and reallocate it as a non-iso full matrix
        ASSERT (C != A && C != B) ;
        GB_phybix_free (C) ;
        // set C->iso = false   OK
        GB_OK (GB_new_bix (&C,  // existing header
            C->type, C->vlen, C->vdim, GB_Ap_null, C->is_csc, GxB_FULL, false,
            C->hyper_switch, -1, GB_nnz_full (C), true, false)) ;
        C->magic = GB_MAGIC ;
    }
    else if (!GB_IS_FULL (C))
    { 
        // ensure C is full
        GB_convert_any_to_full (C) ;
    }
    ASSERT (GB_IS_FULL (C)) ;

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

        #define GB_Cewise_fulln(op,xname) \
            GB (_Cewise_fulln_ ## op ## xname)

        #define GB_BINOP_WORKER(op,xname)                           \
        {                                                           \
            info = GB_Cewise_fulln(op,xname) (C, A, B, nthreads) ;  \
        }                                                           \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        GB_Opcode opcode ;
        GB_Type_code xcode, ycode, zcode ;
        if (GB_binop_builtin (A->type, false, B->type, false,
            op, false, &opcode, &xcode, &ycode, &zcode))
        { 
            #include "GB_binop_factory.c"
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        info = GB_ewise_fulln_jit (C, op, A, B, nthreads) ;
    }

    // no generic kernel: returns GrB_NO_VALUE if no factory kernel exists and
    // no JIT kernel created.

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    { 
        ASSERT_MATRIX_OK (C, "C output, full C=A+B", GB0) ;
    }
    return (info) ;
}

