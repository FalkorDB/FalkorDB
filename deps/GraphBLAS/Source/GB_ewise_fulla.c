//------------------------------------------------------------------------------
// GB_ewise_fulla: C += A+B where all 3 matries are full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C += A+B where no matrix is iso and all three matrices are as-if-full

// JIT: done.

#include "GB_ewise.h"
#include "GB_binop.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_ew__include.h"
#endif

GrB_Info GB_ewise_fulla        // C += A+B, all matrices full
(
    GrB_Matrix C,                   // input/output matrix
    const GrB_BinaryOp op,          // only GB_BINOP_SUBSET operators supported
    const GrB_Matrix A,
    const GrB_Matrix B
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for full C+=A+B", GB0) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ;

    ASSERT_MATRIX_OK (A, "A for full C+=A+B", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    ASSERT_MATRIX_OK (B, "B for full C+=A+B", GB0) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;
    ASSERT (!GB_PENDING (B)) ;

    ASSERT (GB_IS_FULL (C)) ;
    ASSERT (GB_IS_FULL (A)) ;
    ASSERT (GB_IS_FULL (B)) ;

    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (!GB_IS_BITMAP (B)) ;

    ASSERT (!C->iso) ;
    ASSERT (!A->iso) ;
    ASSERT (!B->iso) ;

    ASSERT_BINARYOP_OK (op, "op for full C+=A+B", GB0) ;
    ASSERT (!GB_OP_IS_POSITIONAL (op)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t cnz = GB_nnz (C) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (3 * cnz, chunk, nthreads_max) ;

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

        #define GB_Cewise_fulla(op,xname) \
            GB (_Cewise_fulla_ ## op ## xname)

        #define GB_BINOP_WORKER(op,xname)                               \
        {                                                               \
            info = GB_Cewise_fulla(op,xname) (C, A, B, nthreads) ;      \
        }                                                               \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        GB_Opcode opcode ;
        GB_Type_code xcode, ycode, zcode ;
        if (GB_binop_builtin (A->type, false, B->type, false,
            op, false, &opcode, &xcode, &ycode, &zcode))
        { 
            #define GB_BINOP_SUBSET
            #include "GB_binop_factory.c"
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        info = GB_ewise_fulla_jit (C, op, A, B, nthreads) ;
    }

    // no generic kernel: returns GrB_NO_VALUE if no factory kernel exists and
    // no JIT kernel created.

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    { 
        ASSERT_MATRIX_OK (C, "C output, full C+=A+B", GB0) ;
    }
    return (info) ;
}

