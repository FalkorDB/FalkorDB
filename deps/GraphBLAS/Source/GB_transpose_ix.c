//------------------------------------------------------------------------------
// GB_transpose_ix: transpose the values and pattern of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

// The values of A are typecasted to C->type, the type of the C matrix.

// If A is sparse or hypersparse
//      The pattern of C is constructed.  C is sparse.
//      Workspaces and A_slice are non-NULL.
//      This method is parallel, but not highly scalable.  It uses only
//      nthreads = nnz(A)/(A->vlen) threads.

// If A is full or as-if-full:
//      The pattern of C is not constructed.  C is full.
//      Workspaces and A_slice are NULL.
//      This method is parallel and fully scalable.

// If A is bitmap:
//      C->b is constructed.  C is bitmap.
//      Workspaces and A_slice are NULL.
//      This method is parallel and fully scalable.

#include "GB_transpose.h"
#include "GB_unop.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_unop__include.h"
#endif

GrB_Info GB_transpose_ix        // transpose the pattern and values of a matrix
(
    GrB_Matrix C,                       // output matrix
    const GrB_Matrix A,                 // input matrix
    // for sparse case:
    int64_t *restrict *Workspaces,      // Workspaces, size nworkspaces
    const int64_t *restrict A_slice,    // how A is sliced, size nthreads+1
    int nworkspaces,                    // # of workspaces to use
    // for all cases:
    int nthreads                        // # of threads to use
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    GrB_Info info = GrB_NO_VALUE ;
    GrB_Type ctype = C->type ;
    GB_Type_code code1 = ctype->code ;          // defines ztype
    GB_Type_code code2 = A->type->code ;        // defines atype
    size_t asize = A->type->size ;

    //--------------------------------------------------------------------------
    // built-in worker: transpose and typecast
    //--------------------------------------------------------------------------

    if (C->iso)
    { 

        //----------------------------------------------------------------------
        // via the iso kernel
        //----------------------------------------------------------------------

        // A and C are iso: Cx [0] = (ctype) Ax [0]
        GB_cast_scalar (C->x, code1, A->x, code2, asize) ;

        // C = pattern of A transposed
        #define GB_ISO_TRANSPOSE
        #include "GB_transpose_template.c"
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

            #define GB_unop_tran(zname,aname)                               \
                GB (_unop_tran__identity ## zname ## aname)

            #define GB_WORKER(ignore1,zname,ztype,aname,atype)              \
            {                                                               \
                info = GB_unop_tran (zname,aname)                           \
                    (C, A, Workspaces, A_slice, nworkspaces, nthreads) ;    \
            }                                                               \
            break ;

            //------------------------------------------------------------------
            // launch the switch factory
            //------------------------------------------------------------------

            #include "GB_twotype_factory.c"
        }
        #endif

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            struct GB_UnaryOp_opaque op_header ;
            GrB_Type ctype = C->type ;
            GB_Operator op = GB_unop_identity (ctype, &op_header) ;
            ASSERT_OP_OK (op, "identity op for transpose_ix", GB0) ;
            info = GB_transpose_unop_jit (C, op, A, Workspaces,
                A_slice, nworkspaces, nthreads) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            GB_BURBLE_MATRIX (A, "(generic transpose) ") ;
            size_t csize = C->type->size ;
            GB_cast_function cast_A_to_X = GB_cast_factory (code1, code2) ;

            // Cx [pC] = (ctype) Ax [pA]
            #define GB_APPLY_OP(pC,pA)  \
                cast_A_to_X (Cx +((pC)*csize), Ax +((pA)*asize), asize) ;
            #define GB_A_TYPE GB_void
            #define GB_C_TYPE GB_void
            #include "GB_transpose_template.c"
            info = GrB_SUCCESS ;
        }
    }

    return (info) ;
}
