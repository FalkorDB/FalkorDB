//------------------------------------------------------------------------------
// GB_subassign_05d: C(:,:)<M> = scalar where C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 05d: C(:,:)<M> = scalar ; no S, C is dense

// C:           full
// M:           present, any sparsity structure
// Mask_comp:   false
// Mask_struct: true or false
// C_replace:   false
// accum:       NULL
// A:           scalar
// S:           none

#include "assign/GB_subassign_methods.h"
#include "assign/GB_subassign_dense.h"
#include "include/GB_unused.h"
#include "jitifyer/GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "FactoryKernels/GB_as__include.h"
#endif
#define GB_GENERIC
#define GB_SCALAR_ASSIGN 1
#include "assign/include/GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

GrB_Info GB_subassign_05d
(
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_struct,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix S = NULL ;           // not constructed
    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M

    ASSERT_MATRIX_OK (C, "C for subassign method_05d", GB0) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (GB_IS_FULL (C)) ;

    ASSERT_MATRIX_OK (M, "M for subassign method_05d", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_PENDING (M)) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    // quick return if work has already been done by GB_assign_prep
    if (C->iso) return (GrB_SUCCESS) ;

    const GB_Type_code ccode = C->type->code ;
    const size_t csize = C->type->size ;
    GB_GET_SCALAR ;

    //--------------------------------------------------------------------------
    // Method 05d: C(:,:)<M> = scalar ; no S; C is dense
    //--------------------------------------------------------------------------

    // Time: Optimal:  the method must iterate over all entries in M,
    // and the time is O(nnz(M)).

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

        #define GB_sub05d(cname) GB (_subassign_05d_ ## cname)
        #define GB_WORKER(cname)                                        \
        {                                                               \
            info = GB_sub05d (cname) (C, M, Mask_struct, cwork, Werk) ; \
        }                                                               \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        // The scalar scalar_type is not needed, and there is no accum operator.
        // This method uses cwork = (ctype) scalar, typecasted above, so it
        // works for any scalar type.  As a result, only a test of ccode is
        // required.

        // C<M> = x
        #include "assign/factory/GB_assign_factory.c"
    }
    #endif

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    {
        info = GB_subassign_jit (C,
            /* C_replace: */ false,
            /* I, ni, nI, Ikind, Icolon: */ NULL, 0, 0, GB_ALL, NULL,
            /* J, nj, nJ, Jkind, Jcolon: */ NULL, 0, 0, GB_ALL, NULL,
            M,
            /* Mask_comp: */ false,
            Mask_struct,
            /* accum: */ NULL,
            /* A: */ NULL,
            /* scalar, scalar_type: */ cwork, C->type,
            /* S: */ NULL,
            GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_05d, "subassign_05d",
            Werk) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        #include "generic/GB_generic.h"
        GB_BURBLE_MATRIX (M, "(generic C(:,:)<M>=x assign) ") ;

        // Cx [pC] = cwork
        #undef  GB_COPY_cwork_to_C
        #define GB_COPY_cwork_to_C(Cx, pC, cwork, C_iso) \
            memcpy (Cx + ((pC)*csize), cwork, csize)

        #include "assign/template/GB_subassign_05d_template.c"
        info = GrB_SUCCESS ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    {
        ASSERT_MATRIX_OK (C, "C output for subassign method_05d", GB0) ;
    }
    return (info) ;
}

