//------------------------------------------------------------------------------
// GB_subassign_06d: C(:,:)<A> = A; C is full/bitmap, M and A are aliased
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 06d: C(:,:)<A> = A ; no S, C is dense, M and A are aliased

// M:           present, and aliased to A
// Mask_comp:   false
// Mask_struct: true or false
// C_replace:   false
// accum:       NULL
// A:           matrix, and aliased to M
// S:           none

// C must be bitmap or full.  No entries are deleted and thus no zombies
// are introduced into C.  C can be hypersparse, sparse, bitmap, or full, and
// its sparsity structure does not change.  If C is hypersparse, sparse, or
// full, then the pattern does not change (all entries are present, and this
// does not change), and these cases can all be treated the same (as if full).
// If C is bitmap, new entries can be inserted into the bitmap C->b.

// C and A can have any sparsity structure.

#include "assign/GB_subassign_methods.h"
#include "assign/GB_subassign_dense.h"
#include "jitifyer/GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "FactoryKernels/GB_as__include.h"
#endif
#define GB_GENERIC
#define GB_SCALAR_ASSIGN 0
#include "assign/include/GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

GrB_Info GB_subassign_06d
(
    GrB_Matrix C,
    // input:
    const GrB_Matrix A,
    bool Mask_struct,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix S = NULL ;           // not constructed
    ASSERT_MATRIX_OK (C, "C for subassign method_06d", GB0) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (GB_IS_BITMAP (C) || GB_IS_FULL (C)) ;
    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A
    ASSERT_MATRIX_OK (A, "A for subassign method_06d", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    const GB_Type_code ccode = C->type->code ;

    //--------------------------------------------------------------------------
    // Method 06d: C(:,:)<A> = A ; no S; C is dense, M and A are aliased
    //--------------------------------------------------------------------------

    // Time: Optimal:  the method must iterate over all entries in A,
    // and the time is O(nnz(A)).

    //--------------------------------------------------------------------------
    // C<A> = A for built-in types
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;
    if (C->iso)
    { 

        //----------------------------------------------------------------------
        // via the iso kernel
        //----------------------------------------------------------------------

        // Since C is iso, A must be iso (or effectively iso), which is also
        // the mask M.  An iso mask matrix M is converted into a structural
        // mask by GB_get_mask, and thus Mask_struct must be true if C is iso.

        ASSERT (Mask_struct) ;
        #define GB_ISO_ASSIGN
        #undef  GB_MASK_STRUCT
        #define GB_MASK_STRUCT 1
        #undef  GB_C_ISO
        #define GB_C_ISO 1
        #include "assign/template/GB_subassign_06d_template.c"
        #undef  GB_MASK_STRUCT
        #undef  GB_C_ISO
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

            #define GB_sub06d(cname) GB (_subassign_06d_ ## cname)
            #define GB_WORKER(cname)                                \
            {                                                       \
                info = GB_sub06d(cname) (C, A, Mask_struct, Werk) ; \
            }                                                       \
            break ;

            //------------------------------------------------------------------
            // launch the switch factory
            //------------------------------------------------------------------

            if (C->type == A->type && ccode < GB_UDT_code)
            { 
                // C<A> = A
                #include "assign/factory/GB_assign_factory.c"
            }
        }
        #endif

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_subassign_jit (C,
                /* C_replace: */ false,
                /* I, ni, nI, Ikind, Icolon: */ NULL, 0, 0, GB_ALL, NULL,
                /* J, nj, nJ, Jkind, Jcolon: */ NULL, 0, 0, GB_ALL, NULL,
                /* M and A are aliased: */ A,
                /* Mask_comp: */ false,
                Mask_struct,
                /* accum: */ NULL,
                /* A: */ A,
                /* scalar, scalar_type: */ NULL, NULL,
                /* S: */ NULL,
                GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_06d, "subassign_06d",
                Werk) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            #include "generic/GB_generic.h"
            GB_BURBLE_MATRIX (A, "(generic C(:,:)<A>=A assign) ") ;

            const size_t csize = C->type->size ;
            const size_t asize = A->type->size ;
            const GB_Type_code acode = A->type->code ;
            GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;

            #undef  GB_AX_MASK
            #define GB_AX_MASK(Ax,pA,asize) GB_MCAST (Ax, pA, asize)
            #undef  GB_C_ISO
            #define GB_C_ISO 0
            #undef  GB_MASK_STRUCT
            #define GB_MASK_STRUCT Mask_struct
            #include "assign/template/GB_subassign_06d_template.c"
            info = GrB_SUCCESS ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    {
        ASSERT_MATRIX_OK (C, "C output for subassign method_06d", GB0) ;
    }
    return (info) ;
}

