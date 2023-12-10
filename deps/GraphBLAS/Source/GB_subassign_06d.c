//------------------------------------------------------------------------------
// GB_subassign_06d: C(:,:)<A> = A; C is full/bitmap, M and A are aliased
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

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

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"
#include "GB_subassign_dense.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_as__include.h"
#endif

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
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

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

    const GB_Type_code ccode = C->type->code ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

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
        #include "GB_subassign_06d_template.c"
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
                switch (ccode)
                {
                    case GB_BOOL_code   : GB_WORKER (_bool  )
                    case GB_INT8_code   : GB_WORKER (_int8  )
                    case GB_INT16_code  : GB_WORKER (_int16 )
                    case GB_INT32_code  : GB_WORKER (_int32 )
                    case GB_INT64_code  : GB_WORKER (_int64 )
                    case GB_UINT8_code  : GB_WORKER (_uint8 )
                    case GB_UINT16_code : GB_WORKER (_uint16)
                    case GB_UINT32_code : GB_WORKER (_uint32)
                    case GB_UINT64_code : GB_WORKER (_uint64)
                    case GB_FP32_code   : GB_WORKER (_fp32  )
                    case GB_FP64_code   : GB_WORKER (_fp64  )
                    case GB_FC32_code   : GB_WORKER (_fc32  )
                    case GB_FC64_code   : GB_WORKER (_fc64  )
                    default: ;
                }
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
                GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_06d, "subassign_06d",
                Werk) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            #include "GB_generic.h"
            GB_BURBLE_MATRIX (A, "(generic C(:,:)<A>=A assign) ") ;

            const size_t csize = C->type->size ;
            const size_t asize = A->type->size ;
            const GB_Type_code acode = A->type->code ;
            GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;

            #define C_iso false
            #undef  GB_AX_MASK
            #define GB_AX_MASK(Ax,pA,asize) GB_MCAST (Ax, pA, asize)

            #include "GB_subassign_06d_template.c"
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

