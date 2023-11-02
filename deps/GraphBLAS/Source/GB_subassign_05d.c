//------------------------------------------------------------------------------
// GB_subassign_05d: C(:,:)<M> = scalar where C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

// Method 05d: C(:,:)<M> = scalar ; no S, C is dense

// M:           present
// Mask_comp:   false
// Mask_struct: true or false
// C_replace:   false
// accum:       NULL
// A:           scalar
// S:           none

// C can have any sparsity structure, but it must be entirely dense with
// all entries present.

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"
#include "GB_subassign_dense.h"
#include "GB_unused.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_as__include.h"
#endif

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

    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    ASSERT_MATRIX_OK (C, "C for subassign method_05d", GB0) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (GB_IS_FULL (C)) ;

    ASSERT_MATRIX_OK (M, "M for subassign method_05d", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_PENDING (M)) ;

    // quick return if work has already been done by GB_assign_prep
    if (C->iso) return (GrB_SUCCESS) ;

    const GB_Type_code ccode = C->type->code ;
    const size_t csize = C->type->size ;
    GB_GET_SCALAR ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

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
            GB_SUBASSIGN, GB_JIT_KERNEL_SUBASSIGN_05d, "subassign_05d",
            Werk) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        #include "GB_generic.h"
        GB_BURBLE_MATRIX (M, "(generic C(:,:)<M>=x assign) ") ;

        // Cx [pC] = cwork
        #undef  GB_COPY_scalar_to_C
        #define GB_COPY_scalar_to_C(Cx,pC,cwork) \
            memcpy (Cx + ((pC)*csize), cwork, csize)

        #include "GB_subassign_05d_template.c"
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

