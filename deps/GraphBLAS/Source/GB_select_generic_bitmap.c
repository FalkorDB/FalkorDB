//------------------------------------------------------------------------------
// GB_select_generic_bitmap.c: C=select(A,thunk) when C is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is bitmap or full, C is bitmap

#include "GB_select.h"
#include "GB_ek_slice.h"

GrB_Info GB_select_generic_bitmap
(
    int8_t *Cb,
    int64_t *cnvals_handle,
    GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int nthreads
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    ASSERT (!GB_OPCODE_IS_POSITIONAL (opcode)) ;
    ASSERT (!(A->iso) || (opcode == GB_USER_idxunop_code)) ;
    ASSERT ((opcode >= GB_VALUENE_idxunop_code &&
             opcode <= GB_VALUELE_idxunop_code) ||
             (opcode == GB_USER_idxunop_code)) ;

    //--------------------------------------------------------------------------
    // generic entry selector when C is bitmap
    //--------------------------------------------------------------------------

    ASSERT_TYPE_OK (op->xtype, "op->xtype", GB0) ;
    GB_Type_code zcode = op->ztype->code ;
    GB_Type_code xcode = op->xtype->code ;
    GB_Type_code acode = A->type->code ;
    size_t zsize = op->ztype->size ;
    size_t xsize = op->xtype->size ;
    size_t asize = A->type->size ;
    GxB_index_unary_function fkeep = op->idxunop_function ;
    GB_cast_function cast_Z_to_bool, cast_A_to_X ;

    #define GB_GENERIC
    #define GB_ENTRY_SELECTOR
    #define GB_A_TYPE GB_void
    #include "GB_select_shared_definitions.h"

    if (A->iso)
    {

        //----------------------------------------------------------------------
        // A is iso
        //----------------------------------------------------------------------

        // x = (xtype) Ax [0]
        GB_void x [GB_VLA(xsize)] ;
        GB_cast_scalar (x, xcode, A->x, acode, asize) ;

        if (op->ztype == GrB_BOOL)
        { 

            //------------------------------------------------------------------
            // A is iso and z is bool
            //------------------------------------------------------------------

            #define GB_TEST_VALUE_OF_ENTRY(keep,p)                          \
                bool keep ;                                                 \
                fkeep (&keep, x, flipij ? j : i, flipij ? i : j, ythunk) ;

            #include "GB_select_bitmap_template.c"

        }
        else
        { 

            //------------------------------------------------------------------
            // A is iso and z requires typecasting
            //------------------------------------------------------------------

            cast_Z_to_bool = GB_cast_factory (GB_BOOL_code, zcode) ; 

            #undef  GB_TEST_VALUE_OF_ENTRY
            #define GB_TEST_VALUE_OF_ENTRY(keep,p)                          \
                bool keep ;                                                 \
                GB_void z [GB_VLA(zsize)] ;                                 \
                fkeep (z, x, flipij ? j : i, flipij ? i : j, ythunk) ;      \
                cast_Z_to_bool (&keep, z, zsize) ;

            #include "GB_select_bitmap_template.c"
        }

    }
    else
    {

        if (op->ztype == GrB_BOOL && op->xtype == A->type)
        { 

            //------------------------------------------------------------------
            // A is non-iso and no typecasting is required
            //------------------------------------------------------------------

            #undef  GB_TEST_VALUE_OF_ENTRY
            #define GB_TEST_VALUE_OF_ENTRY(keep,p)                          \
                bool keep ;                                                 \
                fkeep (&keep, Ax +(p)*asize,                                \
                    flipij ? j : i, flipij ? i : j, ythunk) ;

            #include "GB_select_bitmap_template.c"

        }
        else
        { 

            //------------------------------------------------------------------
            // A is non-iso and typecasting is required
            //------------------------------------------------------------------

            cast_A_to_X = GB_cast_factory (xcode, acode) ;
            cast_Z_to_bool = GB_cast_factory (GB_BOOL_code, zcode) ; 

            #undef  GB_TEST_VALUE_OF_ENTRY
            #define GB_TEST_VALUE_OF_ENTRY(keep,p)                          \
                bool keep ;                                                 \
                GB_void z [GB_VLA(zsize)] ;                                 \
                GB_void x [GB_VLA(xsize)] ;                                 \
                cast_A_to_X (x, Ax +(p)*asize, asize) ;                     \
                fkeep (z, x, flipij ? j : i, flipij ? i : j, ythunk) ;      \
                cast_Z_to_bool (&keep, z, zsize) ;

            #include "GB_select_bitmap_template.c"

        }

    }
    return (GrB_SUCCESS) ;
}

