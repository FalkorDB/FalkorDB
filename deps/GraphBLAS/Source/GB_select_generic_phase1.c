//------------------------------------------------------------------------------
// GB_select_generic_phase1.c: count entries for C=select(A,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse, hypersparse, or full, and the op is not positional.
// C is sparse or hypersparse.

#include "GB_select.h"
#include "GB_ek_slice.h"

GrB_Info GB_select_generic_phase1
(
    int64_t *restrict Cp,
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_OPCODE_IS_POSITIONAL (opcode)) ;
    ASSERT (opcode != GB_NONZOMBIE_idxunop_code) ;

    //--------------------------------------------------------------------------
    // phase1: generic entry selector
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
            #include "GB_select_entry_phase1_template.c"

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
            #include "GB_select_entry_phase1_template.c"

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
            #include "GB_select_entry_phase1_template.c"

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
            #include "GB_select_entry_phase1_template.c"

        }
    }

    return (GrB_SUCCESS) ;
}

