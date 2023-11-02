//------------------------------------------------------------------------------
// GB_select_generic_phase2.c: C=select(A,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse, hypersparse, or full, and the op is not positional.
// C is sparse or hypersparse.

#include "GB_select.h"
#include "GB_ek_slice.h"

GrB_Info GB_select_generic_phase2
(
    int64_t *restrict Ci,
    GB_void *restrict Cx,                   // NULL if C is iso-valued
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
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

    // The op is either valued, user-defined, or nonzombie.  If it is the
    // nonzombie op, then A is not iso.  For the VALUEEQ* operators, C is
    // always iso even if A is not iso.  In that case, Cx is passed here as
    // NULL.

    GB_Opcode opcode = op->opcode ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_OPCODE_IS_POSITIONAL (opcode)) ;
    ASSERT (!(A->iso) || opcode == GB_USER_idxunop_code) ;
    ASSERT ((opcode >= GB_VALUENE_idxunop_code &&
             opcode <= GB_VALUELE_idxunop_code)
         || (opcode == GB_NONZOMBIE_idxunop_code && !(A->iso))
         || (opcode == GB_USER_idxunop_code)) ;

    //--------------------------------------------------------------------------
    // phase2: generic entry selector
    //--------------------------------------------------------------------------

    // op->xtype is NULL for GxB_NONZOMBIE
    ASSERT_TYPE_OK_OR_NULL (op->xtype, "op->xtype (OK if NULL)", GB0) ;
    GB_Type_code zcode = op->ztype->code ;
    GB_Type_code xcode = (op->xtype == NULL) ? 0 : op->xtype->code ;
    GB_Type_code acode = A->type->code ;
    size_t zsize = op->ztype->size ;
    size_t xsize = (op->xtype == NULL) ? 0 : op->xtype->size ;
    size_t asize = A->type->size ;
    GxB_index_unary_function fkeep = op->idxunop_function ;
    GB_cast_function cast_Z_to_bool, cast_A_to_X ;

    #define GB_GENERIC
    #define GB_ENTRY_SELECTOR
    #define GB_A_TYPE GB_void
    #include "GB_select_shared_definitions.h"

    // GB_ISO_SELECT is always #defined'd as 1 here, even though C can be iso.
    // The case when C is iso is handled by the GB_SELECT_ENTRY macro itself.

    if (A->iso)
    {

        //----------------------------------------------------------------------
        // A is iso
        //----------------------------------------------------------------------

        // Cx [pC] = Ax [pA], no typecast
        #undef  GB_SELECT_ENTRY
        #define GB_SELECT_ENTRY(Cx,pC,Ax,pA)

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
            #include "GB_select_phase2.c"

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
            #include "GB_select_phase2.c"

        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A is non-iso
        //----------------------------------------------------------------------

        // Cx [pC] = Ax [pA], no typecast
        // If Cx is NULL then C is iso-valued.
        #undef  GB_SELECT_ENTRY
        #define GB_SELECT_ENTRY(Cx,pC,Ax,pA)                                \
        if (Cx != NULL)                                                     \
        {                                                                   \
            memcpy (Cx +((pC)*asize), Ax +((pA)*asize), asize) ;            \
        }

        if (opcode == GB_NONZOMBIE_idxunop_code)
        { 

            //------------------------------------------------------------------
            // nonzombie selector when A is not iso
            //------------------------------------------------------------------

            #undef  GB_TEST_VALUE_OF_ENTRY
            #define GB_TEST_VALUE_OF_ENTRY(keep,p) bool keep = (i >= 0)
            #include "GB_select_phase2.c"

        }
        else if (op->ztype == GrB_BOOL && op->xtype == A->type)
        { 

            //------------------------------------------------------------------
            // A is non-iso; no typecasting is required
            //------------------------------------------------------------------

            #undef  GB_TEST_VALUE_OF_ENTRY
            #define GB_TEST_VALUE_OF_ENTRY(keep,p)                          \
                bool keep ;                                                 \
                fkeep (&keep, Ax +(p)*asize,                                \
                    flipij ? j : i, flipij ? i : j, ythunk) ;
            #include "GB_select_phase2.c"

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
            #include "GB_select_phase2.c"

        }
    }

    return (GrB_SUCCESS) ;
}

