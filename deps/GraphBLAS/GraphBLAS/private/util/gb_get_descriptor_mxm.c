//------------------------------------------------------------------------------
// gb_get_descriptor_mxm: convert gb_descriptor to GrB_Descriptor for GrB_mxm
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#undef  FREE_ALL
#define FREE_ALL \
    GrB_Descriptor_free (&desc) ;

GrB_Info gb_get_descriptor_mxm
(
    // output:
    GrB_Descriptor *desc_handle,    // GraphBLAS descriptor
    // input:
    gb_descriptor gbdesc,           // gb_descriptor, pointer to static struct
    const int arena,
    char err [ERRLEN]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Descriptor desc = NULL ;
    ASSERT (desc_handle != NULL) ;
    ASSERT (gbdesc != NULL) ;
    (*desc_handle) = NULL ;

    //--------------------------------------------------------------------------
    // create the GrB_Descriptor
    //--------------------------------------------------------------------------

    if (gbdesc->kind == KIND_GRB)
    { 
        // use the defaults (GrB_mxm may return jumbled result)
        OK (gb_get_descriptor (&desc, gbdesc, arena, err)) ;
    }
    else
    { 
        // tell GrB_mxm to return C unjumbled
        gbdesc->nondefault = true ;     // ensure GrB_Descriptor is allocated
        OK (gb_get_descriptor (&desc, gbdesc, arena, err)) ;
        ASSERT (desc != NULL) ;
        OK (GrB_Descriptor_set_INT32 (desc, true, GxB_SORT)) ;
    }

    (*desc_handle) = desc ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_ALL
#define FREE_ALL

