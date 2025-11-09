//------------------------------------------------------------------------------
// GxB_unload_Vector_into_Container: unload a GrB_Vector into a Container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// V is returned as a length-0 vector in full data format, with no content.

#include "GB_container.h"
#define GB_FREE_ALL ;

GrB_Info GxB_unload_Vector_into_Container   // GrB_Vector -> GxB_Container
(
    GrB_Vector V,               // vector to unload into the Container
    GxB_Container Container,    // Container to hold the contents of V
    const GrB_Descriptor desc   // currently unused
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (V) ;
    GB_RETURN_IF_NULL (Container) ;
    GB_WHERE_1 (V, "GxB_Vector_unload_into_Container") ;
    ASSERT_VECTOR_OK (V, "Vector to unload into container", GB0) ;

    //--------------------------------------------------------------------------
    // unload the vector
    //--------------------------------------------------------------------------

    GB_OK (GB_unload_into_container ((GrB_Matrix) V, Container, Werk)) ;
    V->vdim = 1 ;
    V->nvec = 1 ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_VECTOR_OK (V, "Vector unloaded", GB0) ;
    return (GrB_SUCCESS) ;
}

