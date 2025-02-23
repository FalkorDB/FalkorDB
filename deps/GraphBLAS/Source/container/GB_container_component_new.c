//------------------------------------------------------------------------------
// GB_container_component_new: create a new component for a Container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_container.h"
#define GB_FREE_ALL ;

GrB_Info GB_container_component_new
(
    GrB_Vector *component,
    GrB_Type type
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (component != NULL) ;

    //--------------------------------------------------------------------------
    // allocate a length-0 full vector and initialize its contents
    //--------------------------------------------------------------------------

    GB_OK (GB_new ((GrB_Matrix *) component,
        type, 0, 1, GB_ph_null, /* is_csc: */ true, GxB_FULL,
        GB_HYPER_SWITCH_DEFAULT, 0, /* pji: */ false, false, false)) ;

    GB_vector_reset (*component) ;

    ASSERT_VECTOR_OK (*component, "new component", GB0) ;
    return (GrB_SUCCESS) ;
}

