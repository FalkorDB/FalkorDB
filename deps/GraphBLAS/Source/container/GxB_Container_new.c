//------------------------------------------------------------------------------
// GxB_Container_new: create a new Container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_container.h"
#define GB_FREE_ALL GxB_Container_free (Container) ;

//------------------------------------------------------------------------------
// GxB_Container_new
//------------------------------------------------------------------------------

GrB_Info GxB_Container_new
(
    GxB_Container *Container
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (Container) ;
    (*Container) = NULL ;

    //--------------------------------------------------------------------------
    // allocate the new Container
    //--------------------------------------------------------------------------

    size_t header_size ;
    (*Container) = GB_CALLOC_MEMORY (1, sizeof (struct GxB_Container_struct),
        &header_size) ;
    if (*Container == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    // clear the Container scalars
    (*Container)->nrows = 0 ;
    (*Container)->ncols = 0 ;
    (*Container)->nrows_nonempty = -1 ;
    (*Container)->ncols_nonempty = -1 ;
    (*Container)->nvals = 0 ;
    (*Container)->format = GxB_FULL ;
    (*Container)->orientation = GrB_ROWMAJOR ;
    (*Container)->iso = false ;
    (*Container)->jumbled = false ;

    //--------------------------------------------------------------------------
    // allocate the p, h, b, i and x components
    //--------------------------------------------------------------------------

    GB_OK (GB_container_component_new (&((*Container)->p), GrB_UINT32)) ;
    GB_OK (GB_container_component_new (&((*Container)->h), GrB_INT32)) ;
    GB_OK (GB_container_component_new (&((*Container)->b), GrB_INT8)) ;
    GB_OK (GB_container_component_new (&((*Container)->i), GrB_INT32)) ;
    GB_OK (GB_container_component_new (&((*Container)->x), GrB_BOOL)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

