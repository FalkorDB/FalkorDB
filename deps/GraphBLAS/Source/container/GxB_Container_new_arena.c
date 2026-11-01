//------------------------------------------------------------------------------
// GxB_Container_new_arena: create a new Container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_container.h"
#define GB_FREE_ALL GxB_Container_free (Container) ;

GrB_Info GxB_Container_new_arena
(
    GxB_Container *Container,
    int header_arena,
    int data_arena
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (Container != NULL) ;
    (*Container) = NULL ;
    GB_OK (GB_check_arena (header_arena)) ;
    GB_OK (GB_check_arena (data_arena)) ;

    //--------------------------------------------------------------------------
    // allocate the new Container
    //--------------------------------------------------------------------------

    uint64_t mem = GB_mem (header_arena, 0) ;

    uint64_t header_mem = mem ;
    (*Container) = GB_CALLOC_MEMORY (1, sizeof (struct GxB_Container_struct),
        &header_mem) ;
    if (*Container == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    // Keep track of just the header_arena of the Container struct.
    // Container_mem is just GB_mem (header_arena, sizeof (struct
    // GxB_Container_struct)).  See GxB_Container_free.
    (*Container)->header_arena = header_arena ;

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

    GB_OK (GB_container_component_new (&((*Container)->p), GrB_UINT32,
        header_arena, data_arena)) ;
    GB_OK (GB_container_component_new (&((*Container)->h), GrB_INT32,
        header_arena, data_arena)) ;
    GB_OK (GB_container_component_new (&((*Container)->b), GrB_INT8,
        header_arena, data_arena)) ;
    GB_OK (GB_container_component_new (&((*Container)->i), GrB_INT32,
        header_arena, data_arena)) ;
    GB_OK (GB_container_component_new (&((*Container)->x), GrB_BOOL,
        header_arena, data_arena)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

