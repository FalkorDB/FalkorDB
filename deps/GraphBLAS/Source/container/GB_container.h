//------------------------------------------------------------------------------
// GB_container.h: Container methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CONTAINER_H
#define GB_CONTAINER_H

#include "GB.h"

// ensure a Container->component exists and is valid
#define GB_CHECK_CONTAINER_COMPONENT(Container,component,type)               \
    if (Container->component == NULL)                                        \
    {                                                                        \
        GB_OK (GB_container_component_new (&(Container->component), type)) ; \
    }                                                                        \
    GB_RETURN_IF_INVALID (Container->component) ;                            \
    ASSERT_VECTOR_OK (Container->component, "Container component", GB0) ;

#define GB_CHECK_CONTAINER(Container)                           \
    GB_CHECK_CONTAINER_COMPONENT (Container, p, GrB_UINT32) ;   \
    GB_CHECK_CONTAINER_COMPONENT (Container, h, GrB_UINT32) ;   \
    GB_CHECK_CONTAINER_COMPONENT (Container, b, GrB_INT8) ;     \
    GB_CHECK_CONTAINER_COMPONENT (Container, i, GrB_UINT32) ;   \
    GB_CHECK_CONTAINER_COMPONENT (Container, x, GrB_BOOL) ;

void GB_vector_load
(
    // input/output:
    GrB_Vector V,           // vector to load from the C array X
    void **X,               // numerical array to load into V
    // input:
    GrB_Type type,          // type of X
    uint64_t n,             // # of entries in X
    uint64_t X_size,        // size of X in bytes (at least n*(sizeof the type))
    bool readonly           // if true, X is treated as readonly
) ;

GrB_Info GB_vector_unload
(
    // input/output:
    GrB_Vector V,           // vector to unload
    void **X,               // numerical array to unload from V
    // output:
    GrB_Type *type,         // type of X
    uint64_t *n,            // # of entries in X
    uint64_t *X_size,       // size of X in bytes (at least n*(sizeof the type))
    bool *readonly,         // if true, X is treated as readonly
    GB_Werk Werk
) ;

GrB_Info GB_unload_into_container   // GrB_Matrix -> GxB_Container
(
    GrB_Matrix A,               // matrix to unload into the Container
    GxB_Container Container,    // Container to hold the contents of A
    GB_Werk Werk
) ;

GrB_Info GB_load_from_container // GxB_Container -> GrB_Matrix
(
    GrB_Matrix A,               // matrix to load from the Container
    GxB_Container Container,    // Container with contents to load into A
    GB_Werk Werk
) ;

void GB_vector_reset    // clear almost all prior content; making V length 0
(
    GrB_Vector V
) ;

GrB_Info GB_container_component_new
(
    GrB_Vector *component,
    GrB_Type type
) ;

#endif

