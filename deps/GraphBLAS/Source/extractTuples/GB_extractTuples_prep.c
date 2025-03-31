//------------------------------------------------------------------------------
// GB_extractTuples_prep: prepare an output GrB_Vector for extractTuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_extractTuples_prep ensures that an output GrB_Vector for I, J, or X is
// dense, of size nvals-by-1, and the right type.

#include "GB.h"
#include "extractTuples/GB_extractTuples.h"
#include "container/GB_container.h"

GrB_Info GB_extractTuples_prep
(
    GrB_Vector V,               // an output vector for I, J, or X
    uint64_t nvals,             // # of values V must hold
    const GrB_Type vtype        // desired type of V
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (V == NULL)
    { 
        // nothing to do; this component is not requested
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // quick return if V already has the right properties
    //--------------------------------------------------------------------------

    uint64_t required_size = nvals * vtype->size ;
    if (GB_IS_FULL (V) && V->nvals == nvals && V->vlen == nvals &&
        V->type == vtype && !(V->x_shallow) && V->x_size >= required_size)
    { 
        // nothing to do; the vector is already in the right format
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // remove V->x and free all other content
    //--------------------------------------------------------------------------

    void *Vx = V->x_shallow ? NULL : V->x ;
    size_t Vx_size = V->x_shallow ? 0 : V->x_size ;
    V->x = NULL ;
    GB_phybix_free ((GrB_Matrix) V) ;

    //--------------------------------------------------------------------------
    // ensure Vx is large enough
    //--------------------------------------------------------------------------

    if (required_size > Vx_size)
    { 
        // If Vx is not large enough, reallocate it.  If Vx was initially not
        // empty, it means the space is growing incrementally, so add 25% extra
        // space for future growth.
        GB_FREE_MEMORY (&Vx, Vx_size) ;
        int64_t n = nvals + ((Vx_size == 0) ? 0 : (nvals / 4)) ;
        Vx = GB_MALLOC_MEMORY (n, vtype->size, &Vx_size) ;
        if (Vx == NULL)
        { 
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // load Vx back, to create V as a dense nvals-by-1 vector of type vtype
    //--------------------------------------------------------------------------

    GB_vector_load (V, &Vx, vtype, nvals, Vx_size, false) ;
    ASSERT_VECTOR_OK (V, "V prepped", GB0) ;
    return (GrB_SUCCESS) ;
}

