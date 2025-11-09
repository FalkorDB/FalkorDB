//------------------------------------------------------------------------------
// GxB_Vector_load: load C array into a dense GrB_Vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is guaranteed to take O(1) time and space.  If V starts as
// dense vector of length 0 with no content (V->x == NULL), then no malloc
// or frees are performed.

// If readonly is true, then V is created as a "shallow" vector.  Its
// numerical content, V->x = (*X), is "shallow" and thus treated as readonly
// by GraphBLAS.  It is not freed if V is freed with GrB_Vector_free.  (*X)
// need not be a malloc'd array at all.  Its allocation/deallocation is the
// responsibility of the user application.

// V is returned as a non-iso vector of length n, in the full data format.

// If handling is GxB_IS_READONLY, *X is returned unchanged.  Otherwise, it is
// returned as NULL to indicate that it has been moved into V.

// The vector V may have readonly components on input; they are simply removed
// from V and not modified.

#include "GB_container.h"

GrB_Info GxB_Vector_load
(
    // input/output:
    GrB_Vector V,           // vector to load from the C array X
    void **X,               // numerical array to load into V
    // input:
    GrB_Type type,          // type of X
    uint64_t n,             // # of entries in X
    uint64_t X_size,        // size of X in bytes (at least n*(sizeof the type))
    int handling,           // GrB_DEFAULT (0): transfer ownership to GraphBLAS
                            // GxB_IS_READONLY: X treated as readonly;
                            //  ownership kept by the user application
    const GrB_Descriptor desc   // currently unused; for future expansion
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (V) ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL (X) ;
    if (n > 0)
    { 
        GB_RETURN_IF_NULL (*X) ;
    }
    if (X_size < n * type->size)
    { 
        // X is too small
        return (GrB_INVALID_VALUE) ;
    }
    ASSERT_VECTOR_OK (V, "V to load (contents mostly ignored)", GB0) ;

    //--------------------------------------------------------------------------
    // clear prior content of V and load X, making V a dense GrB_Vector
    //--------------------------------------------------------------------------

    // V->user_name is preserved; all other content is freed.  get/set controls
    // (hyper_switch, bitmap_switch, [pji]_control, etc) are preserved, except
    // that V->sparsity_control is revised to allow V to become a full vector.

    bool readonly = (handling != GrB_DEFAULT) ;
    if (!readonly)
    { 
        // *X is given to GraphBLAS to be owned by the vector V, so add it to
        // the global debug memtable.
        GB_Global_memtable_add (*X, X_size) ;
    }

    GB_vector_load (V, X, type, n, X_size, readonly) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_VECTOR_OK (V, "V loaded", GB0) ;
    ASSERT (GB_IS_FULL (V)) ;
    return (GrB_SUCCESS) ;
}

