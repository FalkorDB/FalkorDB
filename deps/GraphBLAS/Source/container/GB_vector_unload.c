//------------------------------------------------------------------------------
// GB_vector_unload: unload C array from a dense GrB_Vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is guaranteed to take O(1) time and space, if on input V is a
// non-iso vector in the full data format.

// On input, V is a GrB_Vector with nvals(V) == length(V), in any data format.
// That is, all entries of V must be present.  If this condition does not hold,
// the method returns an error code, GrB_INVALID_OBJECT, indicating that the
// vector is not a valid input for this method.

// V is returned as valid full vector of length 0 with no content (V->x ==
// NULL).  It type is not changed.  If on input V was in the full data format,
// then no mallocs/frees are performed.

// If readonly is returned as true, then V was created as a "shallow" vector
// by GxB_Vector_load.  Its numerical content, V->x = (*X), was "shallow" and
// thus treated as readonly by GraphBLAS.  Its allocation/deallocation is the
// responsibility of the user application that created V via GxB_Vector_load.

// On output, *X is a pointer to the numerical contents of V.  If V had length
// zero on input, *X may be returned as a NULL pointer (which is not an error).

// X is not removed from the debug memtable since this method is used
// internally to move data between GraphBLAS objects.  See GxB_Vector_unload,
// which unloads a GrB_Vector and gives the memory space to the user
// application.

#include "GB_container.h"
#define GB_FREE_ALL ;

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
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_RETURN_IF_NULL_OR_FAULTY (V) ;
    GB_RETURN_IF_NULL (type) ;
    GB_RETURN_IF_NULL (X) ;
    GB_RETURN_IF_NULL (n) ;
    GB_RETURN_IF_NULL (X_size) ;
    ASSERT_VECTOR_OK (V, "V to unload", GB0) ;

    //--------------------------------------------------------------------------
    // finish any pending work and ensure V is not iso
    //--------------------------------------------------------------------------

    // This will do nothing (and take O(1) time) if the GrB_Vector V is a
    // component of a Container obtained by unloading a GrB_Matrix or
    // GrB_Vector into the Container.

    if (GB_ANY_PENDING_WORK (V))
    { 
        GB_OK (GB_wait ((GrB_Matrix) V, "V_to_unload", Werk)) ;
    }
    if (!GB_is_dense ((GrB_Matrix) V))
    { 
        // V must be dense with all entries present
        return (GrB_INVALID_OBJECT) ;
    }
    GB_OK (GB_convert_any_to_non_iso ((GrB_Matrix) V, true)) ;
    ASSERT_VECTOR_OK (V, "V ready to unload", GB0) ;

    //--------------------------------------------------------------------------
    // unload the content from V into X
    //--------------------------------------------------------------------------

    (*X) = V->x ;
    (*n) = V->vlen ;
    (*X_size) = V->x_size ;
    (*type) = V->type ;
    (*readonly) = V->x_shallow ;
    V->x = NULL ;
    V->x_size = 0 ;
    V->x_shallow = false ;

    //--------------------------------------------------------------------------
    // clear prior content of V, making V a dense GrB_Vector of length 0
    //--------------------------------------------------------------------------

    GB_vector_reset (V) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_VECTOR_OK (V, "V unloaded", GB0) ;
    ASSERT (GB_IS_FULL (V)) ;
    return (GrB_SUCCESS) ;
}

