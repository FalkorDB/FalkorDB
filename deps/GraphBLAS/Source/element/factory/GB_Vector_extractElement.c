//------------------------------------------------------------------------------
// GB_Vector_extractElement: x = V(i)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extract the value of single scalar, x = V(i), typecasting from the
// type of V to the type of x, as needed.

// Returns GrB_SUCCESS if V(i) is present, and sets x to its value.
// Returns GrB_NO_VALUE if V(i) is not present, and x is unmodified.

// This template constructs GrB_Vector_extractElement_[TYPE], for each of the
// 13 built-in types, and the _UDT method for all user-defined types.
// It also constructs GxB_Vector_isStoredElement.

// The search strategy has been revised in GraphBLAS 10.4.0:  Do not wait
// unless jumbled.  First try to find the element.  If found (live or zombie),
// no need to wait.  If not found and pending tuples exist, wait and then
// search for it again.  Prior to this version, any call to this method would
// always finish any pending work.

GrB_Info GB_EXTRACT_ELEMENT     // extract a single entry, x = V(i)
(
    #ifdef GB_XTYPE
    GB_XTYPE *x,                // scalar to extract, not modified if not found
    #endif
    const GrB_Vector V,         // vector to extract a scalar from
    uint64_t i                  // index
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_RETURN_IF_NULL_OR_INVALID (V) ;
    #ifdef GB_XTYPE
    GB_RETURN_IF_NULL (x) ;
    #endif

    if (V->jumbled)
    { 
        GB_WHERE_1 (V, GB_WHERE_STRING) ;
        GB_BURBLE_START ("GrB_Vector_extractElement") ;
        GB_OK (GB_wait ((GrB_Matrix) V, "v", Werk)) ;
        GB_BURBLE_END ;
    }

    // check index
    if (i >= V->vlen)
    { 
        return (GrB_INVALID_INDEX) ;
    }

    //--------------------------------------------------------------------------
    // find the entry V(i)
    //--------------------------------------------------------------------------

    int64_t pleft ;
    bool found, is_zombie ;
    GB_Vector_find_entry (&pleft, &found, &is_zombie, V, i) ;

    //--------------------------------------------------------------------------
    // check again if not found and the vector has pending tuples
    //--------------------------------------------------------------------------

    if (!found && V->Pending != NULL)
    { 
        GB_WHERE_1 (V, GB_WHERE_STRING) ;
        GB_BURBLE_START ("GrB_Vector_extractElement") ;
        GB_OK (GB_wait ((GrB_Matrix) V, "v", Werk)) ;
        GB_BURBLE_END ;
        GB_Vector_find_entry (&pleft, &found, &is_zombie, V, i) ;
        ASSERT (!is_zombie) ;   // GB_wait has removed all zombies
    }

    //--------------------------------------------------------------------------
    // extract the element
    //--------------------------------------------------------------------------

    if (found && !is_zombie)
    {
        // entry found
        #ifdef GB_XTYPE
        GB_Type_code vcode = V->type->code ;
        #if !defined ( GB_UDT_EXTRACT )
        if (GB_XCODE == vcode)
        { 
            // copy Vx [pleft] into x, no typecasting, for built-in types only.
            GB_XTYPE *restrict Vx = ((GB_XTYPE *) (V->x)) ;
            (*x) = Vx [V->iso ? 0:pleft] ;
        }
        else
        #endif
        { 
            // typecast the value from Vx [pleft] into x
            if (!GB_code_compatible (GB_XCODE, vcode))
            { 
                // x (GB_XCODE) and V (vcode) must be compatible
                return (GrB_DOMAIN_MISMATCH) ;
            }
            size_t vsize = V->type->size ;
            void *vx = ((GB_void *) V->x) + (V->iso ? 0 : (pleft*vsize)) ;
            GB_cast_scalar (x, GB_XCODE, vx, vcode, vsize) ;
        }
        #pragma omp flush
        #endif
        return (GrB_SUCCESS) ;
    }
    else
    { 
        // entry not found
        return (GrB_NO_VALUE) ;
    }
}

#undef GB_UDT_EXTRACT
#undef GB_EXTRACT_ELEMENT
#undef GB_XTYPE
#undef GB_XCODE

