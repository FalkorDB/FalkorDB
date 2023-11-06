//------------------------------------------------------------------------------
// GB_Vector_extractElement: x = V(i)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.

// Extract the value of single scalar, x = V(i), typecasting from the
// type of V to the type of x, as needed.

// Returns GrB_SUCCESS if V(i) is present, and sets x to its value.
// Returns GrB_NO_VALUE if V(i) is not present, and x is unmodified.

// This template constructs GrB_Vector_extractElement_[TYPE], for each of the
// 13 built-in types, and the _UDT method for all user-defined types.
// It also constructs GxB_Vector_isStoredElement.

// FUTURE: tolerate zombies

GrB_Info GB_EXTRACT_ELEMENT     // extract a single entry, x = V(i)
(
    #ifdef GB_XTYPE
    GB_XTYPE *x,                // scalar to extract, not modified if not found
    #endif
    const GrB_Vector V,         // vector to extract a scalar from
    GrB_Index i                 // index
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (V) ;
    #ifdef GB_XTYPE
    GB_RETURN_IF_NULL (x) ;
    #endif

    // delete any lingering zombies, assemble any pending tuples, and unjumble
    if (GB_ANY_PENDING_WORK (V))
    { 
        GrB_Info info ;
        GB_WHERE1 (GB_WHERE_STRING) ;
        GB_BURBLE_START ("GrB_Vector_extractElement") ;
        GB_OK (GB_wait ((GrB_Matrix) V, "v", Werk)) ;
        GB_BURBLE_END ;
    }

    ASSERT (!GB_ANY_PENDING_WORK (V)) ;

    // check index
    if (i >= V->vlen)
    { 
        return (GrB_INVALID_INDEX) ;
    }

    //--------------------------------------------------------------------------
    // find the entry V(i)
    //--------------------------------------------------------------------------

    int64_t pleft ;
    bool found ;
    const int64_t *restrict Vp = V->p ;

    if (Vp != NULL)
    { 
        // V is sparse
        pleft = 0 ;
        int64_t pright = Vp [1] - 1 ;
        // Time taken for this step is at most O(log(nnz(V))).
        const int64_t *restrict Vi = V->i ;
        GB_BINARY_SEARCH (i, Vi, pleft, pright, found) ;
    }
    else
    {
        // V is bitmap or full
        pleft = i ;
        const int8_t *restrict Vb = V->b ;
        if (Vb != NULL)
        { 
            // V is bitmap
            found = (Vb [pleft] == 1) ;
        }
        else
        { 
            // V is full
            found = true ;
        }
    }

    //--------------------------------------------------------------------------
    // extract the element
    //--------------------------------------------------------------------------

    if (found)
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
        // TODO: do not flush if extracting to GrB_Scalar
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

