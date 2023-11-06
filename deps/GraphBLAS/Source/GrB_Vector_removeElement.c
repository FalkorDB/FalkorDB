//------------------------------------------------------------------------------
// GrB_Vector_removeElement: remove a single entry from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Removes a single entry, V (i), from the vector V.

#include "GB.h"

#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// GB_removeElement: remove V(i) if it exists
//------------------------------------------------------------------------------

static inline bool GB_removeElement     // returns true if found
(
    GrB_Vector V,
    GrB_Index i
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_FULL (V)) ;

    //--------------------------------------------------------------------------
    // remove V(i)
    //--------------------------------------------------------------------------

    if (GB_IS_BITMAP (V))
    {

        //----------------------------------------------------------------------
        // V is bitmap
        //----------------------------------------------------------------------

        int8_t *restrict Vb = V->b ;
        int8_t vb = Vb [i] ;
        if (vb != 0)
        { 
            // V(i) is present; remove it
            Vb [i] = 0 ;
            V->nvals-- ;
        }
        // V(i) is always found, whether present or not
        return (true) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // V is sparse
        //----------------------------------------------------------------------

        const int64_t *restrict Vp = V->p ;
        const int64_t *restrict Vi = V->i ;
        bool found ;

        // look in V(:)
        int64_t pleft = 0 ;
        int64_t pright = Vp [1] ;
        int64_t vnz = pright ;

        bool is_zombie ;
        if (vnz == V->vlen)
        { 
            // V(:) is as-if-full so no binary search is needed to find V(i)
            pleft = i ;
            ASSERT (GB_UNFLIP (Vi [pleft]) == i) ;
            found = true ;
            is_zombie = GB_IS_ZOMBIE (Vi [pleft]) ;
        }
        else
        { 
            // binary search for V(i): time is O(log(vnz))
            int64_t nzombies = V->nzombies ;
            pright-- ;
            GB_BINARY_SEARCH_ZOMBIE (i, Vi, pleft, pright, found,
                nzombies, is_zombie) ;
        }

        // remove the entry
        if (found && !is_zombie)
        { 
            // V(i) becomes a zombie
            V->i [pleft] = GB_FLIP (i) ;
            V->nzombies++ ;
        }
        return (found) ;
    }
}

//------------------------------------------------------------------------------
// GB_Vector_removeElement: remove a single entry from a vector
//------------------------------------------------------------------------------

GrB_Info GB_Vector_removeElement
(
    GrB_Vector V,               // vector to remove entry from
    GrB_Index i,                // index
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // if V is jumbled, wait on the vector first.  If full, convert to nonfull
    //--------------------------------------------------------------------------

    if (V->jumbled || GB_IS_FULL (V))
    {
        GrB_Info info ;
        if (GB_IS_FULL (V))
        { 
            // convert V from full to sparse
            GB_OK (GB_convert_to_nonfull ((GrB_Matrix) V, Werk)) ;
        }
        else
        { 
            // V is sparse and jumbled
            GB_OK (GB_wait ((GrB_Matrix) V, "v (removeElement:jumbled",
                Werk)) ;
        }
        ASSERT (!GB_IS_FULL (V)) ;
        ASSERT (!GB_ZOMBIES (V)) ;
        ASSERT (!GB_JUMBLED (V)) ;
        ASSERT (!GB_PENDING (V)) ;
        // remove the entry
        return (GB_Vector_removeElement (V, i, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // V is not jumbled and not full; it may have zombies and pending tuples
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_FULL (V)) ;
    ASSERT (GB_ZOMBIES_OK (V)) ;
    ASSERT (!GB_JUMBLED (V)) ;
    ASSERT (GB_PENDING_OK (V)) ;

    // check index
    if (i >= V->vlen)
    { 
        GB_ERROR (GrB_INVALID_INDEX, "Row index "
            GBu " out of range; must be < " GBd, i, V->vlen) ;
    }

    // if V is sparse, it may have pending tuples
    bool V_is_pending = GB_PENDING (V) ; 
    if (GB_nnz ((GrB_Matrix) V) == 0 && !V_is_pending)
    { 
        // quick return
        return (GrB_SUCCESS) ;
    }

    // remove the entry
    if (GB_removeElement (V, i))
    { 
        // found it; no need to assemble pending tuples
        return (GrB_SUCCESS) ;
    }

    // assemble any pending tuples; zombies are OK
    if (V_is_pending)
    { 
        GrB_Info info ;
        GB_OK (GB_wait ((GrB_Matrix) V, "v (removeElement:pending tuples)",
            Werk)) ;
        ASSERT (!GB_ZOMBIES (V)) ;
        ASSERT (!GB_JUMBLED (V)) ;
        ASSERT (!GB_PENDING (V)) ;
        // look again; remove the entry if it was a pending tuple
        GB_removeElement (V, i) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_removeElement: remove a single entry from a vector
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_removeElement
(
    GrB_Vector V,               // vector to remove entry from
    GrB_Index i                 // index
)
{
    GB_WHERE (V, "GrB_Vector_removeElement (v, i)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (V) ;
    ASSERT (GB_VECTOR_OK (V)) ;
    return (GB_Vector_removeElement (V, i, Werk)) ;
}

