//------------------------------------------------------------------------------
// GrB_Matrix_removeElement: remove a single entry from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Removes a single entry, C (row,col), from the matrix C.

#include "GB.h"

#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// GB_removeElement: remove C(i,j) if it exists
//------------------------------------------------------------------------------

static inline bool GB_removeElement     // return true if found
(
    GrB_Matrix C,
    GrB_Index i,
    GrB_Index j
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_FULL (C)) ;
    int64_t cvlen = C->vlen ;

    //--------------------------------------------------------------------------
    // remove C(i,j)
    //--------------------------------------------------------------------------

    if (GB_IS_BITMAP (C))
    {

        //----------------------------------------------------------------------
        // C is bitmap
        //----------------------------------------------------------------------

        int8_t *restrict Cb = C->b ;
        int64_t p = i + j * cvlen ;
        int8_t cb = Cb [p] ;
        if (cb != 0)
        { 
            // C(i,j) is present; remove it
            Cb [p] = 0 ;
            C->nvals-- ;
        }
        // C(i,j) is always found, whether present or not
        return (true) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C is sparse or hypersparse
        //----------------------------------------------------------------------

        const int64_t *restrict Cp = C->p ;
        const int64_t *restrict Ci = C->i ;
        const int64_t *restrict Ch = C->h ;
        bool found ;
        int64_t pC_start, pC_end ;

        if (Ch != NULL)
        {

            //------------------------------------------------------------------
            // C is hypersparse: look for j in hyperlist C->h [0 ... C->nvec-1]
            //------------------------------------------------------------------

            const int64_t *restrict C_Yp = (C->Y == NULL) ? NULL : C->Y->p ;
            const int64_t *restrict C_Yi = (C->Y == NULL) ? NULL : C->Y->i ;
            const int64_t *restrict C_Yx = (C->Y == NULL) ? NULL : C->Y->x ;
            const int64_t C_hash_bits = (C->Y == NULL) ? 0 : (C->Y->vdim - 1) ;
            const int64_t cnvec = C->nvec ;
            int64_t k = GB_hyper_hash_lookup (Ch, cnvec, Cp, C_Yp, C_Yi, C_Yx,
                C_hash_bits, j, &pC_start, &pC_end) ;
            found = (k >= 0) ;
            if (!found)
            { 
                // vector j is empty
                return (false) ;
            }
            ASSERT (j == Ch [k]) ;

        }
        else
        { 

            //------------------------------------------------------------------
            // C is sparse, C(:,j) is the jth vector of C
            //------------------------------------------------------------------

            pC_start = Cp [j] ;
            pC_end   = Cp [j+1] ;
        }

        // look in C(:,k), the kth vector of C
        int64_t pleft = pC_start ;
        int64_t pright = pC_end-1 ;
        int64_t cknz = pC_end - pC_start ;

        bool is_zombie ;
        if (cknz == cvlen)
        { 
            // C(:,k) is as-if-full so no binary search needed to find C(i,k)
            pleft = pleft + i ;
            ASSERT (GB_UNZOMBIE (Ci [pleft]) == i) ;
            found = true ;
            is_zombie = GB_IS_ZOMBIE (Ci [pleft]) ;
        }
        else
        { 
            // binary search for C(i,k): time is O(log(cknz))
            int64_t nzombies = C->nzombies ;
            GB_BINARY_SEARCH_ZOMBIE (i, Ci, pleft, pright, found,
                nzombies, is_zombie) ;
        }

        // remove the entry
        if (found && !is_zombie)
        { 
            // C(i,j) becomes a zombie
            C->i [pleft] = GB_ZOMBIE (i) ;
            C->nzombies++ ;
        }
        return (found) ;
    }
}

//------------------------------------------------------------------------------
// GB_Matrix_removeElement: remove a single entry from a matrix
//------------------------------------------------------------------------------

GrB_Info GB_Matrix_removeElement
(
    GrB_Matrix C,               // matrix to remove entry from
    GrB_Index row,              // row index
    GrB_Index col,              // column index
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // if C is jumbled, wait on the matrix first.  If full, convert to nonfull
    //--------------------------------------------------------------------------

    if (C->jumbled || GB_IS_FULL (C))
    {
        GrB_Info info ;
        if (GB_IS_FULL (C))
        { 
            // convert C from full to sparse
            GB_OK (GB_convert_to_nonfull (C, Werk)) ;
        }
        else
        { 
            // C is sparse or hypersparse, and jumbled
            GB_OK (GB_wait (C, "C (removeElement:jumbled)", Werk)) ;
        }
        ASSERT (!GB_IS_FULL (C)) ;
        ASSERT (!GB_ZOMBIES (C)) ;
        ASSERT (!GB_JUMBLED (C)) ;
        ASSERT (!GB_PENDING (C)) ;
        // remove the entry
        return (GB_Matrix_removeElement (C, row, col, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // C is not jumbled and not full; it may have zombies and pending tuples
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_FULL (C)) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (GB_PENDING_OK (C)) ;

    // look for index i in vector j
    int64_t i, j, nrows, ncols ;
    if (C->is_csc)
    { 
        // C is stored by column
        i = row ;
        j = col ;
        nrows = C->vlen ;
        ncols = C->vdim ;
    }
    else
    { 
        // C is stored by row
        i = col ;
        j = row ;
        nrows = C->vdim ;
        ncols = C->vlen ;
    }

    // check row and column indices
    if (row >= nrows)
    { 
        GB_ERROR (GrB_INVALID_INDEX, "Row index "
            GBu " out of range; must be < " GBd, row, nrows) ;
    }
    if (col >= ncols)
    { 
        GB_ERROR (GrB_INVALID_INDEX, "Column index "
            GBu " out of range; must be < " GBd, col, ncols) ;
    }

    // if C is sparse or hyper, it may have pending tuples
    bool C_is_pending = GB_PENDING (C) ;
    if (GB_nnz (C) == 0 && !C_is_pending)
    { 
        // quick return
        return (GrB_SUCCESS) ;
    }

    // remove the entry
    if (GB_removeElement (C, i, j))
    { 
        // found it; no need to assemble pending tuples
        return (GrB_SUCCESS) ;
    }

    // assemble any pending tuples; zombies are OK
    if (C_is_pending)
    { 
        GrB_Info info ;
        GB_OK (GB_wait (C, "C (removeElement:pending tuples)", Werk)) ;
        ASSERT (!GB_ZOMBIES (C)) ;
        ASSERT (!GB_JUMBLED (C)) ;
        ASSERT (!GB_PENDING (C)) ;
        // look again; remove the entry if it was a pending tuple
        GB_removeElement (C, i, j) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_removeElement: remove a single entry from a matrix
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_removeElement
(
    GrB_Matrix C,               // matrix to remove entry from
    GrB_Index row,              // row index
    GrB_Index col               // column index
)
{ 
    GB_WHERE (C, "GrB_Matrix_removeElement (C, row, col)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    return (GB_Matrix_removeElement (C, row, col, Werk)) ;
}

