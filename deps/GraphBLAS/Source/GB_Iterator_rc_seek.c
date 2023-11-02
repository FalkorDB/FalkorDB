//------------------------------------------------------------------------------
// GB_Iterator_rc_seek: seek a row/col iterator to A(:,j) or to jth vector of A
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Seek a row iterator to A(j,:), a col iterator to A(:,j).  If jth_vector is
// true, seek to the jth vector instead.  For sparse, bitmap, or full matrices,
// this is the same as A(j,:) for a row iterator or A(:,j) for a col iterator.
// The jth_vector parameter only affects how hypersparse matrices are
// traversed.

#include "GB.h"

GrB_Info GB_Iterator_rc_seek
(
    GxB_Iterator iterator,
    GrB_Index j,
    bool jth_vector
)
{

    //--------------------------------------------------------------------------
    // check if the iterator is exhausted
    //--------------------------------------------------------------------------

    if (j >= ((jth_vector) ? iterator->anvec : iterator->avdim))
    { 
        iterator->pstart = 0 ;
        iterator->pend = 0 ;
        iterator->p = 0 ;
        iterator->k = iterator->anvec ;
        return (GxB_EXHAUSTED) ;
    }

    //--------------------------------------------------------------------------
    // attach the iterator to A(:,j)
    //--------------------------------------------------------------------------

    switch (iterator->A_sparsity)
    {
        default: 
        case GxB_SPARSE : 
        { 
            // attach to A(:,j), which is also the jth vector of A
            iterator->pstart = iterator->Ap [j] ;
            iterator->pend = iterator->Ap [j+1] ;
            iterator->p = iterator->pstart ;
            iterator->k = j ;
        }
        break ;

        case GxB_HYPERSPARSE : 
        {
            int64_t k ;
            if (jth_vector)
            { 
                // attach to the jth vector of A; this is much faster than
                // searching Ah for the value j, to attach to A(:,j)
                k = j ;
                iterator->pstart = iterator->Ap [k] ;
                iterator->pend = iterator->Ap [k+1] ;
                iterator->p = iterator->pstart ;
                iterator->k = k ;
            }
            else
            {
                // find k so that j = Ah [k], or if not found, return k as the
                // smallest value so that j < Ah [k]. 
                k = 0 ;
                const int64_t *restrict Ah = iterator->Ah ;
                if (j > 0)
                { 
                    bool found ;
                    int64_t pright = iterator->anvec-1 ;
                    GB_SPLIT_BINARY_SEARCH (j, Ah, k, pright, found) ;
                }
            }
            // If j is found, A(:,j) is the kth vector in the Ah hyperlist.
            // If j is not found, the iterator is placed at the first vector
            // after j in the hyperlist, if this vector exists.
            if (k >= iterator->anvec)
            { 
                // the kth vector does not exist
                iterator->pstart = 0 ;
                iterator->pend = 0 ;
                iterator->p = 0 ;
                iterator->k = iterator->anvec ;
                return (GxB_EXHAUSTED) ;
            }
            else
            { 
                // the kth vector exists
                iterator->pstart = iterator->Ap [k] ;
                iterator->pend = iterator->Ap [k+1] ;
                iterator->p = iterator->pstart ;
                iterator->k = k ;
            }
        }
        break ;

        case GxB_BITMAP : 
        { 
            // attach to A(:,j), which is also the jth vector of A
            iterator->pstart = j * iterator->avlen ;
            iterator->pend = (j+1) * iterator->avlen ;
            iterator->p = iterator->pstart ;
            iterator->k = j ;
            return (GB_Iterator_rc_bitmap_next (iterator)) ;
        }
        break ;

        case GxB_FULL : 
        { 
            // attach to A(:,j), which is also the jth vector of A
            iterator->pstart = j * iterator->avlen ;
            iterator->pend = (j+1) * iterator->avlen ;
            iterator->p = iterator->pstart ;
            iterator->k = j ;
        }
        break ;
    }

    return ((iterator->p >= iterator->pend) ? GrB_NO_VALUE : GrB_SUCCESS) ;
}

