//------------------------------------------------------------------------------
// GxB_Matrix_Iterator: seek to a specific entry for a matrix iterator 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "slice/include/GB_search_for_vector.h"

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_attach: attach an entry iterator to a matrix
//------------------------------------------------------------------------------

// On input, the iterator must already exist, having been created by
// GxB_Iterator_new.

// GxB_Matrix_Iterator_attach attaches an entry iterator to a matrix.  If the
// iterator is already attached to a matrix, it is detached and then attached
// to the given matrix A.

// The following error conditions are returned:
// GrB_NULL_POINTER:    if the iterator or A are NULL.
// GrB_INVALID_OBJECT:  if the matrix A is invalid.
// GrB_OUT_OF_MEMORY:   if the method runs out of memory.

// If successful, the entry iterator is attached to the matrix, but not to any
// specific entry.  Use GxB_Matrix_Iterator_*seek* to move the iterator to a
// particular entry.

GrB_Info GxB_Matrix_Iterator_attach
(
    GxB_Iterator iterator,
    GrB_Matrix A,
    GrB_Descriptor desc
)
{ 
    return (GB(Iterator_attach)(iterator, A, GxB_NO_FORMAT, desc)) ;
}

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_getpmax: return the range of the iterator
//------------------------------------------------------------------------------

// On input, the entry iterator must be already attached to a matrix via
// GxB_Matrix_Iterator_attach; results are undefined if this condition is not
// met.

// Entries in a matrix are given an index p, ranging from 0 to pmax-1, where
// pmax >= nvals(A).  For sparse, hypersparse, and full matrices, pmax is equal
// to nvals(A).  For an m-by-n bitmap matrix, pmax=m*n, or pmax=0 if the
// matrix has no entries.

uint64_t GxB_Matrix_Iterator_getpmax (GxB_Iterator iterator)
{ 
    return (iterator->pmax) ;
}

//------------------------------------------------------------------------------
// GB_full_position: find the vector containing p for a full/bitmap matrix
//------------------------------------------------------------------------------

static inline GrB_Info GB_full_position (GxB_Iterator iterator)
{ 
    iterator->k = iterator->p / iterator->avlen ;
    iterator->pstart = iterator->k * iterator->avlen ;
    iterator->pend = iterator->pstart + iterator->avlen ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_check_for_end_of_vector: move to next vector if current vector is done
//------------------------------------------------------------------------------

static inline GrB_Info GB_check_for_end_of_vector (GxB_Iterator iterator)
{
    // move to the next vector if p has reached the end of the current vector 
    switch (iterator->A_sparsity)
    {
        default: 
        case GxB_SPARSE : 
        case GxB_HYPERSPARSE : 
        {
            if (iterator->p >= iterator->pend)
            {
                // the kth vector is done; move to the next non-empty vector
                iterator->pstart = iterator->pend ;
                iterator->k++ ;
                if (iterator->Ap32 != NULL)
                {
                    while (iterator->Ap32 [iterator->k+1] == iterator->pend)
                    { 
                        // iterator->k is an empty vector; move to the next one
                        iterator->k++ ;
                        ASSERT (iterator->k < iterator->anvec) ;
                    }
                    // iterator->k is now the next non-empty vector
                    iterator->pend = iterator->Ap32 [iterator->k+1] ;
                }
                else
                {
                    while (iterator->Ap64 [iterator->k+1] == iterator->pend)
                    { 
                        // iterator->k is an empty vector; move to the next one
                        iterator->k++ ;
                        ASSERT (iterator->k < iterator->anvec) ;
                    }
                    // iterator->k is now the next non-empty vector
                    iterator->pend = iterator->Ap64 [iterator->k+1] ;
                }
                return (GrB_SUCCESS) ;
            }
        }
        break ;

        case GxB_BITMAP : 
        {
            for ( ; iterator->p < iterator->pmax ; iterator->p++)
            {
                if (iterator->Ab [iterator->p])
                {
                    // found the next entry; check if it is past the kth vector
                    if (iterator->p >= iterator->pend)
                    { 
                        // find the vector of this entry
                        return (GB_full_position (iterator)) ;
                    }
                    return (GrB_SUCCESS) ;
                }
            }
            return (GxB_EXHAUSTED) ;
        }
        break ;

        case GxB_FULL : 
        {
            if (iterator->p >= iterator->pend)
            { 
                // kth vector is done; move to the next vector
                iterator->k++ ;
                iterator->pstart += iterator->avlen ;
                iterator->pend += iterator->avlen ;
            }
        }
        break ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_next: move an entry iterator to the next entry
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_Iterator_next (GxB_Iterator iterator)
{
    // move to the next entry
    if (++(iterator->p) >= iterator->pmax)
    { 
        // the iterator is exhausted
        iterator->p = iterator->pmax ;
        return (GxB_EXHAUSTED) ;
    }
    else
    { 
        // move to next vector if iterator has reached the end of a vector
        return (GB_check_for_end_of_vector (iterator)) ;
    }
}

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_seek: seek an entry iterator to any entry
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_Iterator_seek
(
    GxB_Iterator iterator,
    uint64_t p_input
)
{
    int64_t p = (int64_t) p_input ;
    if (p >= iterator->pmax)
    { 
        // the iterator is exhausted
        iterator->p = iterator->pmax ;
        return (GxB_EXHAUSTED) ;
    }
    else if (p == 0)
    { 
        // seek to the first entry of the first vector A(:,0)
        iterator->pstart = 0 ;
        iterator->pend = GBp (iterator->Ap, 1, iterator->avlen) ;
        iterator->p = 0 ;
        iterator->k = 0 ;
        // move to the next non-empty vector if A(:,0) is empty
        return (GB_check_for_end_of_vector (iterator)) ;
    }
    else
    {
        // seek to an arbitrary position in the matrix
        iterator->p = p ;
        switch (iterator->A_sparsity)
        {
            default: 
            case GxB_SPARSE : 
            case GxB_HYPERSPARSE : 
            { 
                // find the vector k that contains position p
                if (iterator->Ap32 != NULL)
                {
                    iterator->k = GB_search_for_vector_32 (iterator->Ap32,
                        p, 0, iterator->anvec, iterator->avlen) ;
                    iterator->pstart = iterator->Ap32 [iterator->k] ;
                    iterator->pend   = iterator->Ap32 [iterator->k+1] ;
                }
                else
                {
                    iterator->k = GB_search_for_vector_64 (iterator->Ap64,
                        p, 0, iterator->anvec, iterator->avlen) ;
                    iterator->pstart = iterator->Ap64 [iterator->k] ;
                    iterator->pend   = iterator->Ap64 [iterator->k+1] ;
                }
            }
            break ;
            case GxB_BITMAP : 
            {
                for ( ; iterator->p < iterator->pmax ; iterator->p++)
                {
                    if (iterator->Ab [iterator->p])
                    { 
                        // found next entry; find the vector that contains it
                        return (GB_full_position (iterator)) ;
                    }
                }
                return (GxB_EXHAUSTED) ;
            }
            break ;
            case GxB_FULL : 
            { 
                // find the vector k that contains position p
                return (GB_full_position (iterator)) ;
            }
            break ;
        }
    }
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_getp: get the current position of a matrix iterator
//------------------------------------------------------------------------------

// On input, the entry iterator must be already attached to a matrix via
// GxB_Matrix_Iterator_attach, and the position of the iterator must also have
// been defined by a prior call to GxB_Matrix_Iterator_seek or
// GxB_Matrix_Iterator_next.  Results are undefined if these conditions are not
// met.

uint64_t GxB_Matrix_Iterator_getp (GxB_Iterator iterator)
{ 
    return (iterator->p) ;
}

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_getIndex: get the row and column index of a matrix entry
//------------------------------------------------------------------------------

// On input, the entry iterator must be already attached to a matrix via
// GxB_Matrix_Iterator_attach, and the position of the iterator must also have
// been defined by a prior call to GxB_Matrix_Iterator_seek or
// GxB_Matrix_Iterator_next, with a return value of GrB_SUCCESS.  Results are
// undefined if these conditions are not met.

void GxB_Matrix_Iterator_getIndex
(
    GxB_Iterator iterator,
    uint64_t *row,
    uint64_t *col
)
{
    uint64_t i, j ;

    // get row and column index of current entry, for matrix iterator
    switch (iterator->A_sparsity)
    {
        default:  
        case GxB_SPARSE : 
            i = GB_IGET (iterator->Ai, iterator->p) ;
            j = iterator->k ;
        break ;

        case GxB_HYPERSPARSE : 
            i = GB_IGET (iterator->Ai, iterator->p) ;
            j = GB_IGET (iterator->Ah, iterator->k) ;
        break ;

        case GxB_BITMAP : 
        case GxB_FULL : 
            i = iterator->p - iterator->pstart ;
            j = iterator->k ;
        break ;
    }

    if (iterator->by_col)
    { 
        (*row) = i ;
        (*col) = j ;
    }
    else
    { 
        (*row) = j ;
        (*col) = i ;
    }
}

