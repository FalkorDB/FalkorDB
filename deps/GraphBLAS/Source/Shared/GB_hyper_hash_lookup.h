//------------------------------------------------------------------------------
// GB_hyper_hash_lookup: find k so that j == Ah [k], using the A->Y hyper_hash
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_HYPER_HASH_LOOKUP_H
#define GB_HYPER_HASH_LOOKUP_H

// Let j = Ah [k]
// k = A->Y (j, hash(j)), if present, or k=-1 if not found.

GB_STATIC_INLINE int64_t GB_hyper_hash_lookup  // k if j==Ah[k]; -1 if not found
(
    // new inputs, not modified
    const int64_t *restrict Ah,     // A->h [0..A->nvec-1]: list of vectors
    const int64_t anvec,
    // input, not modified
    const int64_t *restrict Ap,     // A->p [0..A->nvec]: pointers to vectors
    const int64_t *restrict Yp,     // A->Y->p
    const int64_t *restrict Yi,     // A->Y->i
    const int64_t *restrict Yx,     // A->Y->x
    const int64_t hash_bits,        // A->Y->vdim-1, which is hash table size-1
    const int64_t j,                // find j in Ah [0..anvec-1], using A->Y
    // output
    int64_t *restrict pstart,       // start of vector: Ap [k]
    int64_t *restrict pend          // end of vector: Ap [k+1]
)
{

    bool found = false ;
    int64_t k ;

    if (Yp == NULL)
    { 

        //----------------------------------------------------------------------
        // no hyper_hash constructed
        //----------------------------------------------------------------------

        // binary search of Ah [0...A->nvec-1] for the value j
        k = 0 ;
        int64_t pright = anvec - 1 ;
        GB_BINARY_SEARCH (j, Ah, k, pright, found) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // using the hyper_hash
        //----------------------------------------------------------------------

        // determine the hash bucket that would contain vector j
        const int64_t jhash = GB_HASHF2 (j, hash_bits) ;

        //----------------------------------------------------------------------
        // search for j in the jhash bucket: Yi [Yp [jhash] ... Yp [jhash+1]-1]
        //----------------------------------------------------------------------

        const int64_t ypstart = Yp [jhash] ;
        const int64_t ypend = Yp [jhash+1] ;
        k = -1 ;
        if ((ypend - ypstart) > 256)
        {
            // The hash bucket jhash has over 256 entries, which is a very high
            // number of collisions.  The load factor of the hash table ranges
            // from 2 to 4.  Do a binary search as a fallback.
            int64_t p = ypstart ;
            int64_t pright = ypend - 1 ;
            GB_BINARY_SEARCH (j, Yi, p, pright, found) ;
            if (found)
            { 
                k = Yx [p] ;
            }
        }
        else
        {
            // Linear-time search for j in the jhash bucket.
            for (int64_t p = ypstart ; p < ypend ; p++)
            {
                if (j == Yi [p])
                { 
                    // found: j = Ah [k] where k is given by k = Yx [p]
                    k = Yx [p] ;
                    break ;
                }
            }
            found = (k >= 0) ;
        }
    }

    //--------------------------------------------------------------------------
    // if found, return the start and end of A(:,j)
    //--------------------------------------------------------------------------

    if (found)
    { 
        // found: j == Ah [k], get the vector A(:,j)
        (*pstart) = Ap [k] ;
        (*pend  ) = Ap [k+1] ;
    }
    else
    { 
        // not found: j is not in the hyperlist Ah [0..anvec-1]
        k = -1 ;
        (*pstart) = -1 ;
        (*pend  ) = -1 ;
    }
    return (k) ;
}

#endif

