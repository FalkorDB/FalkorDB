//------------------------------------------------------------------------------
// GB_hyper_hash_lookup_template: find k so that j == Ah [k], using hyper_hash
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Let j = Ah [k]
// k = A->Y (j, hash(j)), if present, or k=-1 if not found.

GB_STATIC_INLINE int64_t GB_hyper_hash_lookup_T // k if j==Ah[k]; -1 not found
(
    // inputs, not modified:
    const GB_JTYPE *restrict Ah,    // A->h [0..A->nvec-1]: list of vectors
    const int64_t anvec,
    const GB_PTYPE *restrict Ap,    // A->p [0..A->nvec]: pointers to vectors
    const GB_JTYPE *restrict A_Yp,  // A->Y->p
    const GB_JTYPE *restrict A_Yi,  // A->Y->i
    const GB_JTYPE *restrict A_Yx,  // A->Y->x
    const uint64_t hash_bits,       // A->Y->vdim-1, which is hash table size-1
    const int64_t j,                // find j in Ah [0..anvec-1], using A->Y
    // outputs:
    int64_t *restrict pstart,       // start of vector: Ap [k]
    int64_t *restrict pend          // end of vector: Ap [k+1]
)
{

    bool found = false ;
    int64_t k ;

    if (A_Yp == NULL)
    { 

        //----------------------------------------------------------------------
        // no hyper_hash constructed
        //----------------------------------------------------------------------

        // the hyper_hash is disabled.  Quick lookup for j == Ah [j].
        if (j < anvec && Ah [j] == j)
        { 
            // found j == Ah [j], so return k = j
            k = j ;
            found = true ;
        }

        // binary search of Ah [0...A->nvec-1] for the value j
        if (!found)
        {
            k = 0 ;
            int64_t pright = anvec - 1 ;
            found = GB_binary_search_T (j, Ah, &k, &pright) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // using the hyper_hash
        //----------------------------------------------------------------------

        // determine the hash bucket that would contain vector j
        const int64_t jhash = GB_HASHF2 (j, hash_bits) ;

        //----------------------------------------------------------------------
        // search for j in jhash bucket: A_Yi [A_Yp [jhash] : A_Yp [jhash+1]-1]
        //----------------------------------------------------------------------

        const int64_t ypstart = A_Yp [jhash] ;
        const int64_t ypend = A_Yp [jhash+1] ;
        k = -1 ;
        if ((ypend - ypstart) > 256)
        {
            // The hash bucket jhash has over 256 entries, which is a very high
            // number of collisions.  The load factor of the hash table ranges
            // from 2 to 4.  Do a binary search as a fallback.
            int64_t p = ypstart ;
            int64_t pright = ypend - 1 ;
            found = GB_binary_search_T (j, A_Yi, &p, &pright) ;
            if (found)
            { 
                k = A_Yx [p] ;
            }
        }
        else
        {
            // Linear-time search for j in the jhash bucket.
            for (int64_t p = ypstart ; p < ypend ; p++)
            {
                if (j == A_Yi [p])
                { 
                    // found: j = Ah [k] where k is given by k = A_Yx [p]
                    k = A_Yx [p] ;
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

#undef GB_PTYPE
#undef GB_JTYPE
#undef GB_hyper_hash_lookup_T
#undef GB_binary_search_T
