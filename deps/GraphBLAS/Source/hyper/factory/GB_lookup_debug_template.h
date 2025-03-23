//------------------------------------------------------------------------------
// GB_lookup_debug_template: find k where j == Ah [k], no hyper_hash
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// For debugging only.

static inline bool GB_lookup_debug_T // find j = Ah [k]
(
    // input:
    const bool A_is_hyper,          // true if A is hypersparse
    const GB_JTYPE *restrict Ah,    // A->h [0..A->nvec-1]: list of vectors
    const GB_PTYPE *restrict Ap,    // A->p [0..A->nvec  ]: pointers to vectors
    const int64_t avlen,            // A->vlen
    // input/output:
    int64_t *restrict pleft,        // on input: look in A->h [pleft..pright].
                                    // on output: pleft == k if found.
    // input:
    int64_t pright_in,              // normally A->nvec-1, but can be trimmed
    const int64_t j,                // vector to find, as j = Ah [k]
    // output:
    int64_t *restrict pstart,       // start of vector: Ap [k]
    int64_t *restrict pend          // end of vector: Ap [k+1]
)
{
    if (A_is_hyper)
    {
        // binary search of Ah [pleft...pright] for the value j
        bool found ;
        int64_t pright = pright_in ;
        found = GB_binary_search_T (j, Ah, pleft, &pright) ;
        if (found)
        {
            // j appears in the hyperlist at Ah [pleft]
            // k = (*pleft)
            (*pstart) = Ap [(*pleft)] ;     // OK
            (*pend)   = Ap [(*pleft)+1] ;   // OK
        }
        else
        {
            // j does not appear in the hyperlist Ah
            // k = -1
            (*pstart) = -1 ;
            (*pend)   = -1 ;
        }
        return (found) ;
    }
    else
    {
        // A is sparse, bitmap, or full; j always appears
        // k = j
        #define GBP(Ap,k,avlen) ((Ap) ? Ap [k] : ((k) * (avlen)))
        (*pstart) = GBP (Ap, j, avlen) ;
        (*pend)   = GBP (Ap, j+1, avlen) ;
        #undef GBP
        return (true) ;
    }
}

#undef GB_PTYPE
#undef GB_JTYPE
#undef GB_lookup_debug_T
#undef GB_binary_search_T

