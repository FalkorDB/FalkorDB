//------------------------------------------------------------------------------
// GB_jappend.h: definitions of GB_jappend, and GB_jwrapup
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These methods are now only used by GB_wait.

#ifndef GB_JAPPEND_H
#define GB_JAPPEND_H
#include "GB.h"

//------------------------------------------------------------------------------
// GB_jappend:  append a new vector to the end of a matrix
//------------------------------------------------------------------------------

// Append a new vector to the end of a matrix C.

// If C->h != NULL, C is in hypersparse form with
// C->nvec <= C->plen <= C->vdim.  C->h has size C->plen.

// If C->h == NULL, C is in non-hypersparse form with
// C->nvec == C->plen == C->vdim.  C->h is NULL.
// In both cases, C->p has size C->plen+1.

static inline GrB_Info GB_jappend
(
    GrB_Matrix C,           // matrix to append vector j to
    int64_t j,              // new vector to append
    int64_t *jlast,         // last vector appended, -1 if none
    int64_t cnz,            // nnz(C) after adding this vector j
    int64_t *cnz_last,      // nnz(C) before adding this vector j
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL) ;
    ASSERT (!C->p_shallow) ;
    ASSERT (!C->h_shallow) ;
    ASSERT (C->p != NULL) ;
    ASSERT (!GB_IS_FULL (C)) ;

    if (cnz <= (*cnz_last))
    {
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    // one more non-empty vector
    C->nvec_nonempty++ ;

    if (C->h != NULL)
    { 

        //----------------------------------------------------------------------
        // C is hypersparse; make sure space exists in the hyperlist
        //----------------------------------------------------------------------

        ASSERT (C->p [C->nvec] == (*cnz_last)) ;
        ASSERT (C->h != NULL) ;

        // check if space exists
        if (C->nvec == C->plen)
        { 
            // double the size of C->h and C->p
            GrB_Info info ;
            info = GB_hyper_realloc (C, GB_IMIN (C->vdim, 2*(C->plen+1)),
                Werk) ;
            if (info != GrB_SUCCESS)
            { 
                // out of memory
                return (info) ;
            }
        }

        ASSERT (C->nvec >= 0) ;
        ASSERT (C->nvec < C->plen) ;
        ASSERT (C->plen <= C->vdim) ;
        ASSERT (C->p [C->nvec] == (*cnz_last)) ;

        // add j to the hyperlist
        C->h [C->nvec] = j ;

        // mark the end of C(:,j)
        C->p [C->nvec+1] = cnz ;
        C->nvec++ ;                     // one more vector in the hyperlist

    }
    else
    {

        //----------------------------------------------------------------------
        // C is non-hypersparse
        //----------------------------------------------------------------------

        int64_t *restrict Cp = C->p ;

        ASSERT (C->nvec == C->plen && C->plen == C->vdim) ;
        ASSERT (C->h == NULL) ;
        ASSERT (Cp [(*jlast)+1] == (*cnz_last)) ;

        // Even if C is non-hypersparse, the iteration that uses this function
        // may iterate over a hypersparse input matrix, so not every vector j
        // will be traversed.  So when j is seen, the end of vectors jlast+1 to
        // j must logged in Cp.

        for (int64_t jprior = (*jlast)+1 ; jprior < j ; jprior++)
        { 
            // mark the end of C(:,jprior)
            Cp [jprior+1] = (*cnz_last) ;
        }
        // mark the end of C(:,j)
        Cp [j+1] = cnz ;
    }

    // record the last vector added to C
    (*cnz_last) = cnz ;
    (*jlast) = j ;

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_jwrapup:  finish contructing a new matrix
//------------------------------------------------------------------------------

// Log the end of any vectors in C that are not yet terminated.  Nothing
// happens if C is hypersparse (except for setting C->magic).

static inline void GB_jwrapup
(
    GrB_Matrix C,           // matrix to finish
    int64_t jlast,          // last vector appended, -1 if none
    int64_t cnz             // final nnz(C)
)
{

    if (C->h == NULL)
    {

        //----------------------------------------------------------------------
        // C is non-hypersparse
        //----------------------------------------------------------------------

        // log the end of C(:,jlast+1) to C(:,n-1), in case the last vector
        // j=n-1 has not yet been seen, or has been seen but was empty.

        int64_t *restrict Cp = C->p ;
        int64_t j = C->vdim - 1 ;

        for (int64_t jprior = jlast+1 ; jprior <= j ; jprior++)
        { 
            // mark the end of C(:,jprior)
            Cp [jprior+1] = cnz ;
        }
    }

    // C->p and C->h are now valid
    C->nvals = cnz ;
    C->magic = GB_MAGIC ;
}

#endif

