//------------------------------------------------------------------------------
// GB_cast_matrix: copy or typecast the values from A into C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The values of C must already be allocated, of size large enough to hold
// the values of A.  The pattern of C must match the pattern of A, but the
// pattern is not accessed (except to compute GB_nnz_held (A)).

// Note that A may contain zombies, or entries not in the bitmap pattern of A
// if A is bitmap, and the values of these entries might be uninitialized
// values in A->x.  All entries are typecasted or memcpy'ed from A->x to C->x,
// including zombies, non-entries, and live entries alike.  valgrind may
// complain about typecasting these uninitialized values, but these warnings
// are false positives.

#include "GB.h"
#define GB_FREE_ALL ;

GrB_Info GB_cast_matrix     // copy or typecast the values from A into C
(
    GrB_Matrix C,
    GrB_Matrix A
)
{

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GrB_Info info ;
    const int64_t anz = GB_nnz_held (A) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;
    ASSERT (GB_IMPLIES (anz > 1, A->iso == C->iso)) ;
    if (anz == 0)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // copy or typecast from A into C
    //--------------------------------------------------------------------------

    GB_void *Cx = (GB_void *) C->x ;
    GB_void *Ax = (GB_void *) A->x ;
    if (C->type == A->type)
    {

        //----------------------------------------------------------------------
        // copy A->x into C->x
        //----------------------------------------------------------------------

        if (A->iso)
        { 
            // A iso, ctype == atype
            memcpy (Cx, Ax, C->type->size) ;
        }
        else
        { 
            // copy all the values, no typecast
            GB_memcpy (Cx, Ax, anz * C->type->size, nthreads) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // typecast Ax into C->x
        //----------------------------------------------------------------------

        if (A->iso)
        { 
            // Cx [0] = (ctype) Ax [0]
            GB_unop_iso (Cx, C->type, GB_ISO_A, NULL, A, NULL) ;
        }
        else
        { 
            // typecast all the values from A to Cx
            ASSERT (GB_IMPLIES (anz > 0, Cx != NULL)) ;
            GB_OK (GB_cast_array (Cx, C->type->code, A, nthreads)) ;
        }
    }
    return (GrB_SUCCESS) ;
}

