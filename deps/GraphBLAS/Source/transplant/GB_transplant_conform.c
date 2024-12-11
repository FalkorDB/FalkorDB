//------------------------------------------------------------------------------
// GB_transplant_conform: transplant T into C, then conform C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = (type) T, then conform C to its desired sparsity structure.  T is freed.
// All prior content of C is cleared; zombies and pending tuples are abandoned
// in C.  C and T can have any sparsity structure on input.  If T is iso, then
// so is C.

#include "GB.h"

GrB_Info GB_transplant_conform      // transplant and conform sparsity structure
(
    GrB_Matrix C,                   // destination matrix to transplant into
    GrB_Type ctype,                 // type to cast into
    GrB_Matrix *Thandle,            // source matrix
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL) ;
    ASSERT (Thandle != NULL) ;
    ASSERT_MATRIX_OK (*Thandle, "T to transplant into C", GB0) ;
    ASSERT_TYPE_OK (ctype, "ctype for transplant into C", GB0) ;
    ASSERT (GB_ZOMBIES_OK (*Thandle)) ;
    ASSERT (GB_JUMBLED_OK (*Thandle)) ;
    ASSERT (GB_PENDING_OK (*Thandle)) ;

    //--------------------------------------------------------------------------
    // transplant and typecast T into C, and free T
    //--------------------------------------------------------------------------

    GrB_Info info = GB_transplant (C, ctype, Thandle, Werk) ;

    // T is always freed, even if the transplant runs out of memory
    ASSERT (*Thandle == NULL ||
           (*Thandle != NULL && ((*Thandle)->static_header || GBNSTATIC))) ;

    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (info) ;
    }

    ASSERT_MATRIX_OK (C, "C transplanted", GB0) ;

    //--------------------------------------------------------------------------
    // conform C to its desired sparsity structure
    //--------------------------------------------------------------------------

    info = GB_conform (C, Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (info) ;
    }

    ASSERT_MATRIX_OK (C, "C conformed", GB0) ;
    ASSERT (C->nvec_nonempty >= 0) ;
    return (GrB_SUCCESS) ;
}

