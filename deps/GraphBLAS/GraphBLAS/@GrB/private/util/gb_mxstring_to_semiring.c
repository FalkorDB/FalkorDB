//------------------------------------------------------------------------------
// gb_mxstring_to_semiring: get a GraphBLAS semiring from a built-in string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Semiring gb_mxstring_to_semiring    // return semiring from a string
(
    const mxArray *mxstring,            // built-in string
    const GrB_Type atype,               // type of A
    const GrB_Type btype                // type of B
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (gb_mxarray_is_empty (mxstring))
    { 
        ERROR ("semiring missing") ;
    }

    //--------------------------------------------------------------------------
    // get the string
    //--------------------------------------------------------------------------

    #define LEN 256
    char semiring_string [LEN+2] ;
    gb_mxstring_to_string (semiring_string, LEN, mxstring, "semiring") ;

    //--------------------------------------------------------------------------
    // convert the string to a semiring
    //--------------------------------------------------------------------------

    return (gb_string_to_semiring (semiring_string, atype, btype)) ;
}

