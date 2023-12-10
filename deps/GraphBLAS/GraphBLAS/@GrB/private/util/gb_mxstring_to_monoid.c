//------------------------------------------------------------------------------
// gb_mxstring_to_monoid: get a GraphBLAS monoid from a built-in string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Monoid gb_mxstring_to_monoid        // return monoid from a string
(
    const mxArray *mxstring,            // built-in string
    const GrB_Type type                 // default type if not in the string
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (gb_mxarray_is_empty (mxstring))
    { 
        ERROR ("monoid missing") ;
    }

    //--------------------------------------------------------------------------
    // get the string
    //--------------------------------------------------------------------------

    #define LEN 256
    char opstring [LEN+2] ;
    gb_mxstring_to_string (opstring, LEN, mxstring, "monoid") ;

    //--------------------------------------------------------------------------
    // convert the string to a monoid
    //--------------------------------------------------------------------------

    return (gb_string_to_monoid (opstring, type)) ;
}

