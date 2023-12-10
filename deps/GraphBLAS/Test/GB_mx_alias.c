//------------------------------------------------------------------------------
// GB_mx_alias:  return an aliased argument
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

GrB_Matrix GB_mx_alias      // output matrix (NULL if no match found)
(
    char *arg_name,         // name of the output matrix
    const mxArray *arg,     // string to select the alias
    char *arg1_name,        // name of first possible alias
    GrB_Matrix arg1,        // first possible alias
    char *arg2_name,        // name of 2nd possible alias
    GrB_Matrix arg2         // second possible alias
)
{

    // get the string from the mxArray
    #define LEN 256
    char s [LEN] ;
    mxGetString (arg, s, LEN) ;
    if (MATCH (s, arg1_name))
    {
        return (arg1) ;
    }
    else if (MATCH (s, arg2_name))
    {
        return (arg2) ;
    }

    // no alias found
    return (NULL) ;
}

