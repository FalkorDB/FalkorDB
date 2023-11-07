//------------------------------------------------------------------------------
// gb_norm_kind: determine the kind of norm to compute
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

// 'fro':       Frobenius norm
// 1:           1-norm
// 2:           2-norm
// INFINITY:    inf-norm
// -INFINITY:   (-inf)-norm

int64_t gb_norm_kind (const mxArray *arg)
{
    if (mxIsChar (arg))
    {
        char string [65] ;
        gb_mxstring_to_string (string, 64, arg, "kind") ;
        if (MATCH (string, "fro"))
        { 
            return (0) ;
        }
        else
        { 
            // unknown string
            ERROR ("unknown norm") ;
        }
    }
    else
    {
        double x = mxGetScalar (arg) ;
        if (x == INFINITY)
        { 
            return (INT64_MAX) ;
        }
        else if (x == -INFINITY)
        { 
            return (INT64_MIN) ;
        }
        else if (x == 1 || x == 2)
        { 
            return ((int64_t) x) ;
        }
        else
        { 
            ERROR ("unknown norm") ;
        }
    }
    return (0) ;
}

