//------------------------------------------------------------------------------
// gbmx_norm_kind: determine the kind of norm to compute
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// 'fro':       Frobenius norm
// 1:           1-norm
// 2:           2-norm
// INFINITY:    inf-norm
// -INFINITY:   (-inf)-norm

int64_t gbmx_norm_kind      // determine the kind of norm to compute
(
    const mxArray *arg
)
{
    if (mxIsChar (arg))
    {
        char string [LEN+2] ;
        gbmx_mxstring_to_string (string, LEN, arg, "kind") ;
        if (MATCH (string, "fro"))
        { 
            return (0) ;
        }
        else
        { 
            // unknown string
            ERROR ("unknown norm", GrB_INVALID_VALUE) ;
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
            ERROR ("unknown norm", GrB_INVALID_VALUE) ;
        }
    }
    return (0) ;
}

