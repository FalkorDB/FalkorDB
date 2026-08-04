
//------------------------------------------------------------------------------
// gbmx_get_integer_list:  get a list of integers from an mxArray
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// uint64_t *List = gbmx_get_integer_list (mxList, &len) returns an mxMalloc
// allocated array, List, of size len, containing a list of integers contained
// in the mxArray mxList.

uint64_t *gbmx_get_integer_list
(
    const mxArray *mxList,
    uint64_t *len
)
{
    int64_t n = mxGetNumberOfElements (mxList) ;
    (*len) = (uint64_t) n ;
    mxClassID class = mxGetClassID (mxList) ;
    uint64_t *List = mxMalloc (n * sizeof (uint64_t)) ;
    if (class == mxINT64_CLASS || class == mxUINT64_CLASS)
    { 
        int64_t *p = (int64_t *) mxGetData (mxList) ;
        memcpy (List, p, n * sizeof (int64_t)) ;
    }
    else if (class == mxDOUBLE_CLASS)
    {
        double *p = (double *) mxGetData (mxList) ;
        for (int64_t k = 0 ; k < n ; k++)
        { 
            List [k] = (uint64_t) p [k] ;
            CHECK_ERROR ((double) List [k] != p [k],
                "dimensions must be integer") ;
        }
    }
    else
    { 
        ERROR ("unsupported type", GrB_DOMAIN_MISMATCH) ;
    }
    return (List) ;
}

