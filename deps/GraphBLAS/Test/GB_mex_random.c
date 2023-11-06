//------------------------------------------------------------------------------
// GB_mex_random: construct a random matrix, double or Complex
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "A = GB_mex_random (nrows, ncols, ntuples," \
              " complex, seed, make_symmetric, no_self_edges, method)"

#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;
#define FREE_ALL GB_mx_put_global (true) ;   

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;

    // check inputs
    if (nargout > 1 || nargin == 0 || nargin > 8)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    int64_t GET_SCALAR (0, int64_t , nrows, 4) ;
    int64_t GET_SCALAR (1, int64_t , ncols, 4) ;
    int64_t GET_SCALAR (2, int64_t , ntuples, 0) ;
    bool    GET_SCALAR (3, bool    , A_complex, false) ;
    int64_t GET_SCALAR (4, uint64_t, seed, 1) ;
    bool    GET_SCALAR (5, bool    , make_symmetric, false) ;
    bool    GET_SCALAR (6, bool    , no_self_edges, false) ;
    int     GET_SCALAR (7, int     , method, 0) ;

    // construct the random matrix
    simple_rand_seed (seed) ;
    if (method == 3)
    {
        // test out-of-memory condition
        METHOD (GB_mx_random_matrix (&A, make_symmetric, no_self_edges,
            nrows, ncols, ntuples, method, A_complex)) ;
    }
    else
    {
        GrB_Info info = GB_mx_random_matrix (&A, make_symmetric, no_self_edges,
            nrows, ncols, ntuples, method, A_complex) ;
        if (info != GrB_SUCCESS)
        {
            mexErrMsgTxt ("GB_mx_random_matrix failed") ;
        }
    }

    // return A
    pargout [0] = GB_mx_Matrix_to_mxArray (&A, "A final", false) ;

    FREE_ALL ;
}

