//------------------------------------------------------------------------------
// GB_mex_split: C = split (Tiles)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_split (A, ms, ns)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    mxFree (Tiles) ;                    \
    GB_mx_put_global (true) ;           \
}

#define OK(method)                      \
{                                       \
    info = method ;                     \
    if (info != GrB_SUCCESS)            \
    {                                   \
        printf ("%d at %d\n", info, __LINE__) ;  \
        mexErrMsgTxt ("failed") ;       \
    }                                   \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    GrB_Matrix *Tiles = NULL ;

    // check inputs
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get ms (deep copy)
    if (mxGetClassID (pargin [1]) != mxDOUBLE_CLASS)
    {
        mexErrMsgTxt ("ms must be double") ;
    }
    double *ms_double = mxGetDoubles (pargin [1]) ;
    GrB_Index m = mxGetNumberOfElements (pargin [1]) ;
    GrB_Index *Tile_nrows = mxMalloc (m * sizeof (GrB_Index)) ;
    for (int64_t k = 0 ; k < m ; k++)
    {
        Tile_nrows [k] = (GrB_Index) (ms_double [k]) ;
    }

    // get ns (deep copy)
    if (mxGetClassID (pargin [2]) != mxDOUBLE_CLASS)
    {
        mexErrMsgTxt ("ns must be double") ;
    }
    double *ns_double = mxGetDoubles (pargin [2]) ;
    GrB_Index n = mxGetNumberOfElements (pargin [2]) ;
    GrB_Index *Tile_ncols = mxMalloc (n * sizeof (GrB_Index)) ;
    for (int64_t k = 0 ; k < n ; k++)
    {
        Tile_ncols [k] = (GrB_Index) (ns_double [k]) ;
    }

    // create Tiles
    Tiles = mxCalloc (m * n, sizeof (GrB_Matrix)) ;

    // construct the empty Tiles array
    #define GET_DEEP_COPY                                   \
        memset (Tiles, 0, m * n * sizeof (GrB_Matrix)) ;
    #define FREE_DEEP_COPY                                  \
        for (int64_t k = 0 ; k < m*n ; k++)                 \
        {                                                   \
            GrB_Matrix_free (&(Tiles [k])) ;                \
        }

    // Tiles = split (A, ms, ns)
    METHOD (GxB_Matrix_split (Tiles, m, n, Tile_nrows, Tile_ncols, A, NULL)) ;

    // return C as a cell array and free the GraphBLAS tiles
    pargout [0] = mxCreateCellMatrix (m, n) ;
    for (int64_t j = 0 ; j < n ; j++)
    {
        for (int64_t i = 0 ; i < m ; i++)
        {
            mxArray *T = GB_mx_Matrix_to_mxArray (&(Tiles [i*n+j]),
                "Tile output", true) ;
            mxSetCell (pargout [0], i+j*m, T) ;
        }
    }
    FREE_ALL ;
}

