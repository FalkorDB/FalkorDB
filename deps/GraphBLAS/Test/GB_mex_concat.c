//------------------------------------------------------------------------------
// GB_mex_concat: C = concat (Tiles)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_concat (Tiles, type, fmt)"

#define FREE_ALL                                \
{                                               \
    if (Tiles != NULL)                          \
    {                                           \
        for (int64_t k = 0 ; k < m*n ; k++)     \
        {                                       \
            GrB_Matrix_free_(&(Tiles [k])) ;    \
        }                                       \
    }                                           \
    mxFree (Tiles) ;                            \
    GrB_Matrix_free_(&C) ;                      \
    GB_mx_put_global (true) ;                   \
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
    GrB_Matrix C = NULL ;
    GrB_Matrix *Tiles = NULL ;
    int64_t m = 0, n = 0 ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get Tiles (shallow copy)
    const mxArray *mxTiles = pargin [0] ;
    if (!mxIsCell (mxTiles))
    {
        FREE_ALL ;
        mexErrMsgTxt ("Tiles must be a cell array") ;
    }
    m = mxGetM (mxTiles) ;
    n = mxGetN (mxTiles) ;
    Tiles = mxMalloc (m * n * sizeof (GrB_Matrix)) ;
    for (int64_t j = 0 ; j < n ; j++)
    {
        for (int64_t i = 0 ; i < m ; i++)
        {
            // get the Tiles {i,j} matrix.
            // Tiles is row-major but mxTiles is column-major
            mxArray *Tile = mxGetCell (mxTiles, i+j*m) ;
            Tiles [i*n+j] = GB_mx_mxArray_to_Matrix (Tile, "Tile", false, true);
        }
    }

    // get the type of C, default is double
    GrB_Type ctype = GB_mx_string_to_Type (PARGIN (1), GrB_FP64) ;

    // determine the # of rows of C from Tiles {:,0}
    GrB_Index cnrows = 0 ;
    for (int64_t i = 0 ; i < m ; i++)
    {
        GrB_Index anrows ;
        OK (GrB_Matrix_nrows (&anrows, Tiles [i*n])) ;
        cnrows += anrows ;
    }

    // determine the # of columms of C from Tiles {0,:}
    GrB_Index cncols = 0 ;
    for (int64_t j = 0 ; j < n ; j++)
    {
        GrB_Index ancols ;
        OK (GrB_Matrix_ncols (&ancols, Tiles [j])) ;
        cncols += ancols ;
    }

    // get the format of C, default is by column
    int GET_SCALAR (2, int, fmt, GxB_BY_COL) ;

    // construct the empty C
    #define GET_DEEP_COPY                               \
        GrB_Matrix_new (&C, ctype, cnrows, cncols) ;    \
        GxB_Matrix_Option_set_(C, GxB_FORMAT, fmt) ;
    #define FREE_DEEP_COPY  GrB_Matrix_free_(&C) ;
    GET_DEEP_COPY ;

    // C = concat (Tiles)
    METHOD (GxB_Matrix_concat (C, Tiles, m, n, NULL)) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;
    FREE_ALL ;
}

