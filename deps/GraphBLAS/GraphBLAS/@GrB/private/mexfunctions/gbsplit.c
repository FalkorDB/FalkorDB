//------------------------------------------------------------------------------
// gbsplit: matrix split
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbsplit is an interface to GxB_Matrix_split.

// Usage:

// C = gbsplit (A, m, n, desc)

// where C is a 2D cell array of matrices.

#include "gb_interface.h"

#define USAGE "usage: C = GrB.split (A, m, n, desc)"

//------------------------------------------------------------------------------
// gb_get_tilesizes:  get a list of integers
//------------------------------------------------------------------------------

static inline GrB_Index *gb_get_tilesizes (mxArray *mxList, GrB_Index *len)
{
    int64_t n = mxGetNumberOfElements (mxList) ;
    (*len) = (GrB_Index) n ;
    mxClassID class = mxGetClassID (mxList) ;
    GrB_Index *List = mxMalloc (n * sizeof (GrB_Index)) ;
    // use mxGetData (best for Octave, fine for MATLAB)
    if (class == mxINT64_CLASS)
    {
        int64_t *p = (int64_t *) mxGetData (mxList) ;
        memcpy (List, p, n * sizeof (int64_t)) ;
    }
    else if (class == mxUINT64_CLASS)
    {
        uint64_t *p = (uint64_t *) mxGetData (mxList) ;
        memcpy (List, p, n * sizeof (uint64_t)) ;
    }
    else if (class == mxDOUBLE_CLASS)
    {
        double *p = (double *) mxGetData (mxList) ;
        for (int64_t k = 0 ; k < n ; k++)
        {
            List [k] = (GrB_Index) p [k] ;
            CHECK_ERROR ((double) List [k] != p [k],
                "dimensions must be integer") ;
        }
    }
    else
    {
        ERROR ("unsupported type") ;
    }
    return (List) ;
}

//------------------------------------------------------------------------------
// gbsplit mexFunction
//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage ((nargin == 3 || nargin == 4) && nargout <= 2, USAGE) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    mxArray *Matrix [6], *String [2], *Cell [2] ;
    base_enum_t base ;
    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    int nmatrices, nstrings, ncells, sparsity ;
    GrB_Descriptor desc ;
    gb_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String, &nstrings,
        Cell, &ncells, &desc, &base, &kind, &fmt, &sparsity) ;

    CHECK_ERROR (nmatrices != 3 || nstrings > 0 || ncells > 0, USAGE) ;

    //--------------------------------------------------------------------------
    // get the input matrix A, Tile_nrows, and Tile_ncols
    //--------------------------------------------------------------------------

    GrB_Matrix A = gb_get_shallow (Matrix [0]) ;
    GrB_Index m, n ;
    GrB_Index *Tile_nrows = gb_get_tilesizes (Matrix [1], &m) ;
    GrB_Index *Tile_ncols = gb_get_tilesizes (Matrix [2], &n) ;
    GrB_Matrix *Tiles = mxMalloc (m * n * sizeof (GrB_Matrix)) ;

    //--------------------------------------------------------------------------
    // Tiles = split (A)
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_split (Tiles, m, n, Tile_nrows, Tile_ncols, A, desc)) ;

    //--------------------------------------------------------------------------
    // convert the Tiles array to a built-in cell array
    //--------------------------------------------------------------------------

    mxArray *C = mxCreateCellMatrix (m, n) ;
    for (int64_t i = 0 ; i < m ; i++)
    {
        for (int64_t j = 0 ; j < n ; j++)
        {
            // Tiles is in row-major form and C is in column-major form
            mxSetCell (C, i+j*m, gb_export (&Tiles [i*n+j], kind)) ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Descriptor_free (&desc)) ;
    mxFree (Tiles) ;
    mxFree (Tile_nrows) ;
    mxFree (Tile_ncols) ;

    //--------------------------------------------------------------------------
    // export the output cell array C
    //--------------------------------------------------------------------------

    pargout [0] = C ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    GB_WRAPUP ;
}

