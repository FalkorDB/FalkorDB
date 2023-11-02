//------------------------------------------------------------------------------
// gbcat: matrix concatenation
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbcat is an interface to GxB_Matrix_concat.

// Usage:

// C = gbcat (Tiles, desc)

// where Tiles is a 2D cell array of matrices.

#include "gb_interface.h"

#define USAGE "usage: C = GrB.cat (Tiles, desc)"

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

    gb_usage (nargin >= 1 && nargin <= 2 && nargout <= 2, USAGE) ;

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

    CHECK_ERROR (nmatrices > 0 || nstrings > 0 || ncells != 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the tiles
    //--------------------------------------------------------------------------

    mxArray *mxTiles = Cell [0] ;
    int64_t m = mxGetM (mxTiles) ;
    int64_t n = mxGetN (mxTiles) ;
    GrB_Matrix *Tiles = mxMalloc (m * n * sizeof (GrB_Matrix)) ;
    for (int64_t j = 0 ; j < n ; j++)
    {
        for (int64_t i = 0 ; i < m ; i++)
        {
            // get the Tiles {i,j} matrix.
            // Tiles is row-major but mxTiles is column-major
            Tiles [i*n+j] = gb_get_shallow (mxGetCell (mxTiles, i+j*m)) ;
        }
    }

    //--------------------------------------------------------------------------
    // determine the # of rows of C from Tiles {:,0}
    //--------------------------------------------------------------------------

    GrB_Index cnrows = 0 ;
    for (int64_t i = 0 ; i < m ; i++)
    {
        GrB_Index anrows ;
        OK (GrB_Matrix_nrows (&anrows, Tiles [i*n])) ;
        cnrows += anrows ;
    }

    //--------------------------------------------------------------------------
    // determine the # of columms of C from Tiles {0,:}
    //--------------------------------------------------------------------------

    GrB_Index cncols = 0 ;
    for (int64_t j = 0 ; j < n ; j++)
    {
        GrB_Index ancols ;
        OK (GrB_Matrix_ncols (&ancols, Tiles [j])) ;
        cncols += ancols ;
    }

    //--------------------------------------------------------------------------
    // determine the type of C
    //--------------------------------------------------------------------------

    GrB_Type ctype ;
    OK (GxB_Matrix_type (&ctype, Tiles [0])) ;
    for (int64_t k = 1 ; k < m*n ; k++)
    {
        GrB_Type atype ;
        OK (GxB_Matrix_type (&atype, Tiles [k])) ;
        ctype = gb_default_type (ctype, atype) ;
    }

    //--------------------------------------------------------------------------
    // create the matrix C and set its format and sparsity
    //--------------------------------------------------------------------------

    fmt = gb_get_format (cnrows, cncols, NULL, NULL, fmt) ;
    GrB_Matrix C = gb_new (ctype, cnrows, cncols, fmt, sparsity) ;

    //--------------------------------------------------------------------------
    // C = concatenate (Tiles)
    //--------------------------------------------------------------------------

    OK1 (C, GxB_Matrix_concat (C, Tiles, m, n, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    for (int64_t k = 0 ; k < m*n ; k++)
    {
        OK (GrB_Matrix_free (&(Tiles [k]))) ;
    }
    OK (GrB_Descriptor_free (&desc)) ;
    mxFree (Tiles) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    GB_WRAPUP ;
}

