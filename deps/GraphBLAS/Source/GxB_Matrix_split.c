//------------------------------------------------------------------------------
// GxB_Matrix_split: split a matrix into an array of matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input matrix A is split into a 2D array of size m-by-n.  The Tile{i,j}
// matrix has dimension Tile_nrows[i]-by-Tile_ncols[j].

#include "GB_split.h"

GrB_Info GxB_Matrix_split           // split a matrix into 2D array of matrices
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const GrB_Index m,
    const GrB_Index n,
    const GrB_Index *Tile_nrows,    // array of size m
    const GrB_Index *Tile_ncols,    // array of size n
    const GrB_Matrix A,             // input matrix to split
    const GrB_Descriptor desc       // unused, except threading control
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_split (Tiles, m, n, Tile_nrows, Tile_ncols, A, "
        "desc)") ;
    GB_BURBLE_START ("GxB_Matrix_split") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    if (m <= 0 || n <= 0)
    { 
        return (GrB_INVALID_VALUE) ;
    }
    GB_RETURN_IF_NULL (Tiles) ;
    GB_RETURN_IF_NULL (Tile_nrows) ;
    GB_RETURN_IF_NULL (Tile_ncols) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // Tiles = split (A)
    //--------------------------------------------------------------------------

    info = GB_split (Tiles, m, n, Tile_nrows, Tile_ncols, A, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

