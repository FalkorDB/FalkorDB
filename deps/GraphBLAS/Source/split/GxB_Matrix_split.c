//------------------------------------------------------------------------------
// GxB_Matrix_split: split a matrix into an array of matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input matrix A is split into a 2D array of size m-by-n.  The Tile{i,j}
// matrix has dimension Tile_nrows[i]-by-Tile_ncols[j].

// The Tile matrices are allocated in the arenas from the current Context.

#include "split/GB_split.h"

GrB_Info GxB_Matrix_split           // split a matrix into 2D array of matrices
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const uint64_t m,
    const uint64_t n,
    const uint64_t *Tile_nrows,     // array of size m
    const uint64_t *Tile_ncols,     // array of size n
    const GrB_Matrix A,             // input matrix to split
    const GrB_Descriptor desc       // unused, except threading control
)
{
    GB_RETURN_IF_NULL (A) ;
    int header_arena = GB_Context_header_arena ( ) ;
    int data_arena = GB_Context_data_arena ( ) ;
    return (GxB_Matrix_split_arena (Tiles, m, n, Tile_nrows, Tile_ncols, A,
        header_arena, data_arena, desc)) ;
}

