//------------------------------------------------------------------------------
// GB_split.h: definitions for GB_split and related functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SPLIT_H
#define GB_SPLIT_H

#include "GB.h"
#define GB_TILE(Tiles,i,j) (*(Tiles + (i) * n + (j)))

GrB_Info GB_split                   // split a matrix
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const int64_t m,
    const int64_t n,
    const int64_t *Tile_nrows,      // array of size m
    const int64_t *Tile_ncols,      // array of size n
    const GrB_Matrix A,             // input matrix
    GB_Werk Werk
) ;

GrB_Info GB_split_bitmap            // split a bitmap matrix
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const int64_t m,
    const int64_t n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    const GrB_Matrix A,             // input matrix
    GB_Werk Werk
) ;

GrB_Info GB_split_full              // split a full matrix
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const int64_t m,
    const int64_t n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    const GrB_Matrix A,             // input matrix
    GB_Werk Werk
) ;

GrB_Info GB_split_sparse            // split a sparse matrix
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const int64_t m,
    const int64_t n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    const GrB_Matrix A,             // input matrix
    GB_Werk Werk
) ;

#endif

