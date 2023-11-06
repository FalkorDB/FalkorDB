//------------------------------------------------------------------------------
// GB_concat.h: definitions for GxB_concat
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CONCAT_H
#define GB_CONCAT_H
#include "GB.h"
#include "GB_transpose.h"
#include "GB_ek_slice.h"
#include "GB_build.h"

#define GB_TILE(Tiles,i,j) (*(Tiles + (i) * n + (j)))

GrB_Info GB_concat                  // concatenate a 2D array of matrices
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n
    const GrB_Index m,
    const GrB_Index n,
    GB_Werk Werk
) ;

GrB_Info GB_concat_full             // concatenate into a full matrix
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_iso,               // if true, construct C as iso
    const GB_void *cscalar,         // iso value of C, if C is io 
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n,
    const GrB_Index m,
    const GrB_Index n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    GB_Werk Werk
) ;

GrB_Info GB_concat_bitmap           // concatenate into a bitmap matrix
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_iso,               // if true, construct C as iso
    const GB_void *cscalar,         // iso value of C, if C is io 
    const int64_t cnz,              // # of entries in C
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n,
    const GrB_Index m,
    const GrB_Index n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    GB_Werk Werk
) ;

GrB_Info GB_concat_hyper            // concatenate into a hypersparse matrix
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_iso,               // if true, construct C as iso
    const GB_void *cscalar,         // iso value of C, if C is io 
    const int64_t cnz,              // # of entries in C
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n,
    const GrB_Index m,
    const GrB_Index n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    GB_Werk Werk
) ;

GrB_Info GB_concat_sparse           // concatenate into a sparse matrix
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_iso,               // if true, construct C as iso
    const GB_void *cscalar,         // iso value of C, if C is io 
    const int64_t cnz,              // # of entries in C
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n,
    const GrB_Index m,
    const GrB_Index n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    GB_Werk Werk
) ;

#endif

