//------------------------------------------------------------------------------
// GB_split: split a matrix into an array of matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORKSPACE                   \
    GB_WERK_POP (Tile_cols, int64_t) ;      \
    GB_WERK_POP (Tile_rows, int64_t) ;

#define GB_FREE_ALL                         \
    GB_FREE_WORKSPACE ;                     \
    for (int64_t k = 0 ; k < m*n ; k++)     \
    {                                       \
        GB_Matrix_free (&(Tiles [k])) ;     \
    }

#include "GB_split.h"

GrB_Info GB_split                   // split a matrix
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const GrB_Index m,
    const GrB_Index n,
    const GrB_Index *Tile_nrows,    // array of size m
    const GrB_Index *Tile_ncols,    // array of size n
    const GrB_Matrix A,             // input matrix
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    // set all Tiles to NULL
    GrB_Info info ;
    ASSERT (Tiles != NULL) ;
    memset (Tiles, 0, m * n * sizeof (GrB_Matrix)) ;

    GB_WERK_DECLARE (Tile_rows, int64_t) ;
    GB_WERK_DECLARE (Tile_cols, int64_t) ;
    GB_WERK_PUSH (Tile_rows, m+1, int64_t) ;
    GB_WERK_PUSH (Tile_cols, n+1, int64_t) ;
    if (Tile_rows == NULL || Tile_cols == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A input for GB_split", GB0) ;
    GB_MATRIX_WAIT (A) ;
    if (A->iso)
    { 
        GBURBLE ("(iso split) ") ;
    }

    //--------------------------------------------------------------------------
    // check the sizes of each tile
    //--------------------------------------------------------------------------

    int64_t nrows = GB_NROWS (A) ;
    int64_t ncols = GB_NCOLS (A) ;

    int64_t s = 0 ;
    for (int64_t i = 0 ; i < m ; i++)
    {
        GrB_Index tile_nrows = Tile_nrows [i] ;     // # of rows in Tile{i,:}
        if (tile_nrows < 0 || tile_nrows > nrows)
        { 
            return (GrB_DIMENSION_MISMATCH) ;
        }
        Tile_rows [i] = s ;                         // cumulative sum
        s += tile_nrows ;
    }
    if (s != nrows)
    { 
        return (GrB_DIMENSION_MISMATCH) ;
    }
    Tile_rows [m] = nrows ;

    s = 0 ;
    for (int64_t j = 0 ; j < n ; j++)
    {
        GrB_Index tile_ncols = Tile_ncols [j] ;     // # of cols in Tile{:,j}
        if (tile_ncols < 0 || tile_ncols > ncols)
        { 
            return (GrB_DIMENSION_MISMATCH) ;
        }
        Tile_cols [j] = s ;                         // cumulative sum
        s += tile_ncols ;
    }
    if (s != ncols)
    { 
        return (GrB_DIMENSION_MISMATCH) ;
    }
    Tile_cols [n] = ncols ;

    //--------------------------------------------------------------------------
    // Tiles = split (A)
    //--------------------------------------------------------------------------

    if (GB_IS_FULL (A))
    { 
        // A is full
        GBURBLE ("(full split) ") ;
        GB_OK (GB_split_full (Tiles, m, n, Tile_rows, Tile_cols, A, Werk)) ;
    }
    else if (GB_IS_BITMAP (A))
    { 
        // A is bitmap
        GBURBLE ("(bitmap split) ") ;
        GB_OK (GB_split_bitmap (Tiles, m, n, Tile_rows, Tile_cols, A, Werk));
    }
    else
    { 
        // A is sparse/hypersparse, each Tile has the same sparsity as A
        GBURBLE ("(sparse/hyper split) ") ;
        GB_OK (GB_split_sparse (Tiles, m, n, Tile_rows, Tile_cols, A, Werk));
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

