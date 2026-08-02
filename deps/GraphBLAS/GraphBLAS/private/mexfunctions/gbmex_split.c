//------------------------------------------------------------------------------
// gbmex_split: matrix split
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_split is an interface to GxB_Matrix_split.

// Usage:

// C = gbmex_split (ghb, A, m, n, desc)

// where C is a 2D cell array of matrices.

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    mxFree (Tiles) ;                \
    mxFree (Tiles_opaque) ;         \
    mxFree (Tile_nrows) ;           \
    mxFree (Tile_ncols) ;           \
    GrB_Matrix_free (&A_to_free) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;

#define USAGE "usage: C = GrB.split (A, m, n)"

//------------------------------------------------------------------------------
// gbmex_split mexFunction
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
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, A_to_free = NULL ;

    GBMX_USAGE (nargin == 4 && nargout <= 1, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    //--------------------------------------------------------------------------
    // get the tile sizes, kind, and create the output arguments
    //--------------------------------------------------------------------------

    uint64_t m, n ;
    uint64_t *Tile_nrows = gbmx_get_integer_list (pargin [2], &m) ;
    uint64_t *Tile_ncols = gbmx_get_integer_list (pargin [3], &n) ;

    GrB_Matrix *Tiles = mxCalloc (m * n, sizeof (GrB_Matrix)) ;
    GrB_Matrix **Tiles_opaque = NULL ;

    pargout [0] = mxCreateCellMatrix (m, n) ;

    if (ghb)
    {
        Tiles_opaque = mxCalloc (m * n, sizeof (GrB_Matrix *)) ;
        for (int64_t i = 0 ; i < m ; i++)
        { 
            for (int64_t j = 0 ; j < n ; j++)
            { 
                // pargout [0] and Tiles_opaque are in column-major form
                mxArray *mxCell_entry = 
                    gbmx_export_ghb_mxstruct (&(Tiles_opaque [i+j*m])) ;
                mxSetCell (pargout [0], i+j*m, mxCell_entry) ;
            }
        }
    }

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [1]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix A
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;

    //--------------------------------------------------------------------------
    // Tiles = split (A)
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_split_arena (Tiles, m, n, Tile_nrows, Tile_ncols, A,
        arena, arena, NULL)) ;

    //--------------------------------------------------------------------------
    // export the Tiles array into the output cell array
    //--------------------------------------------------------------------------

    for (int64_t i = 0 ; i < m ; i++)
    { 
        for (int64_t j = 0 ; j < n ; j++)
        { 
            // Tiles is in row-major form;
            // Tiles_opaque is in column-major form
            GrB_Matrix *Cell_opaque = NULL ;
            if (ghb) Cell_opaque = Tiles_opaque [i+j*m] ;
            OK (gb_export (Cell_opaque, &Tiles [i*n+j], KIND_GRB, ghb, err)) ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        for (int64_t i = 0 ; i < m ; i++)
        { 
            for (int64_t j = 0 ; j < n ; j++)
            { 
                // Tiles is in row-major form;
                // pargout [0] is in column-major form
                mxSetCell (pargout [0], i+j*m,  
                    gbmx_export_grb_mxstruct (&Tiles [i*n+j])) ;
            }
        }
    }

    FREE_WORK ;
    gb_wrapup ( ) ;
}

