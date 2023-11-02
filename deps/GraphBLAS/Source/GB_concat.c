//------------------------------------------------------------------------------
// GB_concat: concatenate an array of matrices into a single matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORKSPACE               \
    GB_WERK_POP (Tile_cols, int64_t) ;  \
    GB_WERK_POP (Tile_rows, int64_t) ;

#define GB_FREE_ALL                     \
    GB_FREE_WORKSPACE ;                 \
    GB_phybix_free (C) ;

#include "GB_concat.h"

GrB_Info GB_concat                  // concatenate a 2D array of matrices
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n
    const GrB_Index m,
    const GrB_Index n,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

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

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C input for GB_concat", GB0) ;
    for (int64_t k = 0 ; k < m*n ; k++)
    { 
        GrB_Matrix A = Tiles [k] ;
        GB_RETURN_IF_NULL_OR_FAULTY (A) ;
        ASSERT_MATRIX_OK (A, "Tile[k] input for GB_concat", GB0) ;
        GB_MATRIX_WAIT (A) ;
    }

    //--------------------------------------------------------------------------
    // check the sizes and types of each tile
    //--------------------------------------------------------------------------

    bool csc = C->is_csc ;
    GrB_Type ctype = C->type ;

    for (int64_t i = 0 ; i < m ; i++)
    { 
        GrB_Matrix A = GB_TILE (Tiles, i, 0) ;
        Tile_rows [i] = GB_NROWS (A) ;
    }

    for (int64_t j = 0 ; j < n ; j++)
    { 
        GrB_Matrix A = GB_TILE (Tiles, 0, j) ;
        Tile_cols [j] = GB_NCOLS (A) ;
    }

    bool C_is_full = true ;
    bool C_iso = false ;
    const size_t csize = ctype->size ;
    const GB_Type_code ccode = ctype->code ;
    GB_void cscalar [GB_VLA(csize)] ;
    GB_void ascalar [GB_VLA(csize)] ;
    memset (cscalar, 0, csize) ;
    memset (ascalar, 0, csize) ;
    int64_t cnz = 0 ;
    int64_t cnvec_estimate = 0 ;    // upper bound on C->nvec if hypersparse

    for (int64_t i = 0 ; i < m ; i++)
    {
        for (int64_t j = 0 ; j < n ; j++)
        {

            //------------------------------------------------------------------
            // get the (i,j) tile
            //------------------------------------------------------------------

            GrB_Matrix A = GB_TILE (Tiles, i, j) ;

            //------------------------------------------------------------------
            // check the types and dimensions
            //------------------------------------------------------------------

            int64_t nrows = GB_NROWS (A) ;
            int64_t ncols = GB_NCOLS (A) ;
            int64_t anz = GB_nnz (A) ;
            int A_sparsity = GB_sparsity (A) ;
            if (A_sparsity == GxB_HYPERSPARSE)
            { 
                cnvec_estimate += A->nvec ;
            }
            else
            { 
                int64_t n = csc ? ncols : nrows ;
                cnvec_estimate += GB_IMIN (n, anz) ;
            }
            GrB_Type atype = A->type ;
            #define offset (GB_Global_print_one_based_get ( ) ? 1 : 0)
            if (!GB_Type_compatible (ctype, atype))
            { 
                GB_FREE_WORKSPACE ;
                GB_ERROR (GrB_DOMAIN_MISMATCH,
                    "Input matrix Tiles{" GBd "," GBd "} of type [%s]\n"
                    "cannot be typecast to output of type [%s]\n",
                    i+offset, j+offset, atype->name, ctype->name) ;
            }
            int64_t tile_rows = Tile_rows [i] ;
            if (tile_rows != nrows)
            { 
                GB_FREE_WORKSPACE ;
                GB_ERROR (GrB_DIMENSION_MISMATCH,
                    "Input matrix Tiles{" GBd "," GBd "} is " GBd "-by-" GBd
                    "; its row\ndimension must match all other matrices Tiles{"
                    GBd ",:}, which is " GBd "\n", i+offset, j+offset,
                    nrows, ncols, i+offset, tile_rows) ;
            }
            int64_t tile_cols = Tile_cols [j] ;
            if (tile_cols != ncols)
            { 
                GB_FREE_WORKSPACE ;
                GB_ERROR (GrB_DIMENSION_MISMATCH,
                    "Input matrix Tiles{" GBd "," GBd "} is " GBd "-by-" GBd
                    "; its column\ndimension must match all other matrices "
                    "Tiles{:," GBd "}, which is " GBd "\n", i+offset, j+offset,
                    nrows, ncols, j+offset, tile_cols) ;
            }

            //------------------------------------------------------------------
            // check if C is iso, full, and/or empty
            //------------------------------------------------------------------

            bool A_full = (A_sparsity == GxB_FULL) ;
            bool A_empty = (anz == 0) ;
            bool A_iso = A->iso || (anz == 1 && A_sparsity != GxB_BITMAP) ;

            // C is full only if all tiles are full.
            C_is_full = C_is_full && A_full ;

            // get the iso value of an iso tile, typecasted to C->type
            if (A_iso)
            {
                GB_cast_scalar (ascalar, ccode, A->x, A->type->code, csize) ;
                if (cnz == 0)
                { 
                    // A is the first non-empty iso tile seen while C is empty;
                    // C becomes non-empty and iso, with the iso value from A.
                    C_iso = true ;
                    memcpy (cscalar, ascalar, csize) ;
                }
            }

            // C is iso only if at least one tile is iso, and all others empty
            // or iso with the same value as the first non-empty iso tile
            if (C_iso)
            {
                if (A_empty)
                { 
                    // C remains iso
                }
                else if (A_iso)
                { 
                    // C and A are both iso; check if iso values are the same
                    C_iso = C_iso && (memcmp (cscalar, ascalar, csize) == 0) ;
                }
                else
                { 
                    // otherwise, C is non-iso
                    C_iso = false ;
                }
            }

            cnz += anz ;
        }
    }

    //--------------------------------------------------------------------------
    // replace Tile_rows and Tile_cols with their cumulative sum
    //--------------------------------------------------------------------------

    GB_cumsum (Tile_rows, m, NULL, 1, Werk) ;
    GB_cumsum (Tile_cols, n, NULL, 1, Werk) ;
    int64_t cnrows = Tile_rows [m] ;
    int64_t cncols = Tile_cols [n] ;
    if (cnrows != GB_NROWS (C) || cncols != GB_NCOLS (C))
    { 
        GB_FREE_WORKSPACE ;
        GB_ERROR (GrB_DIMENSION_MISMATCH,
            "C is " GBd "-by-" GBd " but Tiles{:,:} is " GBd "-by-" GBd "\n",
            GB_NROWS (C), GB_NCOLS (C), cnrows, cncols) ;
    }

    //--------------------------------------------------------------------------
    // C = concatenate (Tiles)
    //--------------------------------------------------------------------------

    if (cnz == 0)
    { 
        // construct C as an empty matrix
        GBURBLE ("(empty concat) ") ;
        GB_OK (GB_clear (C, Werk)) ;
    }
    else if (C_is_full)
    { 
        // construct C as full
        GBURBLE ("(%sfull concat) ", C_iso ? "iso " : "") ;
        GB_OK (GB_concat_full (C, C_iso, cscalar,
            Tiles, m, n, Tile_rows, Tile_cols, Werk)) ;
    }
    else if (GB_convert_s2b_test (C->bitmap_switch, cnz, cnrows, cncols))
    { 
        // construct C as bitmap
        GBURBLE ("(%sbitmap concat) ", C_iso ? "iso " : "") ;
        GB_OK (GB_concat_bitmap (C, C_iso, cscalar,
            cnz, Tiles, m, n, Tile_rows, Tile_cols, Werk)) ;
    }
    else if (GB_convert_sparse_to_hyper_test (C->hyper_switch, cnvec_estimate,
        C->vdim))
    { 
        // construct C as hypersparse
        GBURBLE ("(%shyper concat) ", C_iso ? "iso " : "") ;
        GB_OK (GB_concat_hyper (C, C_iso, cscalar,
            cnz, Tiles, m, n, Tile_rows, Tile_cols, Werk)) ;
    }
    else
    { 
        // construct C as sparse
        GBURBLE ("(%ssparse concat) ", C_iso ? "iso " : "") ;
        GB_OK (GB_concat_sparse (C, C_iso, cscalar,
            cnz, Tiles, m, n, Tile_rows, Tile_cols, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // conform C to its desired format and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (C, "C before conform for GB_concat", GB0) ;
    GB_OK (GB_conform (C, Werk)) ;
    ASSERT_MATRIX_OK (C, "C output for GB_concat", GB0) ;
    return (GrB_SUCCESS) ;
}

