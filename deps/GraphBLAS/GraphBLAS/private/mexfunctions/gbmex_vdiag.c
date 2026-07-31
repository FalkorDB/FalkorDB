//------------------------------------------------------------------------------
// gbmex_vdiag: extract a diagonal of a matrix, as a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// v = gbmex_vdiag (ghb, A, k, desc)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Descriptor_free (&desc) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&V) ;

#define USAGE "usage: v = gbmex_vdiag (ghb, A, k, desc)"

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

    GrB_Matrix *V_opaque = NULL, V = NULL, A = NULL, A_to_free = NULL ;
    GrB_Descriptor desc = NULL ;
    int64_t k = 0 ;

    GBMX_USAGE (nargin >= 2 && nargin <= 4 && nargout <= 1, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&V_opaque) ;

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [6] ;
    mxArray *Cell [2] ;
    char String [2][LEN+2] ;
    int nmatrices, nstrings, ncells ;
    struct gb_descriptor_struct gbdesc ;
    gbmx_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String,
        &nstrings, Cell, &ncells, &gbdesc) ;

    if (gbdesc.is_present) nargin-- ;

    if (nargin > 2)
    { 
        k = gbmx_get_int64_scalar (pargin [2], "k") ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the inputs
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;

    //--------------------------------------------------------------------------
    // construct V
    //--------------------------------------------------------------------------

    GrB_Type vtype = NULL ;
    int64_t n, nrows, ncols ;
    OK (GxB_Matrix_type (&vtype, A)) ;
    OK (GrB_Matrix_nrows ((uint64_t *) &nrows, A)) ;
    OK (GrB_Matrix_ncols ((uint64_t *) &ncols, A)) ;

    if (k >= ncols || k <= -nrows)
    { 
        // output vector V must have zero length
        n = 0 ;
    }
    else if (k >= 0)
    { 
        // if k is in range 0 to n-1, V must have length min (m,n-k)
        n = MIN (nrows, ncols - k) ;
    }
    else
    { 
        // if k is in range -1 to -m+1, V must have length min (m+k,n)
        n = MIN (nrows + k, ncols) ;
    }

    OK (gb_new (&V, vtype, n, 1, GxB_BY_COL, 0, arena, err)) ;

    //--------------------------------------------------------------------------
    // compute v = diag (A, k)
    //--------------------------------------------------------------------------

    OK1 (V, GxB_Vector_diag ((GrB_Vector) V, A, k, desc)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (V_opaque, &V, gbdesc.kind, ghb, err)) ;
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&V) ;
    }

    gb_wrapup ( ) ;
}

