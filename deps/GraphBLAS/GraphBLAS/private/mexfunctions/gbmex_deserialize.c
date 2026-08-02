//------------------------------------------------------------------------------
// gbmex_deserialize: deserialize a blob into a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_deserialize is an interface to GrB_Matrix_deserialize.

// Usage:

// C = gbmex_deserialize (blob)

// The blob can be either a MATLAB or a GrB matrix.  In either case, it must
// be dense (not sparse) with all entries present, and of type GrB_UINT8.
// C is returned as a GrB matrix.

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                       \
    GrB_Matrix_free (&Blob_to_free) ;

#undef  FREE_ALL
#define FREE_ALL                        \
    FREE_WORK ;                         \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = GrB.deserialize (blob)"

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

    GrB_Matrix *C_opaque = NULL, C = NULL, Blob = NULL, Blob_to_free = NULL ;

    GBMX_USAGE (nargin == 2 && nargout <= 1, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [1]) ;

    CHECK_ERROR (Matrix [0].type != GrB_UINT8,
        "blob must be a uint8 dense matrix/vector") ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the blob, normally a row or column vector, but can be a dense matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&Blob, &Blob_to_free, &(Matrix [0]), arena, err)) ;

    bool Blob_is_dense = false ;
    OK (gb_is_dense (&Blob_is_dense, Blob, err)) ;
    CHECK_ERROR (!Blob_is_dense, "blob must be a uint8 dense matrix/vector") ;

    uint64_t nvals ;
    OK (GrB_Matrix_nvals (&nvals, Blob)) ;
    uint64_t blob_memsize = nvals * sizeof (uint8_t) ;
    const void *blob = Blob->x ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_deserialize_arena (&C, NULL, blob, blob_memsize,
        arena, arena, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    OK (gb_export (C_opaque, &C, KIND_GRB, ghb, err)) ;
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

