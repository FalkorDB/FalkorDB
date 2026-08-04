//------------------------------------------------------------------------------
// gbmex_mdiag: construct a diagonal matrix from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// C = gbmex_mdiag (ghb, v, k, desc)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&V_to_free) ;  \
    GrB_Descriptor_free (&desc) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = gbmex_mdiag (ghb, v, k, desc)"

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

    GrB_Matrix *C_opaque = NULL, C = NULL, V = NULL, V_to_free = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_Type ctype = NULL ;

    GBMX_USAGE (nargin >= 2 && nargin <= 4 && nargout <= 2, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    double *kind_output = NULL ;
    if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;
    pargout [1] = mxCreateDoubleScalar (0) ;
    kind_output = (double *) mxGetData (pargout [1]) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [6] ;
    mxArray *Cell [2] ;
    char String [2][LEN+2] ;
    int nmatrices, nstrings, ncells ;
    struct gb_descriptor_struct gbdesc ;
    gbmx_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String,
        &nstrings, Cell, &ncells, &gbdesc) ;

    if (gbdesc.is_present) nargin-- ;

    int64_t k = 0 ;
    if (nargin > 2)
    { 
        k = gbmx_get_int64_scalar (pargin [2], "k") ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor (&desc, &gbdesc, arena, err)) ;

    //--------------------------------------------------------------------------
    // get the inputs
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&V, &V_to_free, &(Matrix [0]), arena, err)) ;

    uint64_t ncols ;
    OK (GrB_Matrix_ncols (&ncols, V)) ;
    CHECK_ERROR (ncols != 1, "v must be a column vector") ;

    int s ;
    OK (GrB_Matrix_get_INT32 (V, &s, GxB_SPARSITY_STATUS)) ;
    CHECK_ERROR (s == GxB_HYPERSPARSE, "v cannot be hypersparse") ;

    //--------------------------------------------------------------------------
    // construct C
    //--------------------------------------------------------------------------

    uint64_t n ;
    OK (GxB_Matrix_type (&ctype, V)) ;
    OK (GrB_Matrix_nrows (&n, V)) ;
    n += ABS (k) ;
    OK (gb_get_format (n, n, NULL, NULL, &(gbdesc.fmt), err)) ;
    OK (gb_new (&C, ctype, n, n, gbdesc.fmt, 0, arena, err)) ;

    //--------------------------------------------------------------------------
    // compute C = diag (v, k)
    //--------------------------------------------------------------------------

    OK1 (C, GxB_Matrix_diag (C, (GrB_Vector) V, k, desc)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    OK (gb_export (C_opaque, &C, gbdesc.kind, ghb, err)) ;
    (*kind_output) = (double) gbdesc.kind ;
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

