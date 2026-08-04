//------------------------------------------------------------------------------
// gbmex_mtimes: sparse matrix-matrix multiplication over the standard semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_mtimes provides the mexFunction for computing the overloaded method C =
// mtimes (A,B) using the standard PLUS_TIMES_* semiring, and (mostly) the
// standard Octave/MATLAB rules for the sparsity of C.

// The standard rules state that if A or B are full, then C is always full.
// The rules here are slightly different:  C is full for (sparse or bitmap)
// times full, or full times (sparse or bitmap), using the MATLAB rule.  C is
// not full for hypersparse times full or full times hypersparse.  Instead, it
// is left sparse (or whatever format GraphBLAS decides to use).

// This method also allows for the inputs A and/or B to be transposed, but
// this parameter is not passed by MATLAB to the mtimes method.

// Usage:

// C = gbmex_mtimes (ghb, A, B)
// C = gbmex_mtimes (ghb, A, B, desc)

#include "gb_interface.h"
#include "gb_semiring.c"
#include "gb_string_to_semiring.c"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"
#include "gb_get_descriptor_mxm.c"

#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Matrix_free (&B_to_free) ;  \
    GrB_Scalar_free (&zero) ;       \
    GrB_Descriptor_free (&desc) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = gbmex_mtimes (ghb, A, B, desc)"

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

    GrB_Type atype, btype, ctype ;
    GrB_Matrix *C_opaque = NULL, C = NULL, A = NULL, B = NULL,
        A_to_free = NULL, B_to_free = NULL ;
    GrB_Scalar scalar = NULL, zero = NULL ;
    GrB_Descriptor desc = NULL ;

    GBMX_USAGE (nargin >= 3 && nargin <= 4 && nargout <= 1, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;

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

    CHECK_ERROR (nmatrices != 2 || nstrings > 0 || ncells > 0, USAGE) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor_mxm (&desc, &gbdesc, arena, err)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
    OK (gb_get_matrix (&B, &B_to_free, &(Matrix [1]), arena, err)) ;

    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GxB_Matrix_type (&btype, B)) ;

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp plus = NULL, times = NULL ;
    GrB_Monoid plus_monoid = NULL ;
    GrB_Semiring plus_times = NULL ;
    char semiring_string [LEN+2] ;
    GB_string_copy (semiring_string, "+.*", LEN) ;
    OK (gb_string_to_semiring (&plus_times, semiring_string, atype, btype,
        err)) ;
    OK (GrB_Semiring_get_VOID (plus_times, (void *) &plus_monoid,
        GxB_SEMIRING_MONOID)) ;
    OK (GrB_Semiring_get_VOID (plus_times, (void *) &times,
        GxB_SEMIRING_MULTIPLY)) ;
    OK (GrB_Monoid_get_VOID (plus_monoid, (void *) &plus, GxB_MONOID_OPERATOR));
    OK (gb_binaryop_ztype (&ctype, plus, err)) ;

    //--------------------------------------------------------------------------
    // construct C
    //--------------------------------------------------------------------------

    // get the size of A and B
    uint64_t anrows, ancols, bnrows, bncols, cnrows, cncols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;

    // get the descriptor contents to determine if A and B are transposed
    int in0, in1 ;
    OK (GrB_Descriptor_get_INT32 (desc, &in0, GrB_INP0)) ;
    OK (GrB_Descriptor_get_INT32 (desc, &in1, GrB_INP1)) ;
    bool A_transpose = (in0 == GrB_TRAN) ;
    bool B_transpose = (in1 == GrB_TRAN) ;

    // determine the size of C
    bool binop_bind1st = false ;
    if (anrows == 1 && ancols == 1)
    { 
        // C = alpha * B
        binop_bind1st = true ;
        cnrows = (B_transpose) ? bncols : bnrows ;
        cncols = (B_transpose) ? bnrows : bncols ;
        scalar = (GrB_Scalar) A ;
    }
    else if (bnrows == 1 && bncols == 1)
    { 
        // C = A * beta
        binop_bind1st = false ;
        cnrows = (A_transpose) ? ancols : anrows ;
        cncols = (A_transpose) ? anrows : ancols ;
        scalar = (GrB_Scalar) B ;
    }
    else
    { 
        // C = A * B where A and B are both matrices or vectors
        cnrows = (A_transpose) ? ancols : anrows ;
        cncols = (B_transpose) ? bnrows : bncols ;
    }

    // create the matrix C and set its format and sparsity
    OK (gb_get_format (cnrows, cncols, A, B, &(gbdesc.fmt), err)) ;
    OK (gb_get_sparsity (A, B, &(gbdesc.sparsity), err)) ;
    OK (gb_new (&C, ctype, cnrows, cncols, gbdesc.fmt, gbdesc.sparsity, arena,
        err)) ;

    //--------------------------------------------------------------------------
    // compute C = A*B
    //--------------------------------------------------------------------------

    if (scalar != NULL)
    {

        //----------------------------------------------------------------------
        // C = alpha * B or C = A * beta
        //----------------------------------------------------------------------

        uint64_t nvals ;
        OK (GrB_Scalar_nvals (&nvals, scalar)) ;
        if (nvals == 0)
        { 
            // zero = (ctype) 0
            OK (GxB_Scalar_new_arena (&zero, ctype, arena, arena)) ;
            OK (GrB_Scalar_setElement_FP64 (zero, 0)) ;
            scalar = zero ;
        }
        if (binop_bind1st)
        { 
            // C = alpha * B
            OK1 (C, GrB_Matrix_apply_BinaryOp1st_Scalar (C, NULL, NULL, times,
                scalar, B, desc)) ;
        }
        else
        { 
            // C = A * beta
            OK1 (C, GrB_Matrix_apply_BinaryOp2nd_Scalar (C, NULL, NULL, times,
                A, scalar, desc)) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C = A*B, overriding the sparsity of C for sparse*full and full*sparse
        //----------------------------------------------------------------------

        int A_sparsity, B_sparsity ;
        OK (GrB_Matrix_get_INT32 (A, &A_sparsity, GxB_SPARSITY_STATUS)) ;
        OK (GrB_Matrix_get_INT32 (B, &B_sparsity, GxB_SPARSITY_STATUS)) ;

        bool A_full = (A_sparsity == GxB_FULL) ;
        bool A_sparse = (A_sparsity == GxB_BITMAP || A_sparsity == GxB_SPARSE) ;
        bool B_full = (B_sparsity == GxB_FULL) ;
        bool B_sparse = (B_sparsity == GxB_BITMAP || B_sparsity == GxB_SPARSE) ;

        if ((A_full && B_sparse) || (A_sparse && B_full))
        { 

            //------------------------------------------------------------------
            // sparse-times-full or full-times-sparse
            //------------------------------------------------------------------

            // ensure C can be held as a full matrix
            gbdesc.sparsity = gbdesc.sparsity | GxB_FULL ;
            OK (GrB_Matrix_set_INT32 (C, gbdesc.sparsity,
                GxB_SPARSITY_CONTROL)) ;
            // C = 0
            // zero = (ctype) 0
            OK (GxB_Scalar_new_arena (&zero, ctype, arena, arena)) ;
            OK (GrB_Scalar_setElement_FP64 (zero, 0)) ;
            OK (GrB_Matrix_assign_Scalar (C, NULL, NULL, zero, GrB_ALL, cnrows,
                GrB_ALL, cncols, NULL)) ;
            // C += A*B
            OK1 (C, GrB_mxm (C, NULL, plus, plus_times, A, B, desc)) ;

        }
        else
        { 

            //------------------------------------------------------------------
            // C = A*B for everything else
            //------------------------------------------------------------------

            // If A and/or B are hypersparse, then C is not computed as full,
            // since it would likely be too large.  Instead, it is computed
            // as sparse.

            OK1 (C, GrB_mxm (C, NULL, NULL, plus_times, A, B, desc)) ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    OK (gb_export (C_opaque, &C, gbdesc.kind, ghb, err)) ;
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

