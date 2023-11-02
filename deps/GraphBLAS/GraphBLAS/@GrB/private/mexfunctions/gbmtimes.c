//------------------------------------------------------------------------------
// gbmtimes: sparse matrix-matrix multiplication over the standard semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmtimes provides the mexFunction for computing the overloaded method C =
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

// C = gbmtimes (A, B)
// C = gbmtimes (A, B, desc)

#include "gb_interface.h"

#define USAGE "usage: C = gbmtimes (A, B, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin >= 2 && nargin <= 3 && nargout <= 2, USAGE) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    mxArray *Matrix [6], *String [2], *Cell [2] ;
    base_enum_t base ;
    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    int nmatrices, nstrings, ncells, sparsity ;
    GrB_Descriptor desc ;
    gb_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String, &nstrings,
        Cell, &ncells, &desc, &base, &kind, &fmt, &sparsity) ;

    CHECK_ERROR (nmatrices != 2 || nstrings > 0 || ncells > 0, USAGE) ;

    // ensure the descriptor is present, and set GxB_SORT to true
    if (desc == NULL)
    { 
        OK (GrB_Descriptor_new (&desc)) ;
    }
    OK (GxB_Desc_set (desc, GxB_SORT, true)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    GrB_Type atype, btype, ctype ;
    GrB_Matrix C = NULL, A, B ;
    A = gb_get_shallow (Matrix [0]) ;
    B = gb_get_shallow (Matrix [1]) ;
    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GxB_Matrix_type (&btype, B)) ;

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp plus = NULL, times = NULL ;
    GrB_Monoid plus_monoid = NULL ;
    GrB_Semiring plus_times = NULL ;
    char semiring_string [8] ;
    strcpy (semiring_string, "+.*") ;
    plus_times = gb_string_to_semiring (semiring_string, atype, btype) ;
    OK (GxB_Semiring_add (&plus_monoid, plus_times)) ;
    OK (GxB_Semiring_multiply (&times, plus_times)) ;
    OK (GxB_Monoid_operator (&plus, plus_monoid)) ;
    OK (GxB_BinaryOp_ztype (&ctype, plus)) ;

    //--------------------------------------------------------------------------
    // construct C
    //--------------------------------------------------------------------------

    // get the size of A and B
    GrB_Index anrows, ancols, bnrows, bncols, cnrows, cncols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;

    // get the descriptor contents to determine if A and B are transposed
    GrB_Desc_Value in0, in1 ;
    OK (GxB_Desc_get (desc, GrB_INP0, &in0)) ;
    OK (GxB_Desc_get (desc, GrB_INP1, &in1)) ;
    bool A_transpose = (in0 == GrB_TRAN) ;
    bool B_transpose = (in1 == GrB_TRAN) ;

    // determine the size of C
    GrB_Scalar scalar = NULL, zero = NULL ;
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
    fmt = gb_get_format (cnrows, cncols, A, B, fmt) ;
    sparsity = gb_get_sparsity (A, B, sparsity) ;
    C = gb_new (ctype, cnrows, cncols, fmt, sparsity) ;

    //--------------------------------------------------------------------------
    // compute C = A*B
    //--------------------------------------------------------------------------

    if (scalar != NULL)
    {

        //----------------------------------------------------------------------
        // C = alpha * B or C = A * beta
        //----------------------------------------------------------------------

        GrB_Index nvals ;
        OK (GrB_Scalar_nvals (&nvals, scalar)) ;
        if (nvals == 0)
        {
            // zero = (ctype) 0
            OK (GrB_Scalar_new (&zero, ctype)) ;
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
        OK (GxB_Matrix_Option_get (A, GxB_SPARSITY_STATUS, &A_sparsity)) ;
        OK (GxB_Matrix_Option_get (B, GxB_SPARSITY_STATUS, &B_sparsity)) ;

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
            sparsity = sparsity | GxB_FULL ;
            OK (GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, sparsity)) ;
            // C = 0
            // zero = (ctype) 0
            OK (GrB_Scalar_new (&zero, ctype)) ;
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
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&B)) ;
    OK (GrB_Scalar_free (&zero)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    GB_WRAPUP ;
}

