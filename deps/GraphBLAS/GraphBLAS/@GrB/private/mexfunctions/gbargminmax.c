//------------------------------------------------------------------------------
// gbargminmax: argmin or argmax of a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:

// [x,p] = gbargminmax (A, minmax, dim)

// where minmax is 0 for min or 1 for max, and where dim = 1 to compute the
// argmin/max of each column of A, dim = 2 to compute the argmin/max of each
// row of A, or dim = 0 to compute the argmin/max of all of A.  For dim = 1 or
// 2, x and p are vectors of the same size.  For dim = 0, x is a scalar and p
// is 2-by-1, containing the row and column index of the argmin/max of A.

#include "gb_interface.h"

#define USAGE "usage: [x,p] = gbargminmax (A, minmax, dim)"

//------------------------------------------------------------------------------
// gb_argminmax: compute argmin/max of each row/column of A
//------------------------------------------------------------------------------

static void gb_argminmax
(
    // output
    GrB_Matrix *x,              // min/max value in each row/col of A
    GrB_Matrix *p,              // index of min/max value in each row/col of A
    // input
    GrB_Matrix A,
    int dim,                    // dim=1: cols of A, dim=2: rows of A
    GrB_Semiring minmax_first,  // MIN_FIRST_type or MAX_FIRST_type semiring
    GrB_Semiring any_equal      // ANY_EQ semiring
)
{

    //--------------------------------------------------------------------------
    // get the size and type of A
    //--------------------------------------------------------------------------

    (*x) = NULL ;
    (*p) = NULL ;

    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    GrB_Type type ;
    OK (GxB_Matrix_type (&type, A)) ;

    //--------------------------------------------------------------------------
    // create outputs x and p, and the iso full vector y
    //--------------------------------------------------------------------------

    GrB_Matrix y = NULL ;
    GrB_Matrix G = NULL, D = NULL ;
    GrB_Index n = (dim == 2) ? ncols : nrows ;
    GrB_Index m = (dim == 2) ? nrows : ncols ;
    GrB_Descriptor desc = (dim == 2) ? NULL : GrB_DESC_T0 ;
    OK (GrB_Matrix_new (x, type, m, 1)) ;
    OK (GrB_Matrix_new (&y, type, n, 1)) ;
    OK (GrB_Matrix_new (p, GrB_INT64, m, 1)) ;

    // y (:) = 1, an iso full vector
    OK (GrB_Matrix_assign_INT64 (y, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, 1,
        NULL)) ;

    //--------------------------------------------------------------------------
    // compute x = min/max(A)
    //--------------------------------------------------------------------------

    // for dim=1: x = min/max (A) where x(j) = min/max (A (:,j))
    // for dim=2: x = min/max (A) where x(i) = min/max (A (i,:))

    OK (GrB_mxm (*x, NULL, NULL, minmax_first, A, y, desc)) ;

    //--------------------------------------------------------------------------
    // D = diag (x)
    //--------------------------------------------------------------------------

    // note: typecasting from an m-by-1 GrB_Matrix to a GrB_Vector is
    // not allowed by the GraphBLAS C API, but it can be done in SuiteSparse.
    // A more portable method would construct x as a GrB_Vector,
    // but using x as a GrB_Matrix simplifies the gb_export.

    OK (GrB_Matrix_diag (&D, (GrB_Vector) *x, 0)) ;

    //--------------------------------------------------------------------------
    // compute G, where G(i,j)=1 if A(i,j) is the min/max in its row/col
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&G, GrB_BOOL, nrows, ncols)) ;
    if (dim == 1)
    {
        // G = A*D using the ANY_EQ_type semiring
        OK (GrB_mxm (G, NULL, NULL, any_equal, A, D, NULL)) ;
    }
    else
    {
        // G = D*A using the ANY_EQ_type semiring
        OK (GrB_mxm (G, NULL, NULL, any_equal, D, A, NULL)) ;
    }

    // drop explicit zeros from G
//  OK (GxB_Matrix_select (G, NULL, NULL, GxB_NONZERO, G, NULL, NULL)) ;
    OK (GrB_Matrix_select_BOOL (G, NULL, NULL, GrB_VALUENE_BOOL, G, 0, NULL)) ;

    //--------------------------------------------------------------------------
    // extract the positions of the entries in G
    //--------------------------------------------------------------------------

    // for dim=1: find the position of the min/max entry in each column:
    // p = G'*y, so that p(j) = i if x(j) = A(i,j) = min/max (A (:,j)).

    // for dim=2: find the position of the min/max entry in each row:
    // p = G*y, so that p(i) = j if x(i) = A(i,j) = min/max (A (i,:)).

    // For both cases, use the SECONDI1 operator since built-in indexing is
    // 1-based.  The ANY monoid would be faster, but this uses MIN so that the
    // result for the user is repeatable.
    OK (GrB_mxm (*p, NULL, NULL, GxB_MIN_SECONDI1_INT64, G, y, desc)) ;

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    GrB_Matrix_free (&D) ;
    GrB_Matrix_free (&G) ;
    GrB_Matrix_free (&y) ;
}

//------------------------------------------------------------------------------
// gbargminmax: mexFunction to compute the argmin/max of each row/column of A
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
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin == 3 && nargout == 2, USAGE) ;
    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    bool is_min = (bool) (mxGetScalar (pargin [1]) == 0) ;
    int dim = (int) mxGetScalar (pargin [2]) ;
    CHECK_ERROR (dim < 0 || dim > 2, "invalid dim") ;

    //--------------------------------------------------------------------------
    // select the semirings
    //--------------------------------------------------------------------------

    GrB_Type type ;
    OK (GxB_Matrix_type (&type, A)) ;
    GrB_Semiring minmax_first, any_equal ;

    if (is_min)
    {

        //----------------------------------------------------------------------
        // semirings for argmin
        //----------------------------------------------------------------------

        if (type == GrB_BOOL)
        {
            minmax_first = GxB_LAND_FIRST_BOOL ;
            any_equal = GxB_ANY_EQ_BOOL ;
        }
        else if (type == GrB_INT8)
        {
            minmax_first = GrB_MIN_FIRST_SEMIRING_INT8 ;
            any_equal = GxB_ANY_EQ_INT8 ;
        }
        else if (type == GrB_INT16)
        {
            minmax_first = GrB_MIN_FIRST_SEMIRING_INT16 ;
            any_equal = GxB_ANY_EQ_INT16 ;
        }
        else if (type == GrB_INT32) 
        {
            minmax_first = GrB_MIN_FIRST_SEMIRING_INT32 ;
            any_equal = GxB_ANY_EQ_INT32 ;
        }
        else if (type == GrB_INT64) 
        {
            minmax_first = GrB_MIN_FIRST_SEMIRING_INT64 ;
            any_equal = GxB_ANY_EQ_INT64 ;
        }
        else if (type == GrB_UINT8) 
        {
            minmax_first = GrB_MIN_FIRST_SEMIRING_UINT8 ;
            any_equal = GxB_ANY_EQ_UINT8 ;
        }
        else if (type == GrB_UINT16)
        {
            minmax_first = GrB_MIN_FIRST_SEMIRING_UINT16 ;
            any_equal = GxB_ANY_EQ_UINT16 ;
        }
        else if (type == GrB_UINT32)
        {
            minmax_first = GrB_MIN_FIRST_SEMIRING_UINT32 ;
            any_equal = GxB_ANY_EQ_UINT32 ;
        }
        else if (type == GrB_UINT64)
        {
            minmax_first = GrB_MIN_FIRST_SEMIRING_UINT64 ;
            any_equal = GxB_ANY_EQ_UINT64 ;
        }
        else if (type == GrB_FP32)
        {
            minmax_first = GrB_MIN_FIRST_SEMIRING_FP32 ;
            any_equal = GxB_ANY_EQ_FP32 ;
        }
        else if (type == GrB_FP64)
        {
            minmax_first = GrB_MIN_FIRST_SEMIRING_FP64 ;
            any_equal = GxB_ANY_EQ_FP64 ;
        }
        else
        {
            ERROR ("unsupported type") ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // semirings for argmax
        //----------------------------------------------------------------------

        if (type == GrB_BOOL)
        {
            minmax_first = GxB_LOR_FIRST_BOOL ;
            any_equal = GxB_ANY_EQ_BOOL ;
        }
        else if (type == GrB_INT8)
        {
            minmax_first = GrB_MAX_FIRST_SEMIRING_INT8 ;
            any_equal = GxB_ANY_EQ_INT8 ;
        }
        else if (type == GrB_INT16)
        {
            minmax_first = GrB_MAX_FIRST_SEMIRING_INT16 ;
            any_equal = GxB_ANY_EQ_INT16 ;
        }
        else if (type == GrB_INT32) 
        {
            minmax_first = GrB_MAX_FIRST_SEMIRING_INT32 ;
            any_equal = GxB_ANY_EQ_INT32 ;
        }
        else if (type == GrB_INT64) 
        {
            minmax_first = GrB_MAX_FIRST_SEMIRING_INT64 ;
            any_equal = GxB_ANY_EQ_INT64 ;
        }
        else if (type == GrB_UINT8) 
        {
            minmax_first = GrB_MAX_FIRST_SEMIRING_UINT8 ;
            any_equal = GxB_ANY_EQ_UINT8 ;
        }
        else if (type == GrB_UINT16)
        {
            minmax_first = GrB_MAX_FIRST_SEMIRING_UINT16 ;
            any_equal = GxB_ANY_EQ_UINT16 ;
        }
        else if (type == GrB_UINT32)
        {
            minmax_first = GrB_MAX_FIRST_SEMIRING_UINT32 ;
            any_equal = GxB_ANY_EQ_UINT32 ;
        }
        else if (type == GrB_UINT64)
        {
            minmax_first = GrB_MAX_FIRST_SEMIRING_UINT64 ;
            any_equal = GxB_ANY_EQ_UINT64 ;
        }
        else if (type == GrB_FP32)
        {
            minmax_first = GrB_MAX_FIRST_SEMIRING_FP32 ;
            any_equal = GxB_ANY_EQ_FP32 ;
        }
        else if (type == GrB_FP64)
        {
            minmax_first = GrB_MAX_FIRST_SEMIRING_FP64 ;
            any_equal = GxB_ANY_EQ_FP64 ;
        }
        else
        {
            ERROR ("unsupported type") ;
        }
    }

    //--------------------------------------------------------------------------
    // compute the argmin/max
    //--------------------------------------------------------------------------

    GrB_Matrix x = NULL, p = NULL ;

    if (dim == 0)
    {

        //----------------------------------------------------------------------
        // scalar argmin/max of all of A
        //----------------------------------------------------------------------

        // [x1,p1] = argmin/max of each column of A
        GrB_Matrix x1, p1 ;
        gb_argminmax (&x1, &p1, A, 1, minmax_first, any_equal) ;

        // [x,p] = argmin/max of each entry in x
        gb_argminmax (&x, &p, x1, 1, minmax_first, any_equal) ;

        // get the row and column index of the overall argmin/max of A
        int64_t I [2] = { 0, 0 } ;
        GrB_Index nvals0, nvals1 ;
        OK (GrB_Matrix_nvals (&nvals0, p)) ;
        OK (GrB_Matrix_nvals (&nvals1, p1)) ;
        if (nvals0 > 0 && nvals1 > 0)
        {
            // I [0] = p [0], the row index of the global argmin/max of A
            OK (GrB_Matrix_extractElement_INT64 (&(I [0]), p, 0, 0)) ;
            // I [1] = p [I [0]-1] (use -1 since I[0] is 1-based),
            // which is the column index of the global argmin/max of A
            OK (GrB_Matrix_extractElement_INT64 (&(I [1]), p1, I [0] - 1, 0)) ;
        }

        // free workspace and create p = [row, col] 
        OK (GrB_Matrix_free (&x1)) ;
        OK (GrB_Matrix_free (&p1)) ;
        OK (GrB_Matrix_free (&p)) ;
        OK (GrB_Matrix_new (&p, GrB_INT64, 2,1)) ;
        if (nvals0 > 0 && nvals1 > 0)
        {
            OK (GrB_Matrix_setElement_INT64 (p, I [1], 0, 0)) ;
            OK (GrB_Matrix_setElement_INT64 (p, I [0], 1, 0)) ;
        }

    }
    else if (dim == 1)
    {

        //----------------------------------------------------------------------
        // argmin/max of each column of A
        //----------------------------------------------------------------------

        gb_argminmax (&x, &p, A, 1, minmax_first, any_equal) ;

    }
    else
    {


        //----------------------------------------------------------------------
        // argmin/max of each row of A
        //----------------------------------------------------------------------

        gb_argminmax (&x, &p, A, 2, minmax_first, any_equal) ;

    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    pargout [0] = gb_export (&x, KIND_GRB) ;
    pargout [1] = gb_export (&p, KIND_GRB) ;
    GB_WRAPUP ;
}

