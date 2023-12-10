//------------------------------------------------------------------------------
// GraphBLAS/Test/GB_mx_random_matrix.c: create a random matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Creates a random sparse matrix, using either setElement or build.

#include "GB_mex.h"

#define OK(method)                                                          \
{                                                                           \
    info = method ;                                                         \
    if (! (info == GrB_SUCCESS || info == GrB_NO_VALUE))                    \
    {                                                                       \
        FREE_ALL ;                                                          \
        return (info) ;                                                     \
    }                                                                       \
}

#define FREE_ALL                    \
    GrB_Matrix_free (&A) ;          \
    GrB_Matrix_free (&Areal) ;      \
    GrB_Matrix_free (&Aimag) ;      \
    if (I != NULL) mxFree (I) ;     \
    if (J != NULL) mxFree (J) ;     \
    if (X != NULL) mxFree (X) ;

//------------------------------------------------------------------------------
// create a random matrix
//------------------------------------------------------------------------------

GrB_Info GB_mx_random_matrix      // create a random double-precision matrix
(
    GrB_Matrix *A_output,   // handle of matrix to create
    bool make_symmetric,    // if true, return A as symmetric
    bool no_self_edges,     // if true, then do not create self edges
    int64_t nrows,          // number of rows
    int64_t ncols,          // number of columns
    int64_t nedges,         // number of edges
    int method,             // method to use: 0:setElement, 1:build,
    bool A_complex          // if true, create a Complex matrix
)
{

    GrB_Matrix Areal = NULL, Aimag = NULL, A = NULL ;
    *A_output = NULL ;
    GrB_Index *I = NULL, *J = NULL ;
    double *X = NULL ;
    GrB_Info info ;

    if (make_symmetric)
    {
        nrows = GB_IMAX (nrows, ncols) ;
        ncols = GB_IMAX (nrows, ncols) ;
    }

    //--------------------------------------------------------------------------
    // create a Complex matrix
    //--------------------------------------------------------------------------

    if (A_complex)
    {
        // Areal = real random matrix
        OK (GB_mx_random_matrix (&Areal, make_symmetric, no_self_edges, nrows,
            ncols, nedges, method, false)) ;
        // Aimag = real random matrix
        OK (GB_mx_random_matrix (&Aimag, make_symmetric, no_self_edges, nrows,
            ncols, nedges, method, false)) ;
        // A = Areal + imag(Aimag)
        OK (GrB_Matrix_new (&A, Complex, nrows, ncols)) ;
        OK (GxB_Matrix_Option_set (A, GxB_FORMAT, GxB_BY_COL)) ;
        OK (GrB_Matrix_apply (A, NULL, NULL,         Complex_complex_real,
            Areal, NULL)) ;
        OK (GrB_Matrix_apply (A, NULL, Complex_plus, Complex_complex_imag,
            Aimag, NULL)) ;
        *A_output = A ;
        A = NULL ;
        FREE_ALL ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // create a real double matrix (GrB_FP64)
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_FP64, nrows, ncols)) ;
    OK (GxB_Matrix_Option_set (A, GxB_FORMAT, GxB_BY_COL)) ;

    if (method == 0)
    {

        //----------------------------------------------------------------------
        // use GrB_Matrix_setElement: no need to allocate tuples
        //----------------------------------------------------------------------

        // This is just about as fast as the GrB_Matrix_build method with
        // non-blocking mode (about 10% more time, regardless of the problem
        // size).  This is mainly because setElement doesn't know how many
        // tuples will eventually be added, so it must dynamically reallocate
        // its internal storage.  In constrast, the arrays I, J, and X are a
        // fixed, known size and are not reallocated as tuples are added.

        // Note how simple this code is, below.  A user application can use
        // setElement without having to create its own I,J,X tuple lists.  It
        // can create the tuples in any order.  The code is simpler, and the
        // performance penalty is neglible.

        // With blocking mode, setElement is EXTREMELY slow, since the matrix
        // is rebuilt on every iteration.  In this case, it is easily a 1,000
        // or even a million times slower than using build when the matrix is
        // very large.  Don't attempt to do this with large matrices with
        // blocking mode enabled.  Actual run time could increase from 1 minute
        // to 1 year (!) in the extreme case, with a matrix that can be
        // generated on a commodity laptop.

        for (int64_t k = 0 ; k < nedges ; k++)
        {
            GrB_Index i = simple_rand_i ( ) % nrows ;
            GrB_Index j = simple_rand_i ( ) % ncols ;
            if (no_self_edges && (i == j)) continue ;
            double x = simple_rand_x ( ) ;
            // A (i,j) = x
            OK (GrB_Matrix_setElement_FP64 (A, x, i, j)) ;
            if (make_symmetric)
            {
                // A (j,i) = x
                OK (GrB_Matrix_setElement_FP64 (A, x, j, i)) ;
            }
        }
    }
    else
    {

        //----------------------------------------------------------------------
        // use GrB_Matrix_build: allocate initial space for tuples
        //----------------------------------------------------------------------

        // This method is harder for a user application to use.  It is slightly
        // faster than the setElement method.  Its performance is not affected
        // by the mode (blocking or non-blocking).

        int64_t s = ((make_symmetric) ? 2 : 1) * nedges + 1 ;
        I = (GrB_Index *) mxMalloc (s * sizeof (GrB_Index)) ;
        J = (GrB_Index *) mxMalloc (s * sizeof (GrB_Index)) ;
        X = (double *) mxMalloc (s * sizeof (double   )) ;
        if (I == NULL || J == NULL || X == NULL)
        {   // out of memory
            FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // create the tuples
        //----------------------------------------------------------------------

        int64_t ntuples = 0 ;
        for (int64_t k = 0 ; k < nedges ; k++)
        {
            GrB_Index i = simple_rand_i ( ) % nrows ;
            GrB_Index j = simple_rand_i ( ) % ncols ;
            if (no_self_edges && (i == j)) continue ;
            double x = simple_rand_x ( ) ;
            // A (i,j) = x
            I [ntuples] = i ;
            J [ntuples] = j ;
            X [ntuples] = x ;
            ntuples++ ;
            if (make_symmetric)
            {
                // A (j,i) = x
                I [ntuples] = j ;
                J [ntuples] = i ;
                X [ntuples] = x ;
                ntuples++ ;
            }
        }

        //----------------------------------------------------------------------
        // build the matrix
        //----------------------------------------------------------------------

        OK (GrB_Matrix_build_FP64 (A, I, J, X, ntuples, GrB_SECOND_FP64)) ;
        mxFree (I) ;
        mxFree (J) ;
        mxFree (X) ;
    }

    *A_output = A ;
    return (GrB_SUCCESS) ;
}

