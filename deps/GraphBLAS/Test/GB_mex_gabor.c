//------------------------------------------------------------------------------
// GB_mex_gabor: test case from Gabor Szarnyas and Marton Elekes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This test triggers the C<M>=A assignment where C starts out as sparse with
// has many pending tuples, and is converted to bitmap by the assignment.  In
// this case, C is the vector w.  If w_sparsity is 15 and 'wait' is false, then
// it starts the w<v>=sum(A) reduction with many pending tuples, and converts w
// from sparse/hyper with many pending tuples into a bitmap vector.  The
// outputs w, v, and A should be the same, regardless of the input parameter s.

// s is an optional vector of length 4, containing 4 parameters:
// s = [wait, w_sparsity, v_sparsity, A_sparsity] ;
// with wait 0 or 1, and the sparsity controls in range 1 to 15.

#include "GB_mex.h"

#define USAGE "[w,v,A] = GB_mex_gabor (s)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (false) ;
    GrB_Matrix A = NULL ;
    GrB_Vector v = NULL ;
    GrB_Vector w = NULL ;

    // check inputs
    if (nargout > 3 || nargin > 1)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get the sparsity control for w, v, and A, and the wait flag
    int w_sparsity = 15 ;
    int v_sparsity = 15 ;
    int A_sparsity = 15 ;
    bool wait = false ;

    if (nargin > 0)
    {
        if (mxGetNumberOfElements (pargin [0]) != 4 || !mxIsDouble (pargin [0]))
        {
            mexErrMsgTxt ("Usage: " USAGE
                "\ns must be a double vector of length 4\n") ;
        }
        double *p = mxGetDoubles (pargin [0]) ;
        wait = (bool) p [0] ;
        w_sparsity = (int) p [1] ;
        v_sparsity = (int) p [2] ;
        A_sparsity = (int) p [3] ;
    }

    // define the problem
    uint64_t I [ ] = { 1, 2, 4, 5, 7, 11, 12, 13, 15, 18, 19, 20, 27, 32, 33,
        35, 37, 41, 46, 50, 52, 53, 55, 57, 58, 61, 62, 63, 65, 66, 69, 70, 72,
        73, 74, 75, 78, 79, 81, 84, 86, 87, 90, 91, 94, 96, 97, 98, 99, 100,
        101, 102, 103, 104, 105, 107, 108, 109, 110, 115, 116, 117, 118, 120,
        123, 129, 131, 132, 133, 134, 136, 140, 145, 146, 149, 152, 153, 154,
        156, 158, 159, 160, 161, 163, 164, 165, 166, 168, 169, 172, 176, 177,
        181, 184, 186, 187, 189, 191, 193, 194, 195, 197, 200, 201, 202, 203,
        204, 205, 208, 209, 210, 211, 216, 217, 218, 219, 224, 225, 229, 230,
        232, 235, 236, 238, 239, 242, 243 } ;

    uint64_t nvals = sizeof (I) / sizeof (uint64_t) ;
    uint64_t n = 1000 ;

    // construct a diagonal matrix A where A(i,i)=i for each i in I 
    GrB_Matrix_new (&A, GrB_UINT64, n, n) ;
    GxB_Matrix_Option_set_ (A, GxB_SPARSITY_CONTROL, A_sparsity) ;
    GrB_Matrix_build (A, I, I, I, nvals, GrB_PLUS_UINT64) ;
    if (wait) GrB_Matrix_wait (A, GrB_MATERIALIZE) ;

    // construct v from I, with value v (i) = i 
    GrB_Vector_new (&v, GrB_UINT64, n) ;
    GxB_Vector_Option_set_ (v, GxB_SPARSITY_CONTROL, v_sparsity) ;
    GrB_Vector_build (v, I, I, nvals, GrB_PLUS_UINT64) ;
    if (wait) GrB_Vector_wait (v, GrB_MATERIALIZE) ;

    // w<v> = 1
    GrB_Vector_new (&w, GrB_UINT64, n) ;
    GxB_Vector_Option_set_ (w, GxB_SPARSITY_CONTROL, w_sparsity) ;
    GrB_Vector_assign_UINT64 (w, v, NULL, 1, GrB_ALL, 0, NULL) ;
    if (wait) GrB_Vector_wait (w, GrB_MATERIALIZE) ;

    // w<v> = sum (A)
    GrB_Matrix_reduce_Monoid (w, v, NULL, GrB_PLUS_MONOID_UINT64, A, NULL) ;

    // return A, v, and w as structs
    pargout [0] = GB_mx_Vector_to_mxArray (&w, "w output", true) ;
    pargout [1] = GB_mx_Vector_to_mxArray (&v, "v output", true) ;
    pargout [2] = GB_mx_Matrix_to_mxArray (&A, "A output", true) ;

    // log the test coverage
    GB_mx_put_global (true) ;
}

