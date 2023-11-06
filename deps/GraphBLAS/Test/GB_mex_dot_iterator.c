//------------------------------------------------------------------------------
// GB_mex_dot_iterator: s = X'*Y, dot product of 2 vectors using iterators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// X and Y must have the same type.  If boolean, the lor_land semiring is used.
// Otherwise, the plus_times semiring is used for the given type.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "s = GB_mex_dot_iterator (X, Y, kind)"

#define FREE_ALL                                    \
{                                                   \
    GrB_Vector_free_(&X) ;                          \
    GrB_Vector_free_(&Y) ;                          \
    GxB_Iterator_free (&X_iterator) ;               \
    GxB_Iterator_free (&Y_iterator) ;               \
    GB_mx_put_global (true) ;                       \
}

#define Assert(x)                                   \
{                                                   \
    if (!(x))                                       \
    {                                               \
        printf ("Failure at %d\n", __LINE__) ;      \
        mexErrMsgTxt ("fail") ;                     \
    }                                               \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Vector Y = NULL, X = NULL ;
    GrB_Matrix A = NULL ;
    GxB_Iterator X_iterator = NULL, Y_iterator ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get X (shallow copy)
    X = GB_mx_mxArray_to_Vector (pargin [0], "X input", false, true) ;
    if (X == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("X failed") ;
    }

    // get Y (shallow copy)
    Y = GB_mx_mxArray_to_Vector (pargin [1], "Y input", false, true) ;
    if (Y == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("Y failed") ;
    }

    // get kind:
    // 0: merge, using macros
    // 1: iterate through x and lookup y, using macros
    // 2: merge, using functions
    // 3: iterate through x and lookup y, using functions
    int GET_SCALAR (2, int, kind, 0) ;
    bool use_macros = (kind <= 1) ;
    kind = kind % 2 ;

    GrB_Index n, ny ;
    OK (GrB_Vector_size (&n, X)) ;
    OK (GrB_Vector_size (&ny, Y)) ;

    GB_Global_print_one_based_set (0) ;

    if (n != ny)
    {
        FREE_ALL ;
        mexErrMsgTxt ("X and Y must have the same size") ;
    }

    GrB_Type type = X->type ;
    if (type != Y->type)
    {
        FREE_ALL ;
        mexErrMsgTxt ("X and Y must have the same type") ;
    }

//  bool user_complex = (Complex != GxB_FC64) && (type == Complex) ;

    // create the output scalar and set it to zero
    pargout [0] = GB_mx_create_full (1, 1, type) ;
    GB_void *s = mxGetData (pargout [0]) ;
    memset (s, 0, type->size) ;

    //--------------------------------------------------------------------------
    // s += X'*Y using vector iterators
    //--------------------------------------------------------------------------

    // create the X and Y iterators
    OK (GxB_Iterator_new (&X_iterator)) ;
    OK (GxB_Iterator_new (&Y_iterator)) ;

    GrB_Info X_info = GxB_Vector_Iterator_attach (X_iterator, X, NULL) ;
    GrB_Info Y_info = GxB_Vector_Iterator_attach (Y_iterator, Y, NULL) ;
    OK (X_info) ;
    OK (Y_info) ;

    int x_sparsity, y_sparsity ;
    OK (GxB_Vector_Option_get (X, GxB_SPARSITY_STATUS, &x_sparsity)) ;
    OK (GxB_Vector_Option_get (Y, GxB_SPARSITY_STATUS, &y_sparsity)) ;
    GrB_Index xnvals, ynvals ;
    OK (GrB_Vector_nvals (&xnvals, X)) ;
    OK (GrB_Vector_nvals (&ynvals, Y)) ;

//  if (kind == 0) { GxB_print (X, 3) ; GxB_print (Y, 3) ; }

    if (X->b != NULL && X->type == GrB_FP64)
    {
        // mangle the X vector where entries are not present
        double *Xx = (double *) X->x ;
        int8_t *Xb = X->b ;
        int64_t n = X->vlen ;
        for (int64_t k = 0 ; k < n ; k++)
        {
            if (!Xb [k]) Xx [k] = 42 ;
        }
    }

    if (Y->b != NULL && Y->type == GrB_FP64)
    {
        // mangle the Y vector where entries are not present
        double *Yx = (double *) Y->x ;
        int8_t *Yb = Y->b ;
        int64_t n = Y->vlen ;
        for (int64_t k = 0 ; k < n ; k++)
        {
            if (!Yb [k]) Yx [k] = 42 ;
        }
    }

    if (use_macros)
    {
        // use macros that are #define'd in GraphBLAS.h
        #include "Template/GB_mx_dot_iterator_template.c"
    }
    else
    {
        // use functions whose prototypes are in GraphBLAS.h
        #include "GB_undef_iterator.h"
        #include "Template/GB_mx_dot_iterator_template.c"
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    GB_Global_print_one_based_set (1) ;
    FREE_ALL ;
}

