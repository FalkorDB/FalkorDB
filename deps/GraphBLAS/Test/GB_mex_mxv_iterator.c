//------------------------------------------------------------------------------
// GB_mex_mxv_iterator: Y = A*X using an iterator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A and X must have the same type.  If boolean, the lor_land semiring is used.
// Otherwise, the plus_times semiring is used for the given type.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "Y = GB_mex_mxv_iterator (A, X, kind)"

#define FREE_ALL                                    \
{                                                   \
    GrB_Vector_free_(&X) ;                          \
    GrB_Vector_free_(&Y) ;                          \
    GrB_Matrix_free_(&A) ;                          \
    GxB_Iterator_free (&iterator) ;                 \
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
    GxB_Iterator iterator = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    GB_Global_print_one_based_set (0) ;

    // get X (shallow copy)
    X = GB_mx_mxArray_to_Vector (pargin [1], "X input", false, true) ;
    if (X == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("X failed") ;
    }
    // GxB_print (X, 3) ;

    GrB_Index n ;
    OK (GrB_Vector_size (&n, X)) ;
    if (n != ncols)
    {
        FREE_ALL ;
        mexErrMsgTxt ("X has the wrong size") ;
    }

    GrB_Type type = A->type ;
    if (type != X->type)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A and X must have the same type") ;
    }

    // get kind
    int GET_SCALAR (2, int, kind, 0) ;
    bool use_macros = (kind <= 7) ;
    kind = kind % 8 ;

    // make sure X is full
    int sparsity ;
    OK (GxB_Vector_Option_get (X, GxB_SPARSITY_STATUS, &sparsity)) ;
    if (sparsity != GxB_FULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("X must be full") ;
    }

    bool user_complex = (Complex != GxB_FC64) && (type == Complex) ;

    // create Y and make it full: all zero but non-iso
    OK (GrB_Vector_new (&Y, type, nrows)) ;
    if (user_complex)
    {
        double z [2] ;
        z [0] = 0 ;
        z [1] = 0 ;
        OK (GrB_Vector_assign_UDT (Y, NULL, NULL, z, GrB_ALL, nrows, NULL)) ;
        z [0] = 1 ;
        OK (GrB_Vector_setElement_UDT (Y, z, 0)) ;
        z [0] = 0 ;
        OK (GrB_Vector_setElement_UDT (Y, z, 0)) ;
    }
    else
    {
        OK (GrB_Vector_assign_INT32 (Y, NULL, NULL, 0, GrB_ALL, nrows, NULL)) ;
        OK (GrB_Vector_setElement_INT32 (Y, 1, 0)) ;
        OK (GrB_Vector_setElement_INT32 (Y, 0, 0)) ;
    }
    OK (GrB_Vector_wait (Y, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // Y += A*X using an iterator
    //--------------------------------------------------------------------------

    // create an iterator
    OK (GxB_Iterator_new (&iterator)) ;

    if (use_macros)
    {
        // use macros that are #define'd in GraphBLAS.h
        #include "Template/GB_mx_mxv_iterator_template.c"
    }
    else
    {
        // use functions whose prototypes are in GraphBLAS.h
        #include "GB_undef_iterator.h"
        #include "Template/GB_mx_mxv_iterator_template.c"
    }

    //--------------------------------------------------------------------------
    // error handling
    //--------------------------------------------------------------------------

    if (A->is_csc)
    {
        info = GxB_rowIterator_attach (iterator, A, NULL) ;
        Assert (info == GrB_NOT_IMPLEMENTED) ;
    }
    else
    {
        info = GxB_colIterator_attach (iterator, A, NULL) ;
        Assert (info == GrB_NOT_IMPLEMENTED) ;
    }

    //--------------------------------------------------------------------------
    // return Y as a struct and free the GraphBLAS Y
    //--------------------------------------------------------------------------

    // GxB_print (Y, 3) ;
    GB_Global_print_one_based_set (1) ;
    pargout [0] = GB_mx_Vector_to_mxArray (&Y, "Y output", true) ;
    FREE_ALL ;
}

