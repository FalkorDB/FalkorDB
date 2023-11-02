//------------------------------------------------------------------------------
// GB_mex_band: C = tril (triu (A,lo), hi), or with A'
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Apply a select operator to a matrix

#include "GB_mex.h"

#define USAGE "C = GB_mex_band (A, lo, hi, atranspose)"

#define FREE_ALL                        \
{                                       \
    GrB_Scalar_free_(&Thunk) ;          \
    GrB_Matrix_free_(&C) ;              \
    GrB_Matrix_free_(&A) ;              \
    GrB_Scalar_free_(&Thunk_type) ;     \
    GrB_IndexUnaryOp_free_(&op) ;       \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
}

#define OK(method)                                      \
{                                                       \
    info = method ;                                     \
    if (info != GrB_SUCCESS)                            \
    {                                                   \
        FREE_ALL ;                                      \
        mexErrMsgTxt ("GraphBLAS failed") ;             \
    }                                                   \
}

 typedef struct { int64_t lo ; int64_t hi ; } LoHi_type ; 

#define LOHI_DEFN                                       \
"typedef struct { int64_t lo ; int64_t hi ; } LoHi_type ;"

void LoHi_band (bool *z, /* x is unused: */ const void *x,
    GrB_Index i, GrB_Index j, const LoHi_type *thunk) ;

void LoHi_band (bool *z, /* x is unused: */ const void *x,
    GrB_Index i, GrB_Index j, const LoHi_type *thunk)
{
    int64_t i2 = (int64_t) i ;
    int64_t j2 = (int64_t) j ;
    (*z) = ((thunk->lo <= (j2-i2)) && ((j2-i2) <= thunk->hi)) ;
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix C = NULL ;
    GrB_Matrix A = NULL ;
    GrB_IndexUnaryOp op = NULL ;
    GrB_Info info ;
    GrB_Descriptor desc = NULL ;
    GrB_Scalar Thunk = NULL ;
    GrB_Type Thunk_type = NULL ;

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // check inputs
    if (nargout > 1 || nargin < 3 || nargin > 4)
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

    // create the Thunk
    LoHi_type bandwidth  ;
    OK (GxB_Type_new (&Thunk_type, sizeof (LoHi_type),
        "LoHi_type", LOHI_DEFN)) ;

    // get lo and hi
    bandwidth.lo = (int64_t) mxGetScalar (pargin [1]) ;
    bandwidth.hi = (int64_t) mxGetScalar (pargin [2]) ;

    OK (GrB_Scalar_new (&Thunk, Thunk_type)) ;
    OK (GrB_Scalar_setElement_UDT (Thunk, (void *) &bandwidth)) ;
    OK (GrB_Scalar_wait_(Thunk, GrB_MATERIALIZE)) ;

    // get atranspose
    bool atranspose = false ;
    if (nargin > 3) atranspose = (bool) mxGetScalar (pargin [3]) ;
    if (atranspose)
    {
        OK (GrB_Descriptor_new (&desc)) ;
        OK (GxB_Desc_set (desc, GrB_INP0, GrB_TRAN)) ;
    }

    // create operator
    // use the user-defined operator, from the LoHi_band function.
    // This operator cannot be JIT'd because it doesn't have a name or defn.
    METHOD (GrB_IndexUnaryOp_new (&op, (GxB_index_unary_function) LoHi_band,
        GrB_BOOL, GrB_FP64, Thunk_type)) ;

    GrB_Index nrows, ncols ;
    GrB_Matrix_nrows (&nrows, A) ;
    GrB_Matrix_ncols (&ncols, A) ;
    if (bandwidth.lo == 0 && bandwidth.hi == 0 && nrows == 10 && ncols == 10)
    {
        GxB_IndexUnaryOp_fprint (op, "lohi_op", 3, NULL) ;
    }

    // create result matrix C
    if (atranspose)
    {
        OK (GrB_Matrix_new (&C, GrB_FP64, A->vdim, A->vlen)) ;
    }
    else
    {
        OK (GrB_Matrix_new (&C, GrB_FP64, A->vlen, A->vdim)) ;
    }

    // C<Mask> = accum(C,op(A))
    if (GB_NCOLS (C) == 1 && !atranspose)
    {
        // this is just to test the Vector version
        OK (GrB_Vector_select_Scalar ((GrB_Vector) C, NULL, NULL, op,
            (GrB_Vector) A, Thunk, NULL)) ;
    }
    else
    {
        OK (GrB_Matrix_select_Scalar (C, NULL, NULL, op, A, Thunk, desc)) ;
    }

    // return C as a sparse matrix and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", false) ;

    FREE_ALL ;
}

