//------------------------------------------------------------------------------
// GB_mex_select_idxunop: C<Mask> = accum(C,select(A,y)) or select(A',y)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_select with an index unary operator, thunk is y

#include "GB_mex.h"

#define USAGE "C = GB_mex_select_idxunop (C, Mask, accum, op, how, A, y, desc)"

// if how == 0: use the GrB_Scalar and GrB_Matrix/Vector_select_Scalar
// if how == 1: use the C scalar   and GrB_Matrix/Vector_select_T

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&C) ;              \
    GrB_Matrix_free_(&Mask) ;           \
    GrB_Matrix_free_(&S) ;              \
    GrB_Matrix_free_(&A) ;              \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
}

GrB_Matrix C = NULL, S = NULL ;
GrB_Scalar scalar = NULL ;
GrB_Matrix Mask = NULL ;
GrB_Matrix A = NULL ;
GrB_Descriptor desc = NULL ;
GrB_BinaryOp accum = NULL ;
GrB_IndexUnaryOp op = NULL ;
GrB_Info select_idxunop (bool is_matrix) ;
int how = 0 ;

//------------------------------------------------------------------------------

GrB_Info select_idxunop (bool is_matrix)
{
    GrB_Info info ;
    GrB_Type stype ;
    GxB_Scalar_type (&stype, scalar) ;

    if (is_matrix && how == 1)
    {
        if (stype == GrB_BOOL)
        {
            bool y = *((bool *) (scalar->x)) ;  // OK
            info = GrB_Matrix_select_BOOL_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GrB_INT8)
        {
            int8_t y = *((int8_t *) (scalar->x)) ;  // OK
            info = GrB_Matrix_select_INT8_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GrB_INT16)
        {
            int16_t y = *((int16_t *) (scalar->x)) ;    // OK
            info = GrB_Matrix_select_INT16_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GrB_INT32)
        {
            int32_t y = *((int32_t *) (scalar->x)) ;    // OK
            info = GrB_Matrix_select_INT32_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GrB_INT64)
        {
            int64_t y = *((int64_t *) (scalar->x)) ;    // OK
            info = GrB_Matrix_select_INT64_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GrB_UINT8)
        {
            uint8_t y = *((uint8_t *) (scalar->x)) ;    // OK
            info = GrB_Matrix_select_UINT8_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GrB_UINT16)
        {
            uint16_t y = *((uint16_t *) (scalar->x)) ;  // OK
            info = GrB_Matrix_select_UINT16_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GrB_UINT32)
        {
            uint32_t y = *((uint32_t *) (scalar->x)) ;  // OK
            info = GrB_Matrix_select_UINT32_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GrB_UINT64)
        {
            uint64_t y = *((uint64_t *) (scalar->x)) ;  // OK
            info = GrB_Matrix_select_UINT64_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GrB_FP32)
        {
            float y = *((float *) (scalar->x)) ;    // OK
            info = GrB_Matrix_select_FP32_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GrB_FP64)
        {
            double y = *((double *) (scalar->x)) ;  // OK
            info = GrB_Matrix_select_FP64_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GxB_FC32)
        {
            GxB_FC32_t y = *((GxB_FC32_t *) (scalar->x)) ;  // OK
            info = GxB_Matrix_select_FC32_
                (C, Mask, accum, op, A, y, desc) ;
        }
        else if (stype == GxB_FC64)
        {
            GxB_FC64_t y = *((GxB_FC64_t *) (scalar->x)) ;  // OK
            info = GxB_Matrix_select_FC64_
                (C, Mask, accum, op, A, y, desc) ;
        }
    }
    else if (is_matrix && how == 0)
    {
        info = GrB_Matrix_select_Scalar_
            (C, Mask, accum, op, A, scalar, desc) ;
    }
    else if (!is_matrix && how == 1)
    {
        GrB_Vector w = (GrB_Vector) C ;
        GrB_Vector m = (GrB_Vector) Mask ;
        GrB_Vector a = (GrB_Vector) A ;
        if (stype == GrB_BOOL)
        {
            bool y = *((bool *) (scalar->x)) ;  // OK
            info = GrB_Vector_select_BOOL_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GrB_INT8)
        {
            int8_t y = *((int8_t *) (scalar->x)) ;  // OK
            info = GrB_Vector_select_INT8_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GrB_INT16)
        {
            int16_t y = *((int16_t *) (scalar->x)) ;    // OK
            info = GrB_Vector_select_INT16_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GrB_INT32)
        {
            int32_t y = *((int32_t *) (scalar->x)) ;    // OK
            info = GrB_Vector_select_INT32_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GrB_INT64)
        {
            int64_t y = *((int64_t *) (scalar->x)) ;    // OK
            info = GrB_Vector_select_INT64_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GrB_UINT8)
        {
            uint8_t y = *((uint8_t *) (scalar->x)) ;    // OK
            info = GrB_Vector_select_UINT8_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GrB_UINT16)
        {
            uint16_t y = *((uint16_t *) (scalar->x)) ;  // OK
            info = GrB_Vector_select_UINT16_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GrB_UINT32)
        {
            uint32_t y = *((uint32_t *) (scalar->x)) ;  // OK
            info = GrB_Vector_select_UINT32_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GrB_UINT64)
        {
            uint64_t y = *((uint64_t *) (scalar->x)) ;  // OK
            info = GrB_Vector_select_UINT64_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GrB_FP32)
        {
            float y = *((float *) (scalar->x)) ;    // OK
            info = GrB_Vector_select_FP32_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GrB_FP64)
        {
            double y = *((double *) (scalar->x)) ;  // OK
            info = GrB_Vector_select_FP64_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GxB_FC32)
        {
            GxB_FC32_t y = *((GxB_FC32_t *) (scalar->x)) ;  // OK
            info = GxB_Vector_select_FC32_ (w, m, accum, op, a, y, desc) ;
        }
        else if (stype == GxB_FC64)
        {
            GxB_FC64_t y = *((GxB_FC64_t *) (scalar->x)) ;  // OK
            info = GxB_Vector_select_FC64_ (w, m, accum, op, a, y, desc) ;
        }
    }
    else if (!is_matrix && how == 0)
    {
        GrB_Vector w = (GrB_Vector) C ;
        GrB_Vector m = (GrB_Vector) Mask ;
        GrB_Vector a = (GrB_Vector) A ;
        info = GrB_Vector_select_Scalar_ (w, m, accum, op, a, scalar, desc) ;
    }

    return (info) ;
}

//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;

    // check inputs
    if (nargout > 1 || nargin < 7 || nargin > 8)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get C (make a deep copy)
    #define GET_DEEP_COPY \
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;
    #define FREE_DEEP_COPY GrB_Matrix_free_(&C) ;
    GET_DEEP_COPY ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }

    // get Mask (shallow copy)
    Mask = GB_mx_mxArray_to_Matrix (pargin [1], "Mask", false, false) ;
    if (Mask == NULL && !mxIsEmpty (pargin [1]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("Mask failed") ;
    }

    // get how.  0: use GrB_Scalar, 1: use bare C scalar
    GET_SCALAR (4, int, how, 0) ;

    // get scalar (shallow copy)
    S = GB_mx_mxArray_to_Matrix (pargin [6], "scalar input", false, true) ;
    if (S == NULL || S->magic != GB_MAGIC)
    {
        FREE_ALL ;
        mexErrMsgTxt ("scalar failed") ;
    }
    GrB_Index snrows, sncols, snvals ;
    GrB_Matrix_nrows (&snrows, S) ;
    GrB_Matrix_ncols (&sncols, S) ;
    GrB_Matrix_nvals (&snvals, S) ;
    GxB_Format_Value fmt ;
    GxB_Matrix_Option_get_(S, GxB_FORMAT, &fmt) ;
    if (snrows != 1 || sncols != 1 || snvals != 1 || fmt != GxB_BY_COL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("scalar failed") ;
    }

    int f2 = 99 ;
    GxB_Matrix_Option_get_INT32 (S, GxB_FORMAT, &f2) ;
    if (fmt != GxB_BY_COL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("scalar failed") ;
    }

    scalar = (GrB_Scalar) S ;
    GrB_Info info = GxB_Scalar_fprint (scalar, "scalar", GxB_SILENT, NULL) ;
    if (info != GrB_SUCCESS)
    {
        FREE_ALL ;
        mexErrMsgTxt ("scalar failed") ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [5], "A input", false, true) ;
    if (A == NULL || A->magic != GB_MAGIC)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get accum, if present
    bool user_complex = (Complex != GxB_FC64)
        && (C->type == Complex || A->type == Complex) ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        C->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get op
    if (!GB_mx_mxArray_to_IndexUnaryOp (&op, pargin [3], "op",
        A->type) || op == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("IndexUnaryOp failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (7), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // C<Mask> = accum(C,op(x,A))
    if (GB_NCOLS (C) == 1 && (desc == NULL || desc->in0 == GxB_DEFAULT)
        && GB_VECTOR_OK (C))
    {
        // this is just to test the Vector version
        METHOD (select_idxunop (false)) ;
    }
    else
    {
        METHOD (select_idxunop (true)) ;
    }

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

