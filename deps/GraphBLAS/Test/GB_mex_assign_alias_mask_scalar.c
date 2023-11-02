//------------------------------------------------------------------------------
// GB_mex_assign_alias_mask_scalar: C<C,struct> = scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_assign_alias_mask_scalar (C, scalar)"

#define FREE_ALL                            \
{                                           \
    GrB_Matrix_free_(&C) ;                  \
    GrB_Matrix_free_(&S) ;                  \
    GB_mx_put_global (true) ;               \
}

GrB_Matrix C = NULL ;
GrB_Matrix S = NULL ;
GrB_Scalar scalar = NULL ;
GrB_Info assign_mask_scalar (void) ;

GrB_Info assign_mask_scalar (void)
{
    GrB_Info info ;
    GrB_Index nrows, ncols ;
    GrB_Matrix_nrows (&nrows, C) ;
    GrB_Matrix_ncols (&ncols, C) ;
    GrB_Type stype ;
    GxB_Scalar_type (&stype, scalar) ;

    // some descriptors use GrB_REPLACE, just to test this option.
    // It has no effect on the result.

    if (stype == GrB_BOOL)
    {
        bool y = *((bool *) (scalar->x)) ;  // OK
        info = GrB_Matrix_assign_BOOL_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_S) ;
    }
    else if (stype == GrB_INT8)
    {
        int8_t y = *((int8_t *) (scalar->x)) ;  // OK
        info = GrB_Matrix_assign_INT8_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_S) ;
    }
    else if (stype == GrB_INT16)
    {
        int16_t y = *((int16_t *) (scalar->x)) ;    // OK
        info = GrB_Matrix_assign_INT16_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_S) ;
    }
    else if (stype == GrB_INT32)
    {
        int32_t y = *((int32_t *) (scalar->x)) ;    // OK
        info = GrB_Matrix_assign_INT32_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_S) ;
    }
    else if (stype == GrB_INT64)
    {
        int64_t y = *((int64_t *) (scalar->x)) ;    // OK
        info = GrB_Matrix_assign_INT64_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_S) ;
    }
    else if (stype == GrB_UINT8)
    {
        uint8_t y = *((uint8_t *) (scalar->x)) ;    // OK
        info = GrB_Matrix_assign_UINT8_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_S) ;
    }
    else if (stype == GrB_UINT16)
    {
        uint16_t y = *((uint16_t *) (scalar->x)) ;  // OK
        info = GrB_Matrix_assign_UINT16_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_RS) ;
    }
    else if (stype == GrB_UINT32)
    {
        uint32_t y = *((uint32_t *) (scalar->x)) ;  // OK   // OK
        info = GrB_Matrix_assign_UINT32_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_RS) ;
    }
    else if (stype == GrB_UINT64)
    {
        uint64_t y = *((uint64_t *) (scalar->x)) ;  // OK
        info = GrB_Matrix_assign_UINT64_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_RS) ;
    }
    else if (stype == GrB_FP32)
    {
        float y = *((float *) (scalar->x)) ;    // OK
        info = GrB_Matrix_assign_FP32_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_RS) ;
    }
    else if (stype == GrB_FP64)
    {
        double y = *((double *) (scalar->x)) ;  // OK
        info = GrB_Matrix_assign_FP64_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_S) ;
    }
    else if (stype == GxB_FC32)
    {
        GxB_FC32_t y = *((GxB_FC32_t *) (scalar->x)) ;  // OK
        info = GxB_Matrix_assign_FC32_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_S) ;
    }
    else if (stype == GxB_FC64)
    {
        GxB_FC64_t y = *((GxB_FC64_t *) (scalar->x)) ;  // OK
        info = GxB_Matrix_assign_FC64_
            (C, C, NULL, y, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_RS) ;
    }
    else
    {
        mexErrMsgTxt ("unknown type") ;
    }

    return (info) ;
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

    // check inputs
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get C (make a deep copy)
    #define GET_DEEP_COPY       \
        C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;   \
        GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, C->sparsity_control) ;
    #define FREE_DEEP_COPY      \
        GrB_Matrix_free_(&C) ;
    GET_DEEP_COPY ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }
//  GxB_Matrix_fprint (C, "C", GxB_COMPLETE, NULL) ;

    // get scalar (shallow copy)
    S = GB_mx_mxArray_to_Matrix (pargin [1], "scalar input", false, true) ;
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
    scalar = (GrB_Scalar) S ;
//  GrB_Info info = GxB_Scalar_fprint (scalar, "scalar", GxB_COMPLETE, NULL) ;
//  if (info != GrB_SUCCESS)
//  {
//      FREE_ALL ;
//      mexErrMsgTxt ("scalar failed") ;
//  }

    // C<C,struct> = scalar
    METHOD (assign_mask_scalar ( )) ;
//  GxB_Matrix_fprint (C, "C output", GxB_COMPLETE, NULL) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

