//------------------------------------------------------------------------------
// GB_mx_get_Scalar: get a GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

GrB_Scalar GB_mx_get_Scalar
(
    const mxArray *mx_scalar
)
{

    if (mx_scalar == NULL)
    {
        mexErrMsgTxt ("scalar missing") ;
    }

    if (mxIsSparse (mx_scalar))
    {
        mexErrMsgTxt ("sparse scalar not supported") ;
    }

    GrB_Scalar Scalar = NULL ;
    GrB_Type scalar_type = GB_mx_Type (mx_scalar) ;
    GrB_Scalar_new (&Scalar, scalar_type) ;

    if (scalar_type == GrB_BOOL)
    {
        bool *p = mxGetData (mx_scalar) ;
        GrB_Scalar_setElement_BOOL_(Scalar, *p) ;
    }
    else if (scalar_type == GrB_INT8)
    {
        int8_t *p = mxGetInt8s (mx_scalar) ;
        GrB_Scalar_setElement_INT8_(Scalar, *p) ;
    }
    else if (scalar_type == GrB_INT16)
    {
        int16_t *p = mxGetInt16s (mx_scalar) ;
        GrB_Scalar_setElement_INT16_(Scalar, *p) ;
    }
    else if (scalar_type == GrB_INT32)
    {
        int32_t *p = mxGetInt32s (mx_scalar) ;
        GrB_Scalar_setElement_INT32_(Scalar, *p) ;
    }
    else if (scalar_type == GrB_INT64)
    {
        int64_t *p = mxGetInt64s (mx_scalar) ;
        GrB_Scalar_setElement_INT64_(Scalar, *p) ;
    }
    else if (scalar_type == GrB_UINT8)
    {
        uint8_t *p = mxGetUint8s (mx_scalar) ;
        GrB_Scalar_setElement_UINT8_(Scalar, *p) ;
    }
    else if (scalar_type == GrB_UINT16)
    {
        uint16_t *p = mxGetUint16s (mx_scalar) ;
        GrB_Scalar_setElement_UINT16_(Scalar, *p) ;
    }
    else if (scalar_type == GrB_UINT32)
    {
        uint32_t *p = mxGetUint32s (mx_scalar) ;
        GrB_Scalar_setElement_UINT32_(Scalar, *p) ;
    }
    else if (scalar_type == GrB_UINT64)
    {
        uint64_t *p = mxGetUint64s (mx_scalar) ;
        GrB_Scalar_setElement_UINT64_(Scalar, *p) ;
    }
    else if (scalar_type == GrB_FP32)
    {
        float *p = mxGetSingles (mx_scalar) ;
        GrB_Scalar_setElement_FP32_(Scalar, *p) ;
    }
    else if (scalar_type == GrB_FP64)
    {
        double *p = mxGetDoubles (mx_scalar) ;
        GrB_Scalar_setElement_FP64_(Scalar, *p) ;
    }
    else if (scalar_type == GxB_FC32)
    {
        GxB_FC32_t *p = (GxB_FC32_t *) mxGetComplexSingles (mx_scalar) ;
        GxB_Scalar_setElement_FC32_(Scalar, *p) ;
    }
    else if (scalar_type == GxB_FC64)
    {
        GxB_FC64_t *p = (GxB_FC64_t *) mxGetComplexDoubles (mx_scalar) ;
        GxB_Scalar_setElement_FC64_(Scalar, *p) ;
    }
    else if (scalar_type == Complex)
    {
        GxB_FC64_t *p = (GxB_FC64_t *) mxGetComplexDoubles (mx_scalar) ;
        GrB_Scalar_setElement_UDT (Scalar, p) ;
    }
    else
    {
        GxB_print (scalar_type, 3) ;
        mexErrMsgTxt ("unknown scalar type") ;
    }

    GrB_Scalar_wait_(Scalar, GrB_MATERIALIZE) ;

    return (Scalar) ;
}

