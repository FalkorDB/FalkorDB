//------------------------------------------------------------------------------
// GB_mex_Vector_extractElement: interface for x = v(i)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "y = GB_mex_Vector_extractElement (v, I, xtype, use_scalar)"

#define FREE_ALL                                        \
{                                                       \
    GrB_Scalar_free_(&S) ;                              \
    GrB_Vector_free_(&v) ;                              \
    GB_mx_put_global (true) ;                           \
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
    GrB_Vector v = NULL ;
    GB_void *Y = NULL ;
    GrB_Index *I = NULL, ni = 0, I_range [3] ;
    bool is_list ;
    GrB_Scalar S = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get v (shallow copy)
    v = GB_mx_mxArray_to_Vector (pargin [0], "v input", false, true) ;
    if (v == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("v failed") ;
    }

    // get I
    if (!GB_mx_mxArray_to_indices (&I, pargin [1], &ni, I_range, &is_list))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I failed") ;
    }
    if (!is_list)
    {
        mexErrMsgTxt ("I must be a list") ;
    }

    // get xtype
    GrB_Type xtype = GB_mx_string_to_Type (PARGIN (2), v->type) ;

    // get use_scalar
    bool GET_SCALAR (3, bool, use_scalar, false) ;
    GrB_Scalar_new (&S, xtype) ;

    // create output Y
    pargout [0] = GB_mx_create_full (ni, 1, xtype) ;
    Y = mxGetData (pargout [0]) ;

    size_t s = 2 * sizeof (double) ;

    // x = v (i)
    switch (xtype->code)
    {
        case GB_BOOL_code   :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                bool *X = (bool *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_BOOL_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_BOOL_(&X [k], v, I [k])) ;
                }
            }
            break ;

        case GB_INT8_code   :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                int8_t *X = (int8_t *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_INT8_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_INT8_(&X [k], v, I [k])) ;
                }
            }
            break ;

        case GB_UINT8_code  :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                uint8_t *X = (uint8_t *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_UINT8_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_UINT8_(&X [k], v, I [k])) ;
                }
            }
            break ;

        case GB_INT16_code  :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                int16_t *X = (int16_t *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_INT16_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_INT16_(&X [k], v, I [k])) ;
                }
            }
            break ;

        case GB_UINT16_code :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                uint16_t *X = (uint16_t *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_UINT16_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_UINT16_(&X [k], v, I [k])) ;
                }
            }
            break ;

        case GB_INT32_code  :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                int32_t *X = (int32_t *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_INT32_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_INT32_(&X [k], v, I [k])) ;
                }
            }
            break ;

        case GB_UINT32_code :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                uint32_t *X = (uint32_t *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_UINT32_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_UINT32_(&X [k], v, I [k])) ;
                }
            }
            break ;

        case GB_INT64_code  :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                int64_t *X = (int64_t *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_INT64_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_INT64_(&X [k], v, I [k])) ;
                }
            }
            break ;

        case GB_UINT64_code :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                uint64_t *X = (uint64_t *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_UINT64_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_UINT64_(&X [k], v, I [k])) ;
                }
            }
            break ;

        case GB_FP32_code   :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                float *X = (float *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_FP32_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_FP32_(&X [k], v, I [k])) ;
                }
            }
            break ;

        case GB_FP64_code   :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                double *X = (double *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_FP64_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_FP64_(&X [k], v, I [k])) ;
                }
            }
            break;

        case GB_FC32_code   :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                GxB_FC32_t *X = (void *) Y ;

                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GxB_Scalar_extractElement_FC32_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GxB_Vector_extractElement_FC32_(&X [k], v, I [k])) ;
                }

            }
            break;

        case GB_FC64_code   :

            for (int64_t k = 0 ; k < ni ; k++)
            {
                GxB_FC64_t *X = (void *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GxB_Scalar_extractElement_FC64_(&X [k], S)) ;
                }
                else
                {
                    METHOD (GxB_Vector_extractElement_FC64_(&X [k], v, I [k])) ;
                }
            }
            break;

        case GB_UDT_code   :

            // user-defined Complex
            for (int64_t k = 0 ; k < ni ; k++)
            {
                GxB_FC64_t *X = (void *) Y ;
                if (use_scalar)
                {
                    METHOD (GrB_Vector_extractElement_Scalar_(S, v, I [k])) ;
                    METHOD (GrB_Scalar_extractElement_UDT (&X [k], S)) ;
                }
                else
                {
                    METHOD (GrB_Vector_extractElement_UDT (&X [k], v, I [k])) ;
                }
            }
            break;

        default              : FREE_ALL ; mexErrMsgTxt ("unsupported type") ;
    }

    FREE_ALL ;
}

