//------------------------------------------------------------------------------
// GB_mex_extractTuples: extract all tuples from a matrix or vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[I,J,X] = GB_mex_extractTuples (A, xtype, method)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Vector_free_(&I_vector) ;       \
    GrB_Vector_free_(&J_vector) ;       \
    GrB_Vector_free_(&X_vector) ;       \
    GB_mx_put_global (true) ;           \
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
    GrB_Matrix A = NULL ;
    GrB_Vector I_vector = NULL, J_vector = NULL, X_vector = NULL ;
    void *I_output = NULL, *J_output = NULL ;
    GB_void *X = NULL ;
    GrB_Type xtype = NULL ;
    uint64_t nvals = 0 ;

    // check inputs
    if (nargout > 3 || nargin < 1 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get the number of entries in A
    GrB_Matrix_nvals (&nvals, A) ;

    // get the method:  0: use (uint64_t *), 1: use GrB_Vector for I,J
    int GET_SCALAR (2, int, method, 0) ;

    //--------------------------------------------------------------------------
    // [I,J,X] = find (A)
    //--------------------------------------------------------------------------

    if (method == 0)
    {

        //----------------------------------------------------------------------
        // use GrB_[Matrix,Vector]_extractTuples_TYPE, no GrB_Vectors
        //----------------------------------------------------------------------

        // create I
        pargout [0] = GB_mx_create_full (nvals, 1, GrB_UINT64) ;
        I_output = mxGetData (pargout [0]) ;
        // create J
        if (nargout > 1)
        {
            pargout [1] = GB_mx_create_full (nvals, 1, GrB_UINT64) ;
            J_output = mxGetData (pargout [1]) ;
        }
        // create X
        xtype = GB_mx_string_to_Type (PARGIN (1), A->type) ;
        if (nargout > 2)
        {
            pargout [2] = GB_mx_create_full (nvals, 1, xtype) ;
            X = (GB_void *) mxGetData (pargout [2]) ;
        }

        uint64_t *I = (uint64_t *) I_output ;
        uint64_t *J = (uint64_t *) J_output ;

        if (GB_VECTOR_OK (A))
        {
            // test extract vector methods
            GrB_Vector v = (GrB_Vector) A ;
            switch (xtype->code)
            {
                case GB_BOOL_code   : METHOD (GrB_Vector_extractTuples_BOOL_  (I, (bool       *) X, &nvals, v)) ; break ;
                case GB_INT8_code   : METHOD (GrB_Vector_extractTuples_INT8_  (I, (int8_t     *) X, &nvals, v)) ; break ;
                case GB_UINT8_code  : METHOD (GrB_Vector_extractTuples_UINT8_ (I, (uint8_t    *) X, &nvals, v)) ; break ;
                case GB_INT16_code  : METHOD (GrB_Vector_extractTuples_INT16_ (I, (int16_t    *) X, &nvals, v)) ; break ;
                case GB_UINT16_code : METHOD (GrB_Vector_extractTuples_UINT16_(I, (uint16_t   *) X, &nvals, v)) ; break ;
                case GB_INT32_code  : METHOD (GrB_Vector_extractTuples_INT32_ (I, (int32_t    *) X, &nvals, v)) ; break ;
                case GB_UINT32_code : METHOD (GrB_Vector_extractTuples_UINT32_(I, (uint32_t   *) X, &nvals, v)) ; break ;
                case GB_INT64_code  : METHOD (GrB_Vector_extractTuples_INT64_ (I, (int64_t    *) X, &nvals, v)) ; break ;
                case GB_UINT64_code : METHOD (GrB_Vector_extractTuples_UINT64_(I, (uint64_t   *) X, &nvals, v)) ; break ;
                case GB_FP32_code   : METHOD (GrB_Vector_extractTuples_FP32_  (I, (float      *) X, &nvals, v)) ; break ;
                case GB_FP64_code   : METHOD (GrB_Vector_extractTuples_FP64_  (I, (double     *) X, &nvals, v)) ; break ;
                case GB_FC32_code   : METHOD (GxB_Vector_extractTuples_FC32_  (I, (GxB_FC32_t *) X, &nvals, v)) ; break ;
                case GB_FC64_code   : METHOD (GxB_Vector_extractTuples_FC64_  (I, (GxB_FC64_t *) X, &nvals, v)) ; break ;
                case GB_UDT_code    : METHOD (GrB_Vector_extractTuples_UDT_   (I, (void       *) X, &nvals, v)) ; break ;
                default             : FREE_ALL ; mexErrMsgTxt ("unsupported type") ;
            }
            if (J != NULL)
            {
                for (int64_t p = 0 ; p < nvals ; p++) J [p] = 0 ;
            }
        }
        else
        {
            switch (xtype->code)
            {
                case GB_BOOL_code   : METHOD (GrB_Matrix_extractTuples_BOOL_  (I, J, (bool       *) X, &nvals, A)) ; break ;
                case GB_INT8_code   : METHOD (GrB_Matrix_extractTuples_INT8_  (I, J, (int8_t     *) X, &nvals, A)) ; break ;
                case GB_UINT8_code  : METHOD (GrB_Matrix_extractTuples_UINT8_ (I, J, (uint8_t    *) X, &nvals, A)) ; break ;
                case GB_INT16_code  : METHOD (GrB_Matrix_extractTuples_INT16_ (I, J, (int16_t    *) X, &nvals, A)) ; break ;
                case GB_UINT16_code : METHOD (GrB_Matrix_extractTuples_UINT16_(I, J, (uint16_t   *) X, &nvals, A)) ; break ;
                case GB_INT32_code  : METHOD (GrB_Matrix_extractTuples_INT32_ (I, J, (int32_t    *) X, &nvals, A)) ; break ;
                case GB_UINT32_code : METHOD (GrB_Matrix_extractTuples_UINT32_(I, J, (uint32_t   *) X, &nvals, A)) ; break ;
                case GB_INT64_code  : METHOD (GrB_Matrix_extractTuples_INT64_ (I, J, (int64_t    *) X, &nvals, A)) ; break ;
                case GB_UINT64_code : METHOD (GrB_Matrix_extractTuples_UINT64_(I, J, (uint64_t   *) X, &nvals, A)) ; break ;
                case GB_FP32_code   : METHOD (GrB_Matrix_extractTuples_FP32_  (I, J, (float      *) X, &nvals, A)) ; break ;
                case GB_FP64_code   : METHOD (GrB_Matrix_extractTuples_FP64_  (I, J, (double     *) X, &nvals, A)) ; break ;
                case GB_FC32_code   : METHOD (GxB_Matrix_extractTuples_FC32_  (I, J, (GxB_FC32_t *) X, &nvals, A)) ; break ;
                case GB_FC64_code   : METHOD (GxB_Matrix_extractTuples_FC64_  (I, J, (GxB_FC64_t *) X, &nvals, A)) ; break ;
                case GB_UDT_code    : METHOD (GrB_Matrix_extractTuples_UDT_   (I, J, (void       *) X, &nvals, A)) ; break ;
                default             : FREE_ALL ; mexErrMsgTxt ("unsupported type") ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // use GxB_[Matrix,Vector]_extractTuples_Vector
        //----------------------------------------------------------------------


        #undef GET_DEEP_COPY
        #define GET_DEEP_COPY                                       \
            GrB_Vector_new (&I_vector, GrB_UINT64, 0) ;             \
            if (nargout > 1)                                        \
            {                                                       \
                /* J = zeros (nvals,1) */                           \
                GrB_Vector_new (&J_vector, GrB_UINT32, nvals) ;     \
                GrB_Vector_assign_UINT32 (J_vector, NULL, NULL, 0,  \
                    GrB_ALL, nvals, NULL) ;                         \
                /* ensure J is non-iso */                           \
                if (nvals > 0)                                      \
                {                                                   \
                    uint64_t gunk [1] ;                             \
                    gunk [0] = 0 ;                                  \
                    GrB_Vector_assign_UINT32 (J_vector, NULL, NULL, \
                        1, gunk, 1, NULL) ;                         \
                    GrB_Vector_assign_UINT32 (J_vector, NULL, NULL, \
                        0, gunk, 1, NULL) ;                         \
                }                                                   \
            }                                                       \
            if (nargout > 2)                                        \
            {                                                       \
                GrB_Vector_new (&X_vector, GrB_UINT64, 0) ;         \
            }

        #undef  FREE_DEEP_COPY
        #define FREE_DEEP_COPY                  \
            GrB_Vector_free_(&I_vector) ;       \
            GrB_Vector_free_(&J_vector) ;       \
            GrB_Vector_free_(&X_vector) ;

        GET_DEEP_COPY ;

        if (GB_VECTOR_OK (A))
        {
            GrB_Vector v = (GrB_Vector) A ;
            METHOD (GxB_Vector_extractTuples_Vector_(I_vector, X_vector, v,
                NULL)) ;
            if (nargout > 1)
            {
                // J = zeros (nvals,1)
                GrB_Vector_assign_UINT64 (J_vector, NULL, NULL, 0,
                    GrB_ALL, nvals, NULL) ;
            }
        }
        else
        {
            METHOD (GxB_Matrix_extractTuples_Vector_(I_vector, J_vector,
                X_vector, A, NULL)) ;
        }
        pargout [0] = GB_mx_Vector_to_mxArray (&I_vector, "I", false) ;
        if (nargout > 1)
        {
            pargout [1] = GB_mx_Vector_to_mxArray (&J_vector, "J", false) ;
        }
        if (nargout > 2)
        {
            pargout [2] = GB_mx_Vector_to_mxArray (&X_vector, "X", false) ;
        }
    }

    FREE_ALL ;
}

