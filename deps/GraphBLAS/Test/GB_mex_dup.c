//------------------------------------------------------------------------------
// GB_mex_dup: copy a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// copy and typecast a matrix

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "C = GB_mex_dup (A, type, method, sparsity)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&C) ;              \
    GrB_Descriptor_free_(&desc) ;       \
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
    GrB_Matrix A = NULL, C = NULL ;
    GrB_Descriptor desc = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    GrB_Matrix_set_String (A, "A input", GrB_NAME) ;

    // get ctype of output matrix
    GrB_Type ctype = GB_mx_string_to_Type (PARGIN (1), A->type) ;

    // get method
    int GET_SCALAR (2, int, method, 0) ;

    // get sparsity
    int GET_SCALAR (3, int, sparsity, GxB_DEFAULT) ;

    if (ctype == A->type)
    {
        // copy C with the same type as A, with default sparsity
        if (method == 0 && sparsity == GxB_DEFAULT)
        {
            METHOD (GrB_Matrix_dup (&C, A)) ;

            // get the name of the C matrix
            char name [256] ;
            GrB_Matrix_get_String (C, name, GrB_NAME) ;
            CHECK (MATCH (name, "A input")) ;

        }
        else
        {
            // try another method, just for testing (see User Guide)

            // C = create an exact copy of A, just like GrB_Matrix_dup
            GrB_Type type ;
            GrB_Index nrows, ncols ;

            #undef GET_DEEP_COPY
            #undef FREE_DEEP_COPY

            #define GET_DEEP_COPY                               \
            {                                                   \
                GxB_Matrix_type (&type, A) ;                    \
                GrB_Matrix_nrows (&nrows, A) ;                  \
                GrB_Matrix_ncols (&ncols, A) ;                  \
                GrB_Matrix_new (&C, type, nrows, ncols) ;       \
                GrB_Descriptor_new (&desc) ;                    \
                if (sparsity != GxB_DEFAULT)                    \
                {                                               \
                    GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, sparsity) ;\
                }                                               \
                GxB_Desc_set (desc, GrB_INP0, GrB_TRAN) ;       \
            }
            #define FREE_DEEP_COPY                              \
            {                                                   \
                GrB_Matrix_free_(&C) ;                          \
                GrB_Descriptor_free_(&desc) ;                   \
            }

            GET_DEEP_COPY ;

            if (method == 1)
            {
                // C = A using GrB_transpose with a desc.inp0 = transpose
                METHOD (GrB_transpose (C, NULL, NULL, A, desc)) ;
            }
            else
            {
                // C = A using GrB_assign
                METHOD (GrB_assign (C, NULL, NULL, A,
                    GrB_ALL, nrows, GrB_ALL, ncols, NULL)) ;
            }

            #undef GET_DEEP_COPY
            #undef FREE_DEEP_COPY

        }
    }
    else
    {
        // typecast
        if (A->type == Complex && Complex != GxB_FC64)
        {
            A->type = GxB_FC64 ;
        }

        // C = (ctype) A
        GrB_Index nrows, ncols ;

        #define GET_DEEP_COPY                               \
        {                                                   \
            GrB_Matrix_nrows (&nrows, A) ;                  \
            GrB_Matrix_ncols (&ncols, A) ;                  \
            GrB_Matrix_new (&C, ctype, nrows, ncols) ;      \
            GrB_Descriptor_new (&desc) ;                    \
            if (sparsity != GxB_DEFAULT)                    \
            {                                               \
                GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, sparsity) ; \
            }                                               \
            GxB_Desc_set (desc, GrB_INP0, GrB_TRAN) ;       \
        }
        #define FREE_DEEP_COPY                              \
        {                                                   \
            GrB_Matrix_free_(&C) ;                          \
            GrB_Descriptor_free_(&desc) ;                   \
        }

        GET_DEEP_COPY ;

        if (method == 1)
        {
            // C = A using GrB_transpose with a desc.inp0 = transpose
            METHOD (GrB_transpose (C, NULL, NULL, A, desc)) ;
        }
        else
        {
            // C = A using GrB_assign
            METHOD (GrB_assign (C, NULL, NULL, A,
                GrB_ALL, nrows, GrB_ALL, ncols, NULL)) ;
        }

        #undef GET_DEEP_COPY
        #undef FREE_DEEP_COPY
    }

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

