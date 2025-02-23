//------------------------------------------------------------------------------
// GB_mex_container: copy a matrix, by loading/unloading it into a container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// copy a matrix/vector via load/unload of a container

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "../Source/container/GB_container.h"

#define USAGE "C = GB_mex_container (A)"

#undef  FREE_ALL2
#define FREE_ALL2                       \
{                                       \
    GrB_Matrix_free_(&C) ;              \
    GxB_Container_free (&Container) ;   \
}

#define OK2(method)                     \
{                                       \
    info = (method) ;                   \
    if (info != GrB_SUCCESS)            \
    {                                   \
        FREE_ALL2 ;                     \
        return (info) ;                 \
    }                                   \
}

//------------------------------------------------------------------------------
// matrix_method
//------------------------------------------------------------------------------

GrB_Info matrix_method (GrB_Matrix *C_handle, GrB_Matrix A) ;
GrB_Info matrix_method (GrB_Matrix *C_handle, GrB_Matrix A)
{
    // test matrix variant
    GrB_Info info ;
    GrB_Matrix C = NULL ;
    GxB_Container Container = NULL ;
    OK2 (GxB_Container_new (&Container)) ;
    OK2 (GrB_Matrix_dup (&C, A)) ;
    OK2 (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    OK2 (GxB_unload_Matrix_into_Container (C, Container, NULL)) ;
    uint64_t len ;
    OK2 (GrB_Vector_size (&len, Container->h)) ;
    if (len == 0)
    {
//      printf ("\n----------------- h_empty:\n") ;
        // test case when h_empty is true
        GB_vector_reset (Container->h) ;
    }
//  printf ("\n----------------- GxB_load_Matrix_from_Container:\n") ;
    OK2 (GxB_load_Matrix_from_Container (C, Container, NULL)) ;
    (*C_handle) = C ;
    GxB_Container_free (&Container) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// vector_method
//------------------------------------------------------------------------------

GrB_Info vector_method (GrB_Vector *C_handle, GrB_Vector A) ;
GrB_Info vector_method (GrB_Vector *C_handle, GrB_Vector A)
{
    // test vector variant
    GrB_Info info ;
    GrB_Vector C = NULL ;
    GxB_Container Container ;
    OK2 (GxB_Container_new (&Container)) ;
    OK2 (GrB_Vector_dup (&C, A)) ;
    OK2 (GrB_Vector_wait (C, GrB_MATERIALIZE)) ;
    OK2 (GxB_unload_Vector_into_Container (C, Container, NULL)) ;
    uint64_t len ;
    OK2 (GrB_Vector_size (&len, Container->h)) ;
    if (len == 0)
    {
//      printf ("\n----------------- h_empty:\n") ;
        // test case when h_empty is true
        GB_vector_reset (Container->h) ;
    }
//  printf ("\n----------------- GxB_load_Vector_from_Container:\n") ;
    OK2 (GxB_load_Vector_from_Container (C, Container, NULL)) ;
    (*C_handle) = C ;
    GxB_Container_free (&Container) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_mex_container mexFunction
//------------------------------------------------------------------------------


#undef  FREE_ALL
#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&C) ;              \
    GrB_Matrix_free_(&A) ;              \
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

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix C = NULL, A = NULL ;

    // check inputs
    if (nargout > 1 || nargin != 1)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY \
        GrB_Matrix_free (&C) ;

    // get a shallow copy of the input
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;

    if (GB_VECTOR_OK (A))
    {
        // test vector variant
        METHOD (vector_method ((GrB_Vector *) &C, (GrB_Vector) A)) ;
    }
    else
    {
        // test matrix variant
        METHOD (matrix_method (&C, A)) ;
    }

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

