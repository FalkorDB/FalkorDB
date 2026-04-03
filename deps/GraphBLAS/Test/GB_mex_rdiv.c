//------------------------------------------------------------------------------
// GB_mex_rdiv: compute C=A*B with the rdiv operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is for testing only.  See GrB_mxm instead.  Returns a plain built-in
// matrix, in double.  The semiring is plus-rdiv-fp64 where plus is the 
// built-in GrB_PLUS_FP64 operator, and rdiv is z=y/x in double.

#include "GB_mex.h"

#define USAGE "C = GB_mex_rdiv (A, B, axb_method, cprint)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&B) ;              \
    GrB_Matrix_free_(&C) ;              \
    GrB_BinaryOp_free_(&My_rdiv) ;      \
    GrB_Semiring_free_(&My_plus_rdiv) ; \
    GB_mx_put_global (true) ;           \
}

//------------------------------------------------------------------------------

GrB_Info info ;
bool malloc_debug = false ;
bool ignore = false, ignore1 = false, ignore2 = false ;
bool cprint = false ;
GrB_Matrix A = NULL, B = NULL, C = NULL, MT = NULL ;
int64_t anrows = 0 ;
int64_t ancols = 0 ;
int64_t bnrows = 0 ;
int64_t bncols = 0 ;
int AxB_method = GxB_DEFAULT ;
struct GB_Matrix_opaque MT_header, C_header ;

GrB_Info axb (GB_Werk Werk, bool cprint) ;

GrB_Semiring My_plus_rdiv = NULL ;
GrB_BinaryOp My_rdiv = NULL ;

 void my_rdiv (double *z, const double *x, const double *y) ;

 void my_rdiv (double *z, const double *x, const double *y)
 {
     (*z) = (*y) / (*x) ;
 }

#define MY_RDIV                                                 \
"void my_rdiv (double *z, const double *x, const double *y)\n"  \
"{\n"                                                           \
"    // escape this quote: \"\n"                                \
"    /* escape this backslash \\ */\n"                          \
"    /* revised for GrB 10.1.0 */\n"                            \
"    (*z) = (*y) / (*x) ;\n"                                    \
"}"

//------------------------------------------------------------------------------

GrB_Info axb (GB_Werk Werk, bool cprint)
{
    // create the rdiv operator
    // try with a NULL function pointer, to test the JIT
    info = GxB_BinaryOp_new (&My_rdiv,
        NULL, GrB_FP64, GrB_FP64, GrB_FP64,
        "my_rdiv", MY_RDIV) ;
    if (info == GrB_NULL_POINTER)
    {
        // try again with the my_rdiv function pointer
        info = GxB_BinaryOp_new (&My_rdiv,
            (GxB_binary_function) my_rdiv, GrB_FP64, GrB_FP64, GrB_FP64,
            "my_rdiv", MY_RDIV) ;
    }
    if (info != GrB_SUCCESS) return (info) ;
    GrB_BinaryOp_wait_(My_rdiv, GrB_MATERIALIZE) ;
    if (info != GrB_SUCCESS) return (info) ;
    info = GrB_Semiring_new (&My_plus_rdiv, GxB_PLUS_FP64_MONOID, My_rdiv) ;
    if (info != GrB_SUCCESS)
    {
        GrB_BinaryOp_free_(&My_rdiv) ;
        return (info) ;
    }

    MT = GB_clear_matrix_header (&MT_header) ;
    C  = GB_clear_matrix_header (&C_header) ;

    // C = A*B
    info = GB_AxB_meta (C, NULL,       // C cannot be computed in place
        false,      // C_replace
        true,       // CSC
        MT,         // no MT returned
        &ignore1,   // M_transposed will be false
        NULL,       // no Mask
        false,      // mask not complemented
        false,      // mask not structural
        NULL,       // no accum
        A, B,
        My_plus_rdiv,
        false,      // no A transpose
        false,      // no B transpose
        false,      // no flipxy
        &ignore,    // mask_applied
        &ignore2,   // done_in_place
        AxB_method,
        true,       // do the sort
        Werk) ;

    if (info == GrB_SUCCESS && C != NULL)
    {
        if (cprint) GxB_Matrix_fprint_(C, GxB_COMPLETE, NULL) ;
    }

    GrB_BinaryOp_free_(&My_rdiv) ;
    GrB_Semiring_free_(&My_plus_rdiv) ;

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

    info = GrB_SUCCESS ;
    malloc_debug = GB_mx_get_global (true) ;
    ignore = false ;
    ignore1 = false ;
    ignore2 = false ;
    A = NULL ;
    B = NULL ;
    C = NULL ;

    My_rdiv = NULL ;
    My_plus_rdiv = NULL ;

    GB_WERK (USAGE) ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A and B
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A", false, true) ;
    B = GB_mx_mxArray_to_Matrix (pargin [1], "B", false, true) ;
    if (A == NULL || B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("failed") ;
    }

    if (!A->is_csc || !B->is_csc)
    {
        mexErrMsgTxt ("A and B must be in CSC format") ;
    }

    // get the axb_method
    // 0 or not present: default
    // 7081: Gustavson
    // 7083: dot
    // 7084: hash
    // 7085: saxpy
    GET_SCALAR (2, int, AxB_method, GxB_DEFAULT) ;

    // get the cprint flag
    GET_SCALAR (3, bool, cprint, false) ;

    if (! ((AxB_method == GxB_DEFAULT) ||
        (AxB_method == GxB_AxB_GUSTAVSON) ||
        (AxB_method == GxB_AxB_HASH) ||
        (AxB_method == GxB_AxB_SAXPY) ||
        (AxB_method == GxB_AxB_DOT)))
    {
        mexErrMsgTxt ("unknown method") ;
    }

    // determine the dimensions
    anrows = GB_NROWS (A) ;
    ancols = GB_NCOLS (A) ;
    bnrows = GB_NROWS (B) ;
    bncols = GB_NCOLS (B) ;
    if (ancols != bnrows)
    {
        FREE_ALL ;
        mexErrMsgTxt ("invalid dimensions") ;
    }

    METHOD (axb (Werk, cprint)) ;

    // return C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C AxB result", false) ;

    FREE_ALL ;
}

