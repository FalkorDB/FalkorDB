//------------------------------------------------------------------------------
// GB_mex_AxB_idx: C=A*B, A'*B, A*B', or A'*B' using the indexop semirings
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is for testing only.  See GrB_mxm instead.

// monoid: min, max, plus, times
// mult: firsti, firsti1, firstj, firstj1, secondi, secondi1, secondj, secondj1

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "C = GB_mex_AxB_idx (A, B, atrans, btrans, axb_method," \
    " C_is_csc, builtin, add, mult)"

#define FREE_ALL                                \
{                                               \
    GrB_Matrix_free (&A) ;                      \
    GrB_Matrix_free (&B) ;                      \
    GrB_Matrix_free (&C) ;                      \
    GrB_Scalar_free (&Theta) ;                  \
    GrB_Descriptor_free (&desc) ;               \
    GrB_BinaryOp_free (&mult) ;                 \
    GxB_IndexBinaryOp_free (&Iop) ;             \
    GrB_Monoid_free (&monoid) ;                 \
    GrB_Semiring_free (&semiring) ;             \
    GB_mx_put_global (true) ;                   \
}

//------------------------------------------------------------------------------
// user-defined index binary operators
//------------------------------------------------------------------------------

void gb_firsti_theta (int64_t *z,
    const void *x, uint64_t ix, uint64_t jx,
    const void *y, uint64_t iy, uint64_t jy,
    const int64_t *theta) ;

void gb_firsti_theta (int64_t *z,
    const void *x, uint64_t ix, uint64_t jx,
    const void *y, uint64_t iy, uint64_t jy,
    const int64_t *theta)
{
    (*z) = ix + (*theta) ;
}

#define FIRSTI_THETA_DEFN                               \
"void gb_firsti_theta (int64_t *z,                     \n" \
"    const void *x, uint64_t ix, uint64_t jx,       \n" \
"    const void *y, uint64_t iy, uint64_t jy,       \n" \
"    const int64_t *theta)                          \n" \
"{                                                  \n" \
"    (*z) = ix + (*theta) ;                         \n" \
"}"

void gb_secondi_theta (int64_t *z,
    const void *x, uint64_t ix, uint64_t jx,
    const void *y, uint64_t iy, uint64_t jy,
    const int64_t *theta) ;

void gb_secondi_theta (int64_t *z,
    const void *x, uint64_t ix, uint64_t jx,
    const void *y, uint64_t iy, uint64_t jy,
    const int64_t *theta)
{
    (*z) = iy + (*theta) ;
}

#define SECONDI_THETA_DEFN                              \
"void gb_secondi_theta (int64_t *z,                    \n" \
"    const void *x, uint64_t ix, uint64_t jx,       \n" \
"    const void *y, uint64_t iy, uint64_t jy,       \n" \
"    const int64_t *theta)                          \n" \
"{                                                  \n" \
"    (*z) = iy + (*theta) ;                         \n" \
"}"

void gb_firstj_theta (int64_t *z,
    const void *x, uint64_t ix, uint64_t jx,
    const void *y, uint64_t iy, uint64_t jy,
    const int64_t *theta) ;

void gb_firstj_theta (int64_t *z,
    const void *x, uint64_t ix, uint64_t jx,
    const void *y, uint64_t iy, uint64_t jy,
    const int64_t *theta)
{
    (*z) = jx + (*theta) ;
}

#define FIRSTJ_THETA_DEFN                               \
"void gb_firstj_theta (int64_t *z,                     \n" \
"    const void *x, uint64_t ix, uint64_t jx,       \n" \
"    const void *y, uint64_t iy, uint64_t jy,       \n" \
"    const int64_t *theta)                          \n" \
"{                                                  \n" \
"    (*z) = jx + (*theta) ;                         \n" \
"}"

void gb_secondj_theta (int64_t *z,
    const void *x, uint64_t ix, uint64_t jx,
    const void *y, uint64_t iy, uint64_t jy,
    const int64_t *theta) ;

void gb_secondj_theta (int64_t *z,
    const void *x, uint64_t ix, uint64_t jx,
    const void *y, uint64_t iy, uint64_t jy,
    const int64_t *theta)
{
    (*z) = jy + (*theta) ;
}

#define SECONDJ_THETA_DEFN                              \
"void gb_secondj_theta (int64_t *z,                    \n" \
"    const void *x, uint64_t ix, uint64_t jx,       \n" \
"    const void *y, uint64_t iy, uint64_t jy,       \n" \
"    const int64_t *theta)                          \n" \
"{                                                  \n" \
"    (*z) = jy + (*theta) ;                         \n" \
"}"

//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info = GrB_SUCCESS ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL, B = NULL, C = NULL ;
    GrB_Scalar Theta = NULL ;
    GrB_BinaryOp mult = NULL ;
    GxB_IndexBinaryOp Iop = NULL ;
    GrB_Semiring semiring = NULL ;
    uint64_t anrows = 0, ancols = 0, bnrows = 0, bncols = 0 ;
    GrB_Descriptor desc = NULL ;
    GrB_Monoid monoid = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 9)
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

    // get the atrans option
    bool GET_SCALAR (2, bool, atrans, false) ;

    // get the btrans option
    bool GET_SCALAR (3, bool, btrans, false) ;

    // get the axb_method
    int GET_SCALAR (4, int, AxB_method, GxB_DEFAULT) ;

    // get the C_is_csc option
    bool GET_SCALAR (5, bool, C_is_csc, true) ;

    // get the builtin option
    bool GET_SCALAR (6, bool, builtin, true) ;

    // get the add monoid; defaults to 'min'
    #define LEN 256
    char addname [LEN+1] ;
    strcpy (addname, "min") ;
    if (nargin > 7)
    {
        int len = GB_mx_mxArray_to_string (addname, LEN, pargin [7]) ;
        if (len == -1)
        {
            mexErrMsgTxt ("addname must be char") ;
        }
    }

    // get the mult operator; defaults to 'secondi1'
    char multname [LEN+1] ;
    strcpy (multname, "secondi1") ;
    if (nargin > 8)
    {
        int len = GB_mx_mxArray_to_string (multname, LEN, pargin [8]) ;
        if (len == -1)
        {
            mexErrMsgTxt ("multname must be char") ;
        }
    }

    // set the Descriptor
    OK (GrB_Descriptor_new (&desc)) ;
    OK (GrB_Descriptor_set (desc, GrB_INP0, atrans ? GrB_TRAN : GxB_DEFAULT)) ;
    OK (GrB_Descriptor_set (desc, GrB_INP1, btrans ? GrB_TRAN : GxB_DEFAULT)) ;
    OK (GrB_Descriptor_set (desc, GxB_AxB_METHOD, AxB_method)) ;

    // determine the dimensions
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;
    uint64_t cnrows = (atrans) ? ancols : anrows ;
    uint64_t cncols = (btrans) ? bnrows : bncols ;

    // create the output matrix C
    OK (GrB_Matrix_new (&C, GrB_INT64, cnrows, cncols)) ;
    OK (GrB_Matrix_set_INT32 (C, C_is_csc, GrB_STORAGE_ORIENTATION_HINT)) ;

    // create the monoid
         if (MATCH (addname, "min"  )) monoid = GrB_MIN_MONOID_INT64 ;
    else if (MATCH (addname, "max"  )) monoid = GrB_MAX_MONOID_INT64 ;
    else if (MATCH (addname, "plus" )) monoid = GrB_PLUS_MONOID_INT64 ;
    else if (MATCH (addname, "times")) monoid = GrB_TIMES_MONOID_INT64 ;
    else
    {
        mexErrMsgTxt ("add not supported") ;
    }

    // create the mult operator
    if (builtin)
    {

        //----------------------------------------------------------------------
        // built-in operator
        //----------------------------------------------------------------------

             if (MATCH (multname, "firsti"  )) mult = GxB_FIRSTI_INT64 ;
        else if (MATCH (multname, "firsti1" )) mult = GxB_FIRSTI1_INT64 ;
        else if (MATCH (multname, "firstj"  )) mult = GxB_FIRSTJ_INT64 ;
        else if (MATCH (multname, "firstj1" )) mult = GxB_FIRSTJ1_INT64 ;
        else if (MATCH (multname, "secondi" )) mult = GxB_SECONDI_INT64 ;
        else if (MATCH (multname, "secondi1")) mult = GxB_SECONDI1_INT64 ;
        else if (MATCH (multname, "secondj" )) mult = GxB_SECONDJ_INT64 ;
        else if (MATCH (multname, "secondj1")) mult = GxB_SECONDJ1_INT64 ;
        else
        {
            mexErrMsgTxt ("mult not supported") ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // user-defined operator
        //----------------------------------------------------------------------

        // create the index binary op
        int theta ;
        if (MATCH (multname, "firsti" ))
        {
            theta = 0 ;
            OK (GxB_IndexBinaryOp_new (&Iop,
                (GxB_index_binary_function) gb_firsti_theta,
                GrB_INT64, GrB_FP64, GrB_FP64, GrB_INT64,
                "gb_firsti_theta", FIRSTI_THETA_DEFN)) ;
        }
        else if (MATCH (multname, "firsti1"))
        {
            theta = 1 ;
            OK (GxB_IndexBinaryOp_new (&Iop,
                (GxB_index_binary_function) gb_firsti_theta,
                GrB_INT64, GrB_FP64, GrB_FP64, GrB_INT64,
                "gb_firsti_theta", FIRSTI_THETA_DEFN)) ;
        }
        else if (MATCH (multname, "firstj" ))
        {
            theta = 0 ;
            OK (GxB_IndexBinaryOp_new (&Iop,
                (GxB_index_binary_function) gb_firstj_theta,
                GrB_INT64, GrB_FP64, GrB_FP64, GrB_INT64,
                "gb_firstj_theta", FIRSTJ_THETA_DEFN)) ;
        }
        else if (MATCH (multname, "firstj1"))
        {
            theta = 1 ;
            OK (GxB_IndexBinaryOp_new (&Iop,
                (GxB_index_binary_function) gb_firstj_theta,
                GrB_INT64, GrB_FP64, GrB_FP64, GrB_INT64,
                "gb_firstj_theta", FIRSTJ_THETA_DEFN)) ;
        }
        else if (MATCH (multname, "secondi" ))
        {
            theta = 0 ;
            OK (GxB_IndexBinaryOp_new (&Iop,
                (GxB_index_binary_function) gb_secondi_theta,
                GrB_INT64, GrB_FP64, GrB_FP64, GrB_INT64,
                "gb_secondi_theta", SECONDI_THETA_DEFN)) ;
        }
        else if (MATCH (multname, "secondi1"))
        {
            theta = 1 ;
            OK (GxB_IndexBinaryOp_new (&Iop,
                (GxB_index_binary_function) gb_secondi_theta,
                GrB_INT64, GrB_FP64, GrB_FP64, GrB_INT64,
                "gb_secondi_theta", SECONDI_THETA_DEFN)) ;
        }
        else if (MATCH (multname, "secondj" ))
        {
            theta = 0 ;
            OK (GxB_IndexBinaryOp_new (&Iop,
                (GxB_index_binary_function) gb_secondj_theta,
                GrB_INT64, GrB_FP64, GrB_FP64, GrB_INT64,
                "gb_secondj_theta", SECONDJ_THETA_DEFN)) ;
        }
        else if (MATCH (multname, "secondj1"))
        {
            theta = 1 ;
            OK (GxB_IndexBinaryOp_new (&Iop,
                (GxB_index_binary_function) gb_secondj_theta,
                GrB_INT64, GrB_FP64, GrB_FP64, GrB_INT64,
                "gb_secondj_theta", SECONDJ_THETA_DEFN)) ;
        }
        else
        {
            mexErrMsgTxt ("mult not supported") ;
        }

        // create the mult binary op
        OK (GrB_Scalar_new (&Theta, GrB_INT64)) ;
        OK (GrB_Scalar_setElement_INT64 (Theta, theta)) ;
        OK (GxB_BinaryOp_new_IndexOp (&mult, Iop, Theta)) ;
    }

    // create the semiring
    OK (GrB_Semiring_new (&semiring, monoid, mult)) ;
    // GxB_print (semiring, 5) ;

    // C = A*B, A'*B, A*B', or A'*B'
    OK (GrB_mxm (C, NULL, NULL, semiring, A, B, desc)) ;

    // return C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C AxB idx result", true) ;
    FREE_ALL ;
}

