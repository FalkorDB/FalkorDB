//------------------------------------------------------------------------------
// GB_mex_mxm_generic: C<Mask> = accum(C,A*B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_mxm_generic (C, Mask, accum, semiring, A, B, desc)"

#define FREE_ALL                                    \
{                                                   \
    GrB_Matrix_free_(&A) ;                          \
    GrB_Matrix_free_(&B) ;                          \
    GrB_Matrix_free_(&C) ;                          \
    GrB_Matrix_free_(&Mask) ;                       \
    GrB_Monoid_free_(&myplus_monoid) ;              \
    GrB_BinaryOp_free_(&myplus) ;                   \
    if (semiring != Complex_plus_times)             \
    {                                               \
        GrB_Semiring_free_(&semiring) ;             \
    }                                               \
    GrB_Descriptor_free_(&desc) ;                   \
    GB_mx_put_global (true) ;                       \
}

void gb_My_Plus_int64 (void *z, const void *x, const void *y) ;
void gb_My_Plus_int32 (void *z, const void *x, const void *y) ;
void gb_My_Plus_fp64  (void *z, const void *x, const void *y) ;

    void gb_My_Plus_int64 (void *z, const void *x, const void *y)
    {
        int64_t a = (*((int64_t *) x)) ;
        int64_t b = (*((int64_t *) y)) ;
        int64_t c = a + b ;
        (*((int64_t *) z)) = c ;
    }

#define  MY_PLUS_INT64                                              \
"   void gb_My_Plus_int64 (void *z, const void *x, const void *y)  \n" \
"   {                                                           \n" \
"       int64_t a = (*((int64_t *) x)) ;                        \n" \
"       int64_t b = (*((int64_t *) y)) ;                        \n" \
"       int64_t c = a + b ;                                     \n" \
"       (*((int64_t *) z)) = c ;                                \n" \
"   }"

    void gb_My_Plus_int32 (void *z, const void *x, const void *y)
    {
        int32_t a = (*((int32_t *) x)) ;
        int32_t b = (*((int32_t *) y)) ;
        int32_t c = a + b ;
        (*((int32_t *) z)) = c ;
    }

#define  MY_PLUS_INT32                                              \
"   void gb_My_Plus_int32 (void *z, const void *x, const void *y)  \n" \
"   {                                                           \n" \
"       int32_t a = (*((int32_t *) x)) ;                        \n" \
"       int32_t b = (*((int32_t *) y)) ;                        \n" \
"       int32_t c = a + b ;                                     \n" \
"       (*((int32_t *) z)) = c ;                                \n" \
"   }"

    void gb_My_Plus_fp64  (void *z, const void *x, const void *y)
    {
        double a = (*((double *) x)) ;
        double b = (*((double *) y)) ;
        double c = a + b ;
        (*((double *) z)) = c ;
    }

#define  MY_PLUS_FP64                                               \
"   void gb_My_Plus_fp64  (void *z, const void *x, const void *y)  \n" \
"   {                                                           \n" \
"       double a = (*((double *) x)) ;                          \n" \
"       double b = (*((double *) y)) ;                          \n" \
"       double c = a + b ;                                      \n" \
"       (*((double *) z)) = c ;                                 \n" \
"   }"

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
    GrB_Matrix B = NULL ;
    GrB_Matrix C = NULL ;
    GrB_Matrix Mask = NULL ;
    GrB_Semiring semiring = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_BinaryOp myplus = NULL ;
    GrB_Monoid   myplus_monoid = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 6 || nargin > 7)
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

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [4], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get B (shallow copy)
    B = GB_mx_mxArray_to_Matrix (pargin [5], "B input", false, true) ;
    if (B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("B failed") ;
    }

    bool user_complex = (Complex != GxB_FC64) && (C->type == Complex) ;

    // get semiring
    if (!GB_mx_mxArray_to_Semiring (&semiring, pargin [3], "semiring",  
        C->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("semiring failed") ;
    }

    if (semiring != NULL && semiring->add == GrB_PLUS_MONOID_INT64)
    { 
        // replace the semiring with a user-defined monoid
        GrB_BinaryOp mult = semiring->multiply ;
        GrB_Monoid_free_(&(semiring->add)) ;
        GrB_Semiring_free_(&semiring) ;

        // try with a NULL function pointer to test the JIT
        GrB_Info info = GxB_BinaryOp_new (&myplus, NULL,
            GrB_INT64, GrB_INT64, GrB_INT64,
            "gb_My_Plus_int64", MY_PLUS_INT64) ;
        if (info == GrB_NULL_POINTER)
        {
            // try again with non-NULL function pointer
            GxB_BinaryOp_new (&myplus, gb_My_Plus_int64,
                GrB_INT64, GrB_INT64, GrB_INT64,
                "gb_My_Plus_int64", MY_PLUS_INT64) ;
        }

        // add a spurious terminal value
        GxB_Monoid_terminal_new_INT64 (&myplus_monoid, myplus,
            (int64_t) 0, (int64_t) -111) ;
        GrB_Semiring_new (&semiring, myplus_monoid, mult) ;
    }
    else if (semiring != NULL && semiring->add == GrB_PLUS_MONOID_INT32)
    { 
        // replace the semiring with a user-defined monoid
        GrB_BinaryOp mult = semiring->multiply ;
        GrB_Monoid_free_(&(semiring->add)) ;
        GrB_Semiring_free_(&semiring) ;
//      GrB_BinaryOp_new (&myplus, gb_My_Plus_int32,
//          GrB_INT32, GrB_INT32, GrB_INT32) ;
        GxB_BinaryOp_new (&myplus, gb_My_Plus_int32,
            GrB_INT32, GrB_INT32, GrB_INT32,
            "gb_My_Plus_int32", MY_PLUS_INT32) ;
        // add a spurious terminal value
        GxB_Monoid_terminal_new_INT32 (&myplus_monoid, myplus,
            (int32_t) 0, (int32_t) -111) ;
        GrB_Semiring_new (&semiring, myplus_monoid, mult) ;
    }
    else if (semiring != NULL && semiring->add == GrB_PLUS_MONOID_FP64)
    { 
        // replace the semiring with a user-defined monoid
        GrB_BinaryOp mult = semiring->multiply ;
        GrB_Monoid_free_(&(semiring->add)) ;
        GrB_Semiring_free_(&semiring) ;
//      GrB_BinaryOp_new (&myplus, gb_My_Plus_fp64,
//          GrB_FP64, GrB_FP64, GrB_FP64) ;
        GxB_BinaryOp_new (&myplus, gb_My_Plus_fp64,
            GrB_FP64, GrB_FP64, GrB_FP64,
            "gb_My_Plus_fp64", MY_PLUS_FP64) ;
        GrB_Monoid_new_FP64 (&myplus_monoid, myplus, (double) 0) ;
        GrB_Semiring_new (&semiring, myplus_monoid, mult) ;
    }

    // get accum, if present
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        C->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (6), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // C<Mask> = accum(C,A*B)
    METHOD (GrB_mxm (C, Mask, accum, semiring, A, B, desc)) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output from GrB_mxm", true) ;

    FREE_ALL ;
}

