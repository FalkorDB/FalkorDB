//------------------------------------------------------------------------------
// GB_mex_argmax: compute [x,p]=argmax(A,dim,pr,jit)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is for testing only.  See GrB.argmax instead.

#include "GB_mex.h"
#include "GB_mex_errors.h"

 typedef struct { int64_t k ; double v ; } gb_tuple_kv ;
#define TUPLE_KV \
"typedef struct { int64_t k ; double v ; } gb_tuple_kv ;"

void gb_make_tuple_kv (gb_tuple_kv *z,
    const double *x, uint64_t ix, uint64_t jx,
    const void   *y, uint64_t iy, uint64_t jy,
    const void *theta) ;
void gb_make_tuple_kv (gb_tuple_kv *z,
    const double *x, uint64_t ix, uint64_t jx,
    const void   *y, uint64_t iy, uint64_t jy,
    const void *theta)
{
    z->k = (int64_t) jx + 1 ;
    z->v = (*x) ;
}

#define MAKE_TUPLE_KV_DEFN \
"void gb_make_tuple_kv (gb_tuple_kv *z,              \n" \
"    const double *x, uint64_t ix, uint64_t jx,      \n" \
"    const void   *y, uint64_t iy, uint64_t jy,      \n" \
"    const void *theta)                              \n" \
"{                                                   \n" \
"    z->k = (int64_t) jx + 1 ;                       \n" \
"    z->v = (*x) ;                                   \n" \
"}                                                   \n"

 void gb_getv_tuple_kv (double *z, const gb_tuple_kv *x) ;
 void gb_getv_tuple_kv (double *z, const gb_tuple_kv *x) { (*z) = x->v ; }
#define GETV_TUPLE_KV \
"void gb_getv_tuple_kv (double *z, const gb_tuple_kv *x) { (*z) = x->v ; }"

 void gb_getk_tuple_kv (int64_t *z, const gb_tuple_kv *x) ;
 void gb_getk_tuple_kv (int64_t *z, const gb_tuple_kv *x) { (*z) = x->k ; }
#define GETK_TUPLE_KV \
"void gb_getk_tuple_kv (int64_t *z, const gb_tuple_kv *x) { (*z) = x->k ; }"

void gb_max_tuple_kv (gb_tuple_kv *z, const gb_tuple_kv *x, const gb_tuple_kv *y) ;
void gb_max_tuple_kv (gb_tuple_kv *z, const gb_tuple_kv *x, const gb_tuple_kv *y)
{
    if (x->v > y->v || (x->v == y->v && x->k < y->k))
    {
        z->k = x->k ;
        z->v = x->v ;
    }
    else
    {
        z->k = y->k ;
        z->v = y->v ;
    }
}

#define MAX_TUPLE_KV \
"void gb_max_tuple_kv (gb_tuple_kv *z, const gb_tuple_kv *x, const gb_tuple_kv *y)\n" \
"{                                                                   \n" \
"    if (x->v > y->v || (x->v == y->v && x->k < y->k))               \n" \
"    {                                                               \n" \
"        z->k = x->k ;                                               \n" \
"        z->v = x->v ;                                               \n" \
"    }                                                               \n" \
"    else                                                            \n" \
"    {                                                               \n" \
"        z->k = y->k ;                                               \n" \
"        z->v = y->v ;                                               \n" \
"    }                                                               \n" \
"}                                                                   \n"

#define USAGE "[x,p] = GB_mex_argmax (A, dim, pr, jit)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&x) ;              \
    GrB_Matrix_free_(&p) ;              \
    GrB_Type_free (&Tuple) ;            \
    GxB_IndexBinaryOp_free (&Iop) ;     \
    GrB_BinaryOp_free (&Bop) ;          \
    GrB_BinaryOp_free (&MonOp) ;        \
    GrB_Monoid_free (&Monoid) ;         \
    GrB_Semiring_free (&Semiring) ;     \
    GrB_UnaryOp_free (&Getv) ;          \
    GrB_UnaryOp_free (&Getk) ;          \
    GrB_Matrix_free (&y) ;              \
    GrB_Matrix_free (&c) ;              \
    GrB_Scalar_free (&Theta) ;          \
    GrB_Scalar_free (&Beta) ;           \
    GrB_Scalar_free (&Gunk) ;           \
    GB_mx_put_global (true) ;           \
}

#define FREE_WORK FREE_ALL

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
    GrB_Matrix A = NULL ;

    GrB_Type Tuple = NULL ;
    GxB_IndexBinaryOp Iop = NULL ;
    GrB_BinaryOp Bop = NULL, MonOp = NULL ;
    GrB_Monoid Monoid = NULL ;
    GrB_Semiring Semiring = NULL ;
    GrB_Scalar Theta = NULL, Beta = NULL, Gunk = NULL ;
    GrB_UnaryOp Getv = NULL, Getk = NULL ;
    GrB_Matrix x = NULL, p = NULL, c = NULL, y = NULL, z = NULL ;
    GrB_Scalar s = NULL ;

    GB_WERK (USAGE) ;

    // check inputs
    if (nargout > 2 || nargin < 1 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("failed") ;
    }
    uint64_t nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    if (A->type != GrB_FP64)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A must be double") ;
    }

    // get dim, default is 2
    int dim = (nargin > 1) ? ((int) mxGetScalar (pargin [1])) : 2 ;
    if (!(dim == 1 || dim == 2))
    {
        dim = 1 ;
    }

    // get pr flag, default is false
    bool pr = (nargin > 2) ? ((bool) mxGetScalar (pargin [2])) : false ;

    // get jit flag, default is true
    bool jit = (nargin > 2) ? ((bool) mxGetScalar (pargin [3])) : true ;

    //--------------------------------------------------------------------------
    // create the types and operators
    //--------------------------------------------------------------------------

    OK (GrB_Scalar_new (&Theta, GrB_BOOL)) ;
    OK (GrB_Scalar_setElement_BOOL (Theta, 1)) ;
    if (jit)
    {
        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_kv), "gb_tuple_kv", TUPLE_KV)) ;
        METHOD (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_tuple_kv,
            Tuple, GrB_FP64, GrB_BOOL, GrB_BOOL,
            "gb_make_tuple_kv", MAKE_TUPLE_KV_DEFN)) ;
    }
    else
    {
        OK (GrB_Type_new (&Tuple, sizeof (gb_tuple_kv))) ;
        METHOD (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_tuple_kv,
            Tuple, GrB_FP64, GrB_BOOL, GrB_BOOL,
            NULL, NULL)) ;
    }
    OK (GxB_IndexBinaryOp_wait (Iop, GrB_MATERIALIZE)) ;
    const char *error ;
    OK (GxB_IndexBinaryOp_error (&error, Iop)) ;
    if (error == NULL || strlen (error) > 0)
    {
        mexErrMsgTxt ("index binary op failed") ;
    }
    METHOD (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
    if (pr)
    {
        // printf ("\njit enabled: %d\n", jit) ;
        OK (GxB_IndexBinaryOp_fprint (Iop, "gb_make_tuple_kv idx", 5, stdout)) ;
    }

    gb_tuple_kv id ;
    memset (&id, 0, sizeof (gb_tuple_kv)) ;
    id.k = INT64_MAX ;
    id.v = (double) (-INFINITY) ;

    if (jit)
    {
        OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_tuple_kv,
            Tuple, Tuple, Tuple, "gb_max_tuple_kv", MAX_TUPLE_KV)) ;
    }
    else
    {
        OK (GrB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_tuple_kv,
            Tuple, Tuple, Tuple)) ;
    }

    OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
    OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;

    size_t namelen = 0 ;
    OK (GrB_Semiring_get_SIZE (Semiring, &namelen, GxB_THETA_TYPE_STRING)) ;
    printf ("theta namelen: %d\n", (int) namelen) ;
    CHECK (namelen == strlen ("GrB_BOOL") + 1) ;
    char theta_type_name [256] ;
    theta_type_name [0] = '\0' ;
    OK (GrB_Semiring_get_String (Semiring, theta_type_name,
        GxB_THETA_TYPE_STRING)) ;
    printf ("theta type: [%s]\n", theta_type_name) ;
    CHECK (strcmp (theta_type_name, "GrB_BOOL") == 0) ;

    if (jit)
    {
        OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_tuple_kv,
            GrB_INT64, Tuple, "gb_getk_tuple_kv", GETK_TUPLE_KV)) ;
        OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_tuple_kv,
            GrB_FP64, Tuple, "gb_getv_tuple_kv", GETV_TUPLE_KV)) ;
    }
    else
    {
        OK (GrB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_tuple_kv,
            GrB_INT64, Tuple)) ;
        OK (GrB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_tuple_kv,
            GrB_FP64, Tuple)) ;
    }

    if (pr)
    {
        OK (GxB_Semiring_fprint (Semiring, "(max,maketuple)", 5, stdout)) ;
        OK (GxB_UnaryOp_fprint (Getk, "Getk", 5, stdout)) ;
        OK (GxB_UnaryOp_fprint (Getv, "Getv", 5, stdout)) ;
    }

    //--------------------------------------------------------------------------
    // test get/set
    //--------------------------------------------------------------------------

    OK (GrB_Scalar_new (&Beta, GrB_INT64)) ;
    OK (GxB_IndexBinaryOp_get_Scalar (Iop, Beta, GrB_OUTP_TYPE_CODE)) ;
    int32_t code = -1;
    OK (GrB_Scalar_extractElement_INT32 (&code, Beta)) ;
    // printf ("code %d\n", code) ;
    CHECK (code == GrB_UDT_CODE) ;
    code = 62 ;
    OK (GxB_IndexBinaryOp_get_INT32 (Iop, &code, GrB_OUTP_TYPE_CODE)) ;
    CHECK (code == GrB_UDT_CODE) ;
    size_t name_size ;
    OK (GxB_IndexBinaryOp_get_SIZE (Iop, &name_size, GxB_JIT_C_NAME)) ;
    // printf ("name size %d\n", (int) name_size) ;
    char name [256] ;
    OK (GxB_IndexBinaryOp_get_String (Iop, name, GxB_JIT_C_NAME)) ;
    // printf ("name [%s]\n", name) ;
    int expected = GrB_INVALID_VALUE ;
    ERR (GxB_IndexBinaryOp_get_VOID (Iop, name, GxB_JIT_C_NAME)) ;

    OK (GxB_IndexBinaryOp_set_String (Iop, "my index binop", GrB_NAME)) ;
    name [0] = '\0' ;
    OK (GxB_IndexBinaryOp_get_String (Iop, name, GrB_NAME)) ;
    // printf ("name [%s]\n", name) ;
    CHECK (strcmp (name, "my index binop") == 0) ;

    expected = GrB_DOMAIN_MISMATCH ;
    OK (GrB_Scalar_new (&Gunk, Tuple)) ;
    ERR (GrB_BinaryOp_get_Scalar (Bop, Gunk, GxB_THETA)) ;

    //--------------------------------------------------------------------------
    // compute [x,p] = argmax (A,dim)
    //--------------------------------------------------------------------------

    if (dim == 1)
    { 

        //------------------------------------------------------------------
        // argmin/max of each column of A
        //------------------------------------------------------------------

        // y = zeros (nrows,1) ;
        OK (GrB_Matrix_new (&y, GrB_BOOL, nrows, 1)) ;
        OK (GrB_Matrix_assign_BOOL (y, NULL, NULL, 0,
            GrB_ALL, nrows, GrB_ALL, 1, NULL)) ;

        // c = A'*y using the argmin/argmax semiring
        OK (GrB_Matrix_new (&c, Tuple, ncols, 1)) ;
        OK (GrB_mxm (c, NULL, NULL, Semiring, A, y, GrB_DESC_T0)) ;

        // create x and p
        OK (GrB_Matrix_new (&x, GrB_FP64, ncols, 1)) ;
        OK (GrB_Matrix_new (&p, GrB_INT64, ncols, 1)) ;

    }
    else
    { 

        //------------------------------------------------------------------
        // argmin/max of each row of A
        //------------------------------------------------------------------

        // y = zeros (ncols,1) ;
        OK (GrB_Matrix_new (&y, GrB_BOOL, ncols, 1)) ;
        OK (GrB_Matrix_assign_BOOL (y, NULL, NULL, 0,
            GrB_ALL, ncols, GrB_ALL, 1, NULL)) ;

        // c = A*y using the argmin/argmax semiring
        OK (GrB_Matrix_new (&c, Tuple, nrows, 1)) ;
        OK (GrB_mxm (c, NULL, NULL, Semiring, A, y, NULL)) ;

        // create x and p
        OK (GrB_Matrix_new (&x, GrB_FP64, nrows, 1)) ;
        OK (GrB_Matrix_new (&p, GrB_INT64, nrows, 1)) ;
    }

    // x = getv (c)
    OK (GrB_Matrix_apply (x, NULL, NULL, Getv, c, NULL)) ;
    // p = getk (c)
    OK (GrB_Matrix_apply (p, NULL, NULL, Getk, c, NULL)) ;

    //--------------------------------------------------------------------------
    // return x and p as MATLAB sparse matrices
    //--------------------------------------------------------------------------

    pargout [0] = GB_mx_Matrix_to_mxArray (&x, "x result", false) ;
    pargout [1] = GB_mx_Matrix_to_mxArray (&p, "p result", false) ;
    FREE_ALL ;
}

