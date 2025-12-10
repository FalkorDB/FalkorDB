//------------------------------------------------------------------------------
// GB_mex_test0: test error handling
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This mexFunction intentionally creates many errors for GraphBLAS, to test
// error-handling.  Many error messages are printed.  If the test ends with
// "All tests passed" then all errors were expected.  The test fails if
// GraphBLAS does not catch the error with the right error code, or if it
// generates an unexpected error.

#include "GB_mex.h"

#define USAGE "GB_mex_test0"

#define FREE_ALL                                                          \
{                                                                         \
    GrB_Matrix_free_(&Empty1) ;       CHECK (Empty1       == NULL) ;      \
    GrB_Matrix_free_(&A) ;            CHECK (A            == NULL) ;      \
    GrB_Matrix_free_(&B) ;            CHECK (B            == NULL) ;      \
    GrB_Matrix_free_(&C) ;            CHECK (C            == NULL) ;      \
    GrB_Matrix_free_(&Z) ;            CHECK (Z            == NULL) ;      \
    GrB_Matrix_free_(&E) ;            CHECK (E            == NULL) ;      \
    GrB_Matrix_free_(&F) ;            CHECK (F            == NULL) ;      \
    GrB_Matrix_free_(&H) ;            CHECK (H            == NULL) ;      \
    GrB_Matrix_free_(&Acrud) ;        CHECK (Acrud        == NULL) ;      \
    GrB_Matrix_free_(&Aempty) ;       CHECK (Aempty       == NULL) ;      \
    GrB_Type_free_(&T) ;              CHECK (T            == NULL) ;      \
    GrB_Type_free_(&Tcrud) ;          CHECK (Tcrud        == NULL) ;      \
    GrB_UnaryOp_free_(&op1) ;         CHECK (op1          == NULL) ;      \
    GrB_UnaryOp_free_(&op1crud) ;     CHECK (op1crud      == NULL) ;      \
    GrB_BinaryOp_free_(&op2) ;        CHECK (op2          == NULL) ;      \
    GrB_BinaryOp_free_(&op3) ;        CHECK (op3          == NULL) ;      \
    GrB_UnaryOp_free_(&op1b) ;        CHECK (op1b         == NULL) ;      \
    GrB_BinaryOp_free_(&op2b) ;       CHECK (op2b         == NULL) ;      \
    GrB_Monoid_free_(&monoidb) ;      CHECK (monoidb      == NULL) ;      \
    GrB_Semiring_free_(&semiring2) ;  CHECK (semiring2    == NULL) ;      \
    GrB_Descriptor_free_(&descb) ;    CHECK (descb        == NULL) ;      \
    GrB_Vector_free_(&vb) ;           CHECK (vb           == NULL) ;      \
    GrB_BinaryOp_free_(&op2crud) ;    CHECK (op2crud      == NULL) ;      \
    GrB_Monoid_free_(&monoid) ;       CHECK (monoid       == NULL) ;      \
    GrB_Monoid_free_(&monoid_crud) ;  CHECK (monoid_crud  == NULL) ;      \
    GrB_Semiring_free_(&semiring) ;   CHECK (semiring     == NULL) ;      \
    GrB_Semiring_free_(&semicrud) ;   CHECK (semicrud     == NULL) ;      \
    GrB_Vector_free_(&v) ;            CHECK (v            == NULL) ;      \
    GrB_Vector_free_(&w) ;            CHECK (w            == NULL) ;      \
    GrB_Vector_free_(&u) ;            CHECK (u            == NULL) ;      \
    GrB_Vector_free_(&z) ;            CHECK (z            == NULL) ;      \
    GrB_Vector_free_(&h) ;            CHECK (h            == NULL) ;      \
    GrB_Vector_free_(&vcrud) ;        CHECK (vcrud        == NULL) ;      \
    GrB_Vector_free_(&vempty) ;       CHECK (vempty       == NULL) ;      \
    GrB_Descriptor_free_(&desc) ;     CHECK (desc         == NULL) ;      \
    GrB_Descriptor_free_(&dtn) ;      CHECK (dtn          == NULL) ;      \
    GrB_Descriptor_free_(&dnt) ;      CHECK (dnt          == NULL) ;      \
    GrB_Descriptor_free_(&dtt) ;      CHECK (dtt          == NULL) ;      \
    GrB_Descriptor_free_(&dcrud) ;    CHECK (dcrud        == NULL) ;      \
    GrB_Scalar_free_(&a_scalar) ;                                         \
    GB_mx_put_global (true) ;                                             \
}

#include "GB_mex_errors.h"

#define G3 GxB_COMPLETE
#define G2 GxB_SHORT
#define G1 GxB_SUMMARY
#define G0 GxB_SILENT

typedef int16_t gb_user_int16 ;

void f1 (double *z, const uint32_t *x) ;
void f2 (int32_t *z, const uint8_t *x, const int16_t *y) ;

void f1 (double *z, const uint32_t *x)
{
    (*z) = (*x) + 1 ;
}

void f2 (int32_t *z, const uint8_t *x, const int16_t *y)
{
    (*z) = (*x) + (*y) + 1 ;
}

void f3 (GxB_FC64_t *z, const GxB_FC64_t *x, const double *y) ;
void f3 (GxB_FC64_t *z, const GxB_FC64_t *x, const double *y)
{
    (*z) = GB_FC64_add ((*x), GB_CMPLX64 (0,(*y))) ;
}

bool fselect (uint64_t i, uint64_t j, const double *x, const double *k) ;
bool fselect (uint64_t i, uint64_t j, const double *x, const double *k)
{
    // select entries in triu(A) that are greater than k
    int64_t i2 = (int64_t) i ;
    int64_t j2 = (int64_t) j ;
    return (x > k && (j2-i2) > 0) ;
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    FILE *f = fopen ("errlog.txt", "w") ;
    FILE *ff = fopen ("fprint.txt", "w") ;

    GrB_Info info, expected  ;

    // finalize, but tell GraphBLAS that GrB_init can be called again:
    GB_mx_at_exit ( ) ;
    OK (GrB_init (GrB_NONBLOCKING)) ;

    // finalize, but tell GraphBLAS that GrB_init can be called again:
    GB_mx_at_exit ( ) ;

    OK (GxB_init (GrB_NONBLOCKING, mxMalloc, NULL, NULL, mxFree)) ;

    // mxMalloc, mxCalloc, mxRealloc, and mxFree are not thread safe
    GB_Global_malloc_is_thread_safe_set (false) ;

    GB_Global_abort_set (GB_mx_abort) ;
    GB_Global_malloc_tracking_set (true) ;

    fprintf (f,"\n========================================================\n") ;
    fprintf (f,"=== GB_mex_test0 : testing error handling =============\n") ;
    fprintf (f,"========================================================\n") ;
    fprintf (f,"many errors are expected\n") ;

    OK (GxB_Type_fprint_(GrB_BOOL, G3, ff)) ;
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Type_fprint_(GrB_BOOL, G3, stdin)) ;

    int64_t nmalloc ;
    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("nmalloc %d at start\n", nmalloc) ;
    bool malloc_debug = GB_mx_get_global (true) ;
    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("nmalloc %d after complex init\n", nmalloc) ;

    GrB_Matrix A = NULL, B = NULL, C = NULL, Z = NULL, Acrud = NULL,
               Aempty = NULL, E = NULL, F = NULL, A0 = NULL, H = NULL,
               Empty1 = NULL, A4 = NULL, C4 = NULL ;
    GrB_Vector v = NULL, vcrud = NULL, vempty = NULL, w = NULL, u = NULL,
               v0 = NULL, v2 = NULL, z = NULL, h = NULL, vb = NULL ;
    GrB_Type T = NULL, Tcrud ;
    GrB_UnaryOp op1 = NULL, op1crud = NULL, o1 = NULL, op1b = NULL ;
    GrB_BinaryOp op2 = NULL, op2crud = NULL, o2 = NULL, op0 = NULL, op3 = NULL,
                op2b = NULL ;
    GrB_Monoid monoid = NULL, monoid_crud = NULL, m2 = NULL, m0 = NULL,
        monoidb = NULL ;
    GrB_Semiring semiring = NULL, semicrud = NULL, s2 = NULL, s0 = NULL,
        semiring2 = NULL ;
    GrB_Descriptor desc = NULL, dcrud = NULL, d0 = NULL,
        dnt = NULL, dtn = NULL, dtt = NULL, descb = NULL ;
    int dval ;
    uint64_t n = 0, nvals = 0, n2 = 0, i = 0, j = 0, a, b, uvals = 0 ;
    uint64_t *I0 = NULL, *J0 = NULL ;   // OK
    #define LEN 100
    uint64_t I [5] = { 0,   7,   8,   3,    2 },       I2 [LEN] ;   // OK
    uint64_t J [5] = { 4,   1,   2,   2,    1 },       J2 [LEN] ;   // OK
    double   X [5] = { 4.5, 8.2, 9.1, -1.2, 3.14159 }, X2 [LEN] ;
    GB_Pending A_Pending = NULL ;

    size_t s ;
    bool        x_bool, ok ;
    int8_t      x_int8 ;
    int16_t     x_int16 ;
    uint16_t    x_uint16 ;
    int32_t     x_int32 ;
    uint32_t    x_uint32 ;
    int64_t     x_int64 ;
    uint64_t    x_uint64 ;
    float       x_float ;
    double      x_double, x = 0 ;
    GxB_FC64_t  c ;

    GB_void *pp = NULL ;

    GrB_Scalar a_scalar = NULL ;

    const char *err ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WERK (USAGE) ;
    if (nargout > 0 || nargin > 0)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    //--------------------------------------------------------------------------
    // initialize simple_rand
    //--------------------------------------------------------------------------

    printf ("rand seed----------------------------------------------------\n") ;
    fprintf (f, "random seed is %g\n", (double) simple_rand_getseed ( )) ;
    simple_rand_seed (1) ;
    fprintf (f, "random seed is now %g\n", (double) simple_rand_getseed ( )) ;

    //--------------------------------------------------------------------------
    // init
    //--------------------------------------------------------------------------

    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("nmalloc %d befor 2nd start\n", nmalloc) ;
    OK (Complex_finalize ( )) ;
    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("nmalloc %d after Complex_finialize\n", nmalloc) ;

    printf ("GrB_init-----------------------------------------------------\n") ;

    // can't call it twice
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_init (GrB_NONBLOCKING, mxMalloc, NULL, NULL, mxFree)) ;
    GB_mx_at_exit ( ) ;

    // invalid mode
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_init (42, mxMalloc, NULL, NULL, mxFree)) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_init (GrB_NONBLOCKING, NULL    , NULL, NULL, mxFree)) ;
    ERR (GxB_init (GrB_NONBLOCKING, NULL    , NULL, NULL, mxFree)) ;
    ERR (GxB_init (GrB_NONBLOCKING, mxMalloc, NULL, NULL, NULL  )) ;

    OK (GxB_init (GrB_NONBLOCKING, mxMalloc, NULL, NULL, mxFree)) ;

    nmalloc = GB_Global_nmalloc_get ( ) ;
    bool complex_is_builtin = (Complex == GxB_FC64) ;
    printf ("nmalloc %d at 2nd start, reinit complex with builtin: %d\n",
        nmalloc, complex_is_builtin) ;
    OK (Complex_init (complex_is_builtin)) ;
    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("nmalloc %d after 2nd complex init\n", nmalloc) ;

    //--------------------------------------------------------------------------
    // Type
    //--------------------------------------------------------------------------

    printf ("GrB_Type-----------------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Type_new (NULL, sizeof (int))) ;
    ERR (GxB_Type_size (NULL, NULL)) ;
    ERR (GxB_Type_size (&s, NULL)) ;

    OK (GrB_Type_new (&T, sizeof (int))) ;
    CHECK (T != NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    Tcrud = T ;
    T = NULL ;
    Tcrud->magic = 42 ;
    ERR (GxB_Type_size (&s, Tcrud)) ;

    T = GrB_INT32 ;
    OK (GrB_Type_free_(&GrB_INT32)) ;
    CHECK (GrB_INT32 == T) ;
    T = NULL ;

    OK (GrB_Type_new (&T, sizeof (int))) ;
    CHECK (T != NULL) ;

    OK (GxB_Type_size (&s, T)) ;
    CHECK (s == sizeof (int)) ;
    s = 0 ;

    OK (GrB_Type_free_(&T)) ;
    CHECK (T == NULL) ;

    OK (GrB_Type_free_(&T)) ;
    CHECK (T == NULL) ;

    s = GB_code_size (-1,1) ;
    CHECK (s == 0) ;

    #define FREE_DEEP_COPY ;
    #define GET_DEEP_COPY ;

    METHOD (GrB_Type_new (&T, sizeof (int))) ;
    OK (GB_Type_check (T, "new type", G3, NULL)) ;
    OK (GxB_Type_fprint (T, "new type", G3, ff)) ;
    OK (GrB_Type_free_(&T)) ;
    CHECK (T == NULL) ;

    #undef FREE_DEEP_COPY
    #undef GET_DEEP_COPY

    for (GB_Type_code tcode = 0 ; tcode <= GB_UDT_code ; tcode++)
    {
        GrB_Type utype = Complex ;
        GrB_Type ttype = GB_code_type (tcode, utype) ;
        printf ("\n----------------------------------tcode: %d\n", tcode) ;
        OK (GB_Type_check (ttype, "GB_code_type:", G3, NULL)) ;
        OK (GxB_Type_fprint_(ttype, G3, ff)) ;
    }

    // Tcrud is allocated but uninitialized

    //--------------------------------------------------------------------------
    // UnaryOp
    //--------------------------------------------------------------------------

    printf ("GrB_UnaryOp--------------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_UnaryOp_new (NULL, NULL, NULL, NULL)) ;

    ERR (GrB_UnaryOp_new (&op1, NULL, NULL, NULL)) ;
    CHECK (op1 == NULL) ;

    ERR (GrB_UnaryOp_new (&op1, (GxB_unary_function) f1, NULL, NULL)) ;
    CHECK (op1 == NULL) ;

    ERR (GrB_UnaryOp_new (&op1, (GxB_unary_function) f1, GrB_FP64, NULL)) ;
    CHECK (op1 == NULL) ;

    ERR (GrB_UnaryOp_new (&op1, (GxB_unary_function) f1, NULL, GrB_UINT32)) ;
    CHECK (op1 == NULL) ;

    ERR (GrB_UnaryOp_new (&op1, NULL, GrB_FP64, GrB_UINT32)) ;
    CHECK (op1 == NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_UnaryOp_new (&op1, (GxB_unary_function) f1, Tcrud, Tcrud)) ;
    CHECK (op1 == NULL) ;

    ERR (GrB_UnaryOp_new (&op1, (GxB_unary_function) f1, GrB_FP64, Tcrud)) ;
    CHECK (op1 == NULL) ;

    ERR (GrB_UnaryOp_new (&op1, (GxB_unary_function) f1, Tcrud, GrB_FP64)) ;
    CHECK (op1 == NULL) ;

    OK (GrB_UnaryOp_new (&op1, (GxB_unary_function) f1, GrB_FP64, GrB_UINT32)) ;
    CHECK (op1 != NULL) ;

    expected = GrB_NULL_POINTER ;
    T = NULL ;

    ERR (GxB_UnaryOp_ztype (NULL, op1)) ;
    ERR (GxB_UnaryOp_xtype (NULL, op1)) ;

    ERR (GxB_UnaryOp_ztype (&T, NULL)) ;
    CHECK (T == NULL) ;

    ERR (GxB_UnaryOp_xtype (&T, NULL)) ;
    CHECK (T == NULL) ;

    OK (GxB_UnaryOp_ztype (&T, op1)) ;
    CHECK (T == GrB_FP64) ;

    OK (GxB_UnaryOp_xtype (&T, op1)) ;
    CHECK (T == GrB_UINT32) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    op1crud = op1 ;
    op1 = NULL ;

    op1crud->magic = 99 ;
    T = NULL ;

    ERR (GxB_UnaryOp_ztype (&T, op1crud)) ;
    CHECK (T == NULL) ;

    ERR (GxB_UnaryOp_xtype (&T, op1crud)) ;
    CHECK (T == NULL) ;

    o1 = GrB_IDENTITY_BOOL ;
    OK (GrB_UnaryOp_free_(&o1)) ;
    CHECK (o1 == GrB_IDENTITY_BOOL) ;
    o1 = NULL ;

    OK (GrB_UnaryOp_new (&o1, (GxB_unary_function) f1, GrB_FP64, GrB_UINT32)) ;
    CHECK (o1 != NULL) ;

    OK (GrB_UnaryOp_free_(&o1)) ;
    o1 = NULL ;

    OK (GrB_UnaryOp_free_(&o1)) ;
    o1 = NULL ;

    #define FREE_DEEP_COPY ;
    #define GET_DEEP_COPY ;

    GrB_UnaryOp opzz ;
    METHOD (GrB_UnaryOp_new (&opzz, (GxB_unary_function) f1, GrB_FP64, GrB_UINT32)) ;
    OK (GB_UnaryOp_check (opzz, "new unary opzz", G3, NULL)) ;
    OK (GxB_UnaryOp_fprint (opzz, "new unary opzz", G3, ff)) ;
    OK (GrB_UnaryOp_free_(&opzz)) ;
    CHECK (opzz == NULL) ;

    #undef FREE_DEEP_COPY
    #undef GET_DEEP_COPY

    // op1crud is allocated but uninitialized

    //--------------------------------------------------------------------------
    // BinaryOp
    //--------------------------------------------------------------------------

    printf ("GrB_BinaryOp-------------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;
    ERR (GrB_BinaryOp_new (NULL, NULL, NULL, NULL, NULL)) ;

    ERR (GrB_BinaryOp_new (&op2, NULL, NULL, NULL, NULL)) ;
    CHECK (op2 == NULL) ;

    // void f2 (int32_t *z, uint8_t *x, int16_t *y)
    ERR (GrB_BinaryOp_new (&op2, (GxB_binary_function) f2, NULL, NULL, NULL)) ;
    CHECK (op2 == NULL) ;

    ERR (GrB_BinaryOp_new (&op2, (GxB_binary_function) f2, GrB_INT32, NULL, NULL)) ;
    CHECK (op2 == NULL) ;

    ERR (GrB_BinaryOp_new (&op2, (GxB_binary_function) f2, GrB_INT32, GrB_UINT8, NULL)) ;
    CHECK (op2 == NULL) ;

    ERR (GrB_BinaryOp_new (&op2, (GxB_binary_function) f2, GrB_INT32, NULL, GrB_INT16)) ;
    CHECK (op2 == NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_BinaryOp_new (&op2, (GxB_binary_function) f2, Tcrud, GrB_UINT8, GrB_INT16)) ;
    CHECK (op2 == NULL) ;

    ERR (GrB_BinaryOp_new (&op2, (GxB_binary_function) f2, GrB_INT32, Tcrud, GrB_INT16)) ;
    CHECK (op2 == NULL) ;

    ERR (GrB_BinaryOp_new (&op2, (GxB_binary_function) f2, GrB_INT32, GrB_UINT8, Tcrud)) ;
    CHECK (op2 == NULL) ;

    OK (GrB_BinaryOp_new (&op2, (GxB_binary_function) f2, GrB_INT32, GrB_UINT8, GrB_INT16)) ;
    CHECK (op2 != NULL) ;

    expected = GrB_NULL_POINTER ;

    ERR (GxB_BinaryOp_ztype (NULL, op2)) ;
    ERR (GxB_BinaryOp_xtype (NULL, op2)) ;
    ERR (GxB_BinaryOp_ytype (NULL, op2)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    op2crud = op2 ;
    op2 = NULL ;
    op2crud->magic = 77 ;
    T = NULL ;

    ERR (GxB_BinaryOp_ztype (&T, op2crud)) ;
    CHECK (T == NULL) ;

    ERR (GxB_BinaryOp_xtype (&T, op2crud)) ;
    CHECK (T == NULL) ;

    ERR (GxB_BinaryOp_ytype (&T, op2crud)) ;
    CHECK (T == NULL) ;

    o2 = GrB_PLUS_FP64 ;
    OK (GrB_BinaryOp_free_(&o2)) ;
    CHECK (o2 == GrB_PLUS_FP64) ;
    o2 = NULL ;

    OK (GrB_BinaryOp_new (&o2, (GxB_binary_function) f2, GrB_INT32, GrB_UINT8, GrB_INT16)) ;
    CHECK (o2 != NULL) ;

    OK (GrB_BinaryOp_free_(&o2)) ;
    CHECK (o2 == NULL) ;

    OK (GrB_BinaryOp_free_(&o2)) ;
    CHECK (o2 == NULL) ;

    #define FREE_DEEP_COPY ;
    #define GET_DEEP_COPY ;

    GrB_BinaryOp opxx ;
    METHOD (GrB_BinaryOp_new (&opxx, (GxB_binary_function) f2, GrB_INT32, GrB_UINT8, GrB_INT16)) ;
    OK (GB_BinaryOp_check (opxx, "new binary opxx", G3, NULL)) ;
    OK (GxB_BinaryOp_fprint (opxx, "new binary opxx", G3, ff)) ;
    OK (GxB_BinaryOp_fprint_(opxx, G3, ff)) ;
    OK (GrB_BinaryOp_free_(&opxx)) ;
    CHECK (opxx == NULL) ;

    #undef FREE_DEEP_COPY
    #undef GET_DEEP_COPY

    //--------------------------------------------------------------------------
    // Monoid
    //--------------------------------------------------------------------------

    printf ("GrB_Monoid---------------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Monoid_new_BOOL    (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_INT8    (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_UINT8   (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_INT16   (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_UINT16  (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_INT32   (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_UINT32  (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_INT64   (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_UINT64  (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_FP32    (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_FP64    (NULL, NULL, 0)) ;
    ERR (GrB_Monoid_new_UDT     (NULL, NULL, NULL)) ;

    ERR (GrB_Monoid_new_BOOL    (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_INT8    (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_UINT8   (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_INT16   (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_UINT16  (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_INT32   (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_UINT32  (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_INT64   (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_UINT64  (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_FP32    (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_FP64    (&monoid, NULL, 0)) ; CHECK (monoid == NULL) ;
    ERR (GrB_Monoid_new_UDT     (&monoid, NULL, NULL)) ; CHECK (monoid == NULL);

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_Monoid_new_BOOL    (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);
    ERR (GrB_Monoid_new_INT8    (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);
    ERR (GrB_Monoid_new_UINT8   (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);
    ERR (GrB_Monoid_new_INT16   (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);
    ERR (GrB_Monoid_new_UINT16  (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);
    ERR (GrB_Monoid_new_INT32   (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);
    ERR (GrB_Monoid_new_UINT32  (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);
    ERR (GrB_Monoid_new_INT64   (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);
    ERR (GrB_Monoid_new_UINT64  (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);
    ERR (GrB_Monoid_new_FP32    (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);
    ERR (GrB_Monoid_new_FP64    (&monoid, op2crud, 0)) ; CHECK (monoid == NULL);

    expected = GrB_NULL_POINTER ;
    ERR (GrB_Monoid_new_UDT (&monoid, op2crud, NULL)) ; CHECK (monoid == NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;
    ERR (GrB_Monoid_new_INT32 (&monoid, op2crud, 0)) ;
    CHECK (monoid == NULL) ;

    expected = GrB_NULL_POINTER ;
    ERR (GrB_Monoid_new_UDT (&monoid, GrB_PLUS_FP64, NULL)) ;
    CHECK (monoid == NULL) ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR (GrB_Monoid_new_FP64 (&monoid, GrB_EQ_FP64, (double) 0)) ;
    CHECK (monoid == NULL) ;

    // These feel like they should work, but '0' becomes int, and it does not
    // match the type of the operator.  So it is expected to fail with a
    // domain mismatch.
    ERR (GrB_Monoid_new_INT32 (&monoid, GrB_PLUS_FP64, 0)) ;
    CHECK (monoid == NULL) ;

    // likewise, all these fail:
    ERR (GrB_Monoid_new_BOOL (&monoid, GrB_PLUS_FP64, (bool) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GrB_Monoid_new_INT8 (&monoid, GrB_PLUS_FP64, (int8_t) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GrB_Monoid_new_UINT8 (&monoid, GrB_PLUS_FP64, (uint8_t) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GrB_Monoid_new_INT16 (&monoid, GrB_PLUS_FP64, (int16_t) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GrB_Monoid_new_UINT16 (&monoid, GrB_PLUS_FP64, (uint16_t) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GrB_Monoid_new_INT32 (&monoid, GrB_PLUS_FP64, (int32_t) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GrB_Monoid_new_UINT32 (&monoid, GrB_PLUS_FP64, (uint32_t) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GrB_Monoid_new_INT64 (&monoid, GrB_PLUS_FP64, (int64_t) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GrB_Monoid_new_UINT64 (&monoid, GrB_PLUS_FP64, (uint64_t) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GrB_Monoid_new_FP32 (&monoid, GrB_PLUS_FP64, (float) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GrB_Monoid_new_FP64 (&monoid, GrB_PLUS_FP32, (double) 0)) ;
    CHECK (monoid == NULL) ;

    ERR (GxB_Monoid_new_FC32 (&monoid, GrB_PLUS_FP64, GxB_CMPLXF(0,0))) ;
    CHECK (monoid == NULL) ;

    ERR (GxB_Monoid_new_FC64 (&monoid, GrB_PLUS_FP32, GxB_CMPLX (0,0))) ;
    CHECK (monoid == NULL) ;

    // this works
    OK (GrB_Monoid_new_FP64 (&monoid, GrB_PLUS_FP64, (double) 0)) ;
    CHECK (monoid != NULL) ;

    expected = GrB_NULL_POINTER ;

    ERR (GxB_Monoid_identity (NULL, NULL)) ;
    ERR (GxB_Monoid_identity (NULL, monoid)) ;
    x_double = 97.0 ;
    ERR (GxB_Monoid_identity (&x_double, NULL)) ;
    CHECK (x_double == 97.0) ;

    OK (GxB_Monoid_identity (&x_double, GxB_TIMES_FP64_MONOID)) ;
    CHECK (x_double == 1.0) ;

    bool has_terminal = true ;
    x_double = 42.0 ;
    OK (GxB_Monoid_terminal (&has_terminal, &x_double, GxB_TIMES_FP64_MONOID)) ;
    CHECK (!has_terminal) ;
    CHECK (x_double == 42.0) ;

    OK (GxB_Monoid_terminal (&has_terminal, &x_int64, GxB_MAX_INT64_MONOID)) ;
    CHECK (has_terminal) ;
    CHECK (x_int64 == INT64_MAX) ;

    ERR (GxB_Monoid_terminal (NULL, NULL, GxB_MAX_FP64_MONOID)) ;
    ERR (GxB_Monoid_terminal (&has_terminal, NULL, GxB_MAX_FP64_MONOID)) ;
    ERR (GxB_Monoid_terminal (NULL, &x_double, GxB_MAX_FP64_MONOID)) ;
    ERR (GxB_Monoid_terminal (&has_terminal, &x_double, NULL)) ;

    monoid_crud = monoid ;
    monoid_crud->magic = 8080 ;
    monoid = NULL ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    x_double = 33. ;
    ERR (GxB_Monoid_identity (&x_double, monoid_crud)) ;
    CHECK (x_double == 33.) ;

    expected = GrB_NULL_POINTER ;

    ERR (GxB_Monoid_operator (NULL, NULL)) ;
    ERR (GxB_Monoid_operator (NULL, GxB_TIMES_FP64_MONOID)) ;
    ERR (GxB_Monoid_operator (&o2, NULL)) ;
    CHECK (o2 == NULL) ;

    OK (GxB_Monoid_operator (&o2, GxB_TIMES_FP64_MONOID)) ;
    CHECK (o2 == GrB_TIMES_FP64) ;

    m2 = GxB_TIMES_FP64_MONOID ;
    OK (GrB_Monoid_free_(&m2)) ;
    CHECK (m2 == GxB_TIMES_FP64_MONOID) ;
    m2 = NULL ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    o2 = NULL ;
    ERR (GxB_Monoid_operator (&o2, monoid_crud)) ;
    CHECK (o2 == NULL) ;

    OK (GrB_Monoid_new_FP64 (&m2, GrB_PLUS_FP64, (double) 0)) ;
    CHECK (m2 != NULL) ;

    OK (GrB_Monoid_free_(&m2)) ;
    CHECK (m2 == NULL) ;

    OK (GrB_Monoid_free_(&m2)) ;
    CHECK (m2 == NULL) ;

    // monoid_crud is allocated but not initialized

    //--------------------------------------------------------------------------
    // Semiring
    //--------------------------------------------------------------------------

    printf ("GrB_Semiring-------------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Semiring_new (NULL, NULL, NULL)) ;
    ERR (GrB_Semiring_new (NULL, GxB_MAX_FP64_MONOID, GrB_PLUS_FP64)) ;

    ERR (GrB_Semiring_new (&semiring, NULL, GrB_PLUS_FP64)) ;
    CHECK (semiring == NULL) ;

    ERR (GrB_Semiring_new (&semiring, GxB_MAX_FP64_MONOID, NULL)) ;
    CHECK (semiring == NULL) ;

    ERR (GrB_Semiring_new (&semiring, NULL, NULL)) ;
    CHECK (semiring == NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_Semiring_new (&semiring, monoid_crud, GrB_PLUS_FP64)) ;
    CHECK (semiring == NULL) ;

    ERR (GrB_Semiring_new (&semiring, GxB_MAX_FP32_MONOID, op2crud)) ;
    CHECK (semiring == NULL) ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR (GrB_Semiring_new (&semiring, GxB_MAX_FP32_MONOID, GrB_PLUS_FP64)) ;
    CHECK (semiring == NULL) ;

    OK (GrB_Semiring_new (&semiring, GxB_MAX_FP64_MONOID, GrB_PLUS_FP64)) ;
    CHECK (semiring != NULL) ;

    o2 = NULL ;
    OK (GxB_Semiring_multiply (&o2, GxB_MAX_PLUS_FP64)) ;
    CHECK (o2 == GrB_PLUS_FP64) ;
    o2 = NULL ;

    m2 = NULL ;
    OK (GxB_Semiring_add (&m2, GxB_MAX_PLUS_FP64)) ;
    CHECK (m2 == GxB_MAX_FP64_MONOID) ;
    m2 = NULL ;

    expected = GrB_NULL_POINTER ;

    ERR (GxB_Semiring_multiply (NULL, NULL)) ;
    ERR (GxB_Semiring_multiply (NULL, GxB_MAX_PLUS_FP64)) ;
    ERR (GxB_Semiring_multiply (&o2, NULL)) ;
    CHECK (o2 == NULL) ;

    ERR (GxB_Semiring_add (NULL, NULL)) ;
    ERR (GxB_Semiring_add (NULL, GxB_MAX_PLUS_FP64)) ;
    ERR (GxB_Semiring_add (&m2, NULL)) ;
    CHECK (m2 == NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    semicrud = semiring ;
    semicrud->magic = 747 ;
    semiring = NULL ;

    m2 = GxB_MAX_INT32_MONOID ;
    ERR (GxB_Semiring_add (&m2, semicrud)) ;
    CHECK (m2 == GxB_MAX_INT32_MONOID) ;
    m2 = NULL ;

    o2 = GxB_LXOR_BOOL ;
    ERR (GxB_Semiring_multiply (&o2, semicrud)) ;
    CHECK (o2 == GxB_LXOR_BOOL) ;
    o2 = NULL ;

    s2 = GxB_PLUS_TIMES_FP64 ;
    OK (GrB_Semiring_free_(&s2)) ;
    CHECK (s2 == GxB_PLUS_TIMES_FP64) ;
    s2 = NULL ;

    OK (GrB_Semiring_new (&s2, GxB_MAX_FP64_MONOID, GrB_PLUS_FP64)) ;
    CHECK (s2 != NULL) ;

    OK (GrB_Semiring_free_(&s2)) ;
    CHECK (s2 == NULL) ;

    // semicrud is allocated but not initialized

    //--------------------------------------------------------------------------
    // user defined ops
    //--------------------------------------------------------------------------

    printf ("User defined (complex) --------------------------------------\n") ;

    #define FREE_DEEP_COPY Complex_finalize ( ) ;
    #define GET_DEEP_COPY ;

    printf ("Complex_finalize:\n") ;
    OK (Complex_finalize ( )) ;
    printf ("Complex_init (true): using builtin complex\n") ;
    OK (Complex_init (true)) ;
    CHECK (Complex == GxB_FC64) ;
    printf ("Complex_finalize again:\n") ;
    OK (Complex_finalize ( )) ;
    printf ("Complex_init (false): user-defined complex\n") ;
    OK (Complex_init (false)) ;
    CHECK (Complex != NULL && Complex != GxB_FC64) ;

    #undef FREE_DEEP_COPY
    #undef GET_DEEP_COPY

    //--------------------------------------------------------------------------
    // basic Scalar methods
    //--------------------------------------------------------------------------

    printf ("GrB_Vector---------------------------------------------------\n") ;

    OK (GrB_Scalar_new (&a_scalar, GrB_INT32)) ;
    CHECK (a_scalar != NULL) ;

    int32_t i_scalar = 33 ;
    OK (GrB_Scalar_setElement_INT32 (a_scalar, 42)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i_scalar, a_scalar)) ;
    CHECK (i_scalar == 42) ;
    GxB_Scalar_fprint_(a_scalar, 3, NULL) ;

    // force a zombie
    GB_MDECL (Ai, , ) ;
    i_scalar = 33 ;
    bool scalar_is_full = GB_IS_FULL (a_scalar) ;
    if (!scalar_is_full)
    {
        Ai = a_scalar->i ;
        GB_IPTR (Ai, a_scalar->i_is_32) ;
        GB_ISET (Ai, 0, i) ;            // Ai [0] = i
        a_scalar->nzombies = 1 ;
    }

    info = GrB_Scalar_extractElement_INT32_(&i_scalar, a_scalar) ;
    CHECK (i_scalar == ((scalar_is_full) ? 42 : 33)) ;
    CHECK (info == (scalar_is_full) ? GrB_SUCCESS : GrB_NO_VALUE) ;

    OK (GrB_Scalar_free_(&a_scalar)) ;

    OK (GrB_Type_new (&T, sizeof (int))) ;

    i_scalar = 207 ;
    expected = GrB_DOMAIN_MISMATCH ;
    OK (GrB_Scalar_new (&a_scalar, T)) ;
    GxB_Scalar_fprint_(a_scalar, 3, NULL) ;
    GxB_Type_fprint_(T, 3, NULL) ;
    ERR1 (a_scalar, GrB_Scalar_setElement_INT32 (a_scalar, 42)) ;
    ERR (GrB_Scalar_extractElement_INT32_(&i_scalar, a_scalar)) ;
    CHECK (i_scalar == 207) ;

    printf ("error expected: %d\n", info) ;

    OK (GrB_Type_free_(&T)) ;
    OK (GrB_Scalar_free_(&a_scalar)) ;

    //--------------------------------------------------------------------------
    // basic Vector methods
    //--------------------------------------------------------------------------

    printf ("GrB_Vector---------------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Vector_new (NULL, NULL, 0)) ;
    ERR (GrB_Vector_new (NULL, GrB_BOOL, 0)) ;
    ERR (GrB_Vector_new (&v, NULL, 0)) ;
    CHECK (v == NULL) ;

    expected = GrB_INVALID_VALUE ;

    ERR (GrB_Vector_new (&v, GrB_BOOL, UINT64_MAX)) ;
    CHECK (v == NULL) ;

    OK (GrB_Vector_new (&vempty, GrB_BOOL, 0)) ;
    CHECK (vempty != NULL) ;

    OK (GrB_Vector_free_(&vempty)) ;
    CHECK (vempty == NULL) ;

    OK (GrB_Vector_free_(&vempty)) ;
    CHECK (vempty == NULL) ;

    OK (GrB_Vector_new (&vempty, GrB_BOOL, 0)) ;
    CHECK (vempty != NULL) ;

    OK (GrB_Vector_new (&vcrud, GrB_BOOL, 1)) ;
    CHECK (vcrud != NULL) ;
    vcrud->magic = 1024 ;

    OK (GrB_Vector_new (&v, GrB_UINT16, 32)) ;
    CHECK (v != NULL) ;

    OK (GrB_Vector_dup (&u, v)) ;
    CHECK (u != NULL) ;

    OK (GrB_Vector_free_(&u)) ;
    CHECK (u == NULL) ;

    OK (GrB_Vector_free_(&u)) ;
    CHECK (u == NULL) ;

    expected = GrB_NULL_POINTER ;

    ERR (GrB_Vector_dup (NULL, NULL)) ;
    ERR (GrB_Vector_dup (NULL, v)) ;
    ERR (GrB_Vector_dup (NULL, vempty)) ;
    ERR (GrB_Vector_clear (NULL)) ;
    ERR (GrB_Vector_dup (NULL, vcrud)) ;
    ERR (GrB_Vector_dup (&u, NULL)) ;
    CHECK (u == NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;
    ERR (GrB_Vector_dup (&u, vcrud)) ;
    CHECK (u == NULL) ;

    OK (GrB_Vector_clear (v)) ;

    expected = GrB_NULL_POINTER ;

    ERR (GrB_Vector_size (NULL, NULL)) ;
    ERR (GrB_Vector_size (NULL, v)) ;
    ERR (GrB_Vector_size (NULL, vcrud)) ;
    ERR (GrB_Vector_size (NULL, vempty)) ;
    ERR (GrB_Vector_size (&n, NULL)) ;
    CHECK (n == 0) ;

    OK (GrB_Vector_size (&n, v)) ;
    CHECK (n == 32) ;

    expected = GrB_UNINITIALIZED_OBJECT ;
    nvals = 7 ;
    ERR (GrB_Vector_dup (&u, vcrud)) ;
    ERR (GrB_Vector_nvals (&nvals, vcrud)) ;
    CHECK (nvals == 7) ;

    OK (GrB_Vector_setElement_INT32  (v, 12, 0)) ;
    OK (GrB_Vector_setElement_BOOL   (v, false, 18)) ;
    OK (GrB_Vector_setElement_INT8   (v, 1, 19)) ;
    OK (GrB_Vector_setElement_UINT8  (v, 2, 20)) ;
    OK (GrB_Vector_setElement_INT16  (v, 3, 21)) ;
    OK (GrB_Vector_setElement_UINT16 (v, 4, 22)) ;
    OK (GrB_Vector_setElement_INT32  (v, 5, 23)) ;
    OK (GrB_Vector_setElement_UINT32 (v, 6, 24)) ;
    OK (GrB_Vector_setElement_INT64  (v, 7, 25)) ;
    OK (GrB_Vector_setElement_UINT64 (v, 8, 26)) ;
    OK (GrB_Vector_setElement_FP32   (v, 9, 27)) ;
    OK (GrB_Vector_setElement_FP64   (v, 10, 28)) ;
    GB_Vector_check (v, "vector 18:28", G3, NULL) ;
    GxB_Vector_fprint (v, "v", G3, ff) ;

    expected = GrB_NULL_POINTER ;

    nvals = 0 ;
    ERR (GrB_Vector_nvals (NULL, NULL)) ;
    ERR (GrB_Vector_nvals (NULL, v)) ;
    ERR (GrB_Vector_nvals (NULL, vempty)) ;
    ERR (GrB_Vector_nvals (&nvals, NULL)) ;
    CHECK (nvals == 0) ;

    OK (GrB_Vector_nvals (&nvals, v)) ;
    OK (GrB_Vector_wait (v, GrB_MATERIALIZE)) ;
    printf ("nvals "GBd"\n", nvals) ;
    GB_Vector_check (v, "vector 18:28", G3, NULL) ;
    GxB_Vector_fprint (v, "v", G3, ff) ;
    CHECK (nvals == 12) ;

    expected = GrB_INVALID_OBJECT ;
    GrB_Vector zz ;
    OK (GrB_Vector_dup (&zz, v)) ;
    OK (GB_Vector_check (zz, "zz ok vector", G3, NULL)) ;
    GB_convert_any_to_hyper ((GrB_Matrix) zz, Werk) ;
    ERR (GB_Vector_check (zz, "zz mangled: vectors cannot be hyper", G3, ff)) ;
    OK (GrB_Vector_free_(&zz)) ;

    OK (GrB_Vector_clear (v)) ;
    OK (GrB_Vector_nvals (&nvals, v)) ;
    CHECK (nvals == 0) ;

    expected = GrB_NULL_POINTER ;

    ERR (GxB_Vector_type (NULL, NULL)) ;

    ERR (GxB_Vector_type (NULL, v)) ;
    ERR (GxB_Vector_type (NULL, vempty)) ;
    ERR (GxB_Vector_type (&T, NULL)) ;
    CHECK (T == NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_Vector_nvals (NULL, vcrud)) ;
    ERR (GxB_Vector_type (NULL, vcrud)) ;
    ERR (GxB_Vector_type (&T, vcrud)) ;
    CHECK (T == NULL) ;

    nvals = 42 ;
    ERR (GrB_Vector_nvals (&nvals, vcrud)) ;
    CHECK (nvals == 42) ;

    OK (GrB_Vector_free_(&v)) ;
    CHECK (v == NULL) ;

    // vcrud is allocated but not initialized

    //--------------------------------------------------------------------------
    // Vector build
    //--------------------------------------------------------------------------

    printf ("GrB_Vector_build---------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Vector_build_BOOL   (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_INT8   (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_UINT8  (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_INT16  (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_UINT16 (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_INT32  (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_UINT32 (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_INT64  (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_UINT64 (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_FP32   (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_FP64   (NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Vector_build_UDT    (NULL, NULL, NULL, 0, NULL)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR1 (vcrud, GrB_Vector_build_BOOL   (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_INT8   (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_UINT8  (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_INT16  (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_UINT16 (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_INT32  (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_UINT32 (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_INT64  (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_UINT64 (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_FP32   (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_FP64   (vcrud, NULL, NULL, 0, NULL)) ;
    ERR1 (vcrud, GrB_Vector_build_UDT    (vcrud, NULL, NULL, 0, NULL)) ;

    expected = GrB_NULL_POINTER ;

    OK (GrB_Vector_new (&v, GrB_FP64, 10)) ;
    ERR1 (v, GrB_Vector_build_FP64 (v, I, NULL, 0, NULL)) ;

    expected = GrB_INVALID_VALUE ;
    o2 = GrB_SECOND_FP64 ;
    ERR1 (v, GrB_Vector_build_FP64 (v, GrB_ALL, X, 0, o2)) ;

    ERR1 (v, GrB_Vector_build_FP64_(v, I, X, GB_NMAX+1, o2)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR1 (v, GrB_Vector_build_FP64_(v, I, X, 5, op2crud)) ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR1 (v, GrB_Vector_build_FP64_(v, I, X, 5, GrB_LE_FP64)) ;

    printf ("build udt:\n") ;
    OK (GrB_Vector_free_(&v)) ;
    GrB_Type user_type = NULL ;
    OK (GxB_Type_new (&user_type, sizeof (gb_user_int16), "gb_user_int16",
        "typedef int16_t gb_user_int16 ;")) ;
    OK (GrB_Vector_new (&v, user_type, 10)) ;
    info = GrB_Vector_build_UDT (v, I, (void *) X, 5, GrB_PLUS_FP64) ;
    GrB_Vector_error (&err, v) ;
    printf ("info %d expected error: %s\n", info, err) ;
    GxB_Vector_fprint (v, "v", G3, NULL) ;
    OK (GrB_Vector_free_(&v)) ;

    OK (GrB_Vector_new (&v, GrB_FP64, 10)) ;

    OK (GxB_BinaryOp_fprint (Complex_plus, "Complex-plus op", G3, f)) ;
    OK (GxB_Type_fprint (Complex, "Complex user type", G3, f)) ;
    OK (GxB_Type_fprint (GxB_FC64, "Complex built-in type", G3, f)) ;
    if (Complex == GxB_FC64)
    {
        OK (GrB_Vector_build_FP64_(v, I, X, 5, Complex_plus)) ;
        GrB_Vector_free_(&v) ;
        OK  (GrB_Vector_new (&v, GrB_FP64, 10)) ;
    }
    else
    {
        ERR1 (v, GrB_Vector_build_FP64_(v, I, X, 5, Complex_plus)) ;
    }

    expected = GrB_OUTPUT_NOT_EMPTY ;

    OK  (GrB_Vector_setElement_INT32 (v, 12, 0)) ;

    ERR1 (v, GrB_Vector_build_FP64_(v, I, X, 5, GrB_PLUS_FP64)) ;

    OK  (GrB_Vector_clear (v)) ;
    OK  (GrB_Vector_build_FP64_(v, I, X, 5, GrB_PLUS_FP64)) ;
    OK  (GrB_Vector_clear (v)) ;
    GB_Vector_check (v, "v clear", G3, NULL) ;
    GxB_Vector_fprint (v, "v", G3, ff) ;

    expected = GrB_INVALID_VALUE ;
    ERR1 (v, GrB_Vector_build_FP64_(v, I, X, GxB_RANGE, GrB_PLUS_FP64)) ;
    GrB_Vector_error_(&err, v) ;
    printf ("%s\n", err) ;

    expected = GrB_INDEX_OUT_OF_BOUNDS ;

    I [0] = 10 ;
    ERR1 (v, GrB_Vector_build_FP64_(v, I, X, 5, GrB_PLUS_FP64)) ;
    GrB_Vector_error_(&err, v) ;
    printf ("expected error, index out of bounds:\n%s\n", err) ;

    GB_Vector_check (v, "v bad", G3, NULL) ;
    GxB_Vector_fprint (v, "v", G3, ff) ;

    expected = GrB_INVALID_OBJECT ;
    ERR (GrB_Vector_nvals (&nvals, v)) ;

    OK (GrB_Vector_free_(&v)) ;
    OK (GrB_Vector_new (&v, GrB_FP64, 10)) ;

    I [0] = -1 ;
    expected = GrB_INDEX_OUT_OF_BOUNDS ;
    ERR1 (v, GrB_Vector_build_FP64_(v, I, X, 5, GrB_PLUS_FP64)) ;

    expected = GrB_INVALID_OBJECT ;
    ERR (GrB_Vector_nvals (&nvals, v)) ;
    I [0] = 0 ;

    OK (GrB_Vector_free_(&v)) ;
    OK (GrB_Vector_new (&v, GrB_FP64, 10)) ;

    // v is a valid 10-by-1 FP64 vector with no entries

    //--------------------------------------------------------------------------
    // Vector setElement
    //--------------------------------------------------------------------------

    printf ("GrB_Vector_setElement----------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Vector_setElement_BOOL   (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_INT8   (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_UINT8  (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_INT16  (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_UINT16 (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_INT32  (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_UINT32 (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_INT64  (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_UINT64 (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_FP32   (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_FP64   (NULL, 0, 0)) ;
    ERR (GrB_Vector_setElement_UDT    (NULL, 0, 0)) ;
    ERR1 (v, GrB_Vector_setElement_UDT    (v, NULL, 0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR1 (vcrud, GrB_Vector_setElement_BOOL   (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_INT8   (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_UINT8  (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_INT16  (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_UINT16 (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_INT32  (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_UINT32 (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_INT64  (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_UINT64 (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_FP32   (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_FP64   (vcrud, 0, 0)) ;
    ERR1 (vcrud, GrB_Vector_setElement_UDT    (vcrud, 0, 0)) ;

    expected = GrB_INVALID_INDEX ;

    ERR1 (v, GrB_Vector_setElement_INT32 (v, 0, -1)) ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR1 (v, GrB_Vector_setElement_UDT (v, (void *) X, 0)) ;

    //--------------------------------------------------------------------------
    // Vector extractElement
    //--------------------------------------------------------------------------

    printf ("GrB_Vector_extractElement------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Vector_extractElement_BOOL   (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_INT8   (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_UINT8  (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_INT16  (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_UINT16 (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_INT32  (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_UINT32 (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_INT64  (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_UINT64 (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_FP32   (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_FP64   (NULL, NULL, 0)) ;
    ERR (GrB_Vector_extractElement_UDT    (NULL, NULL, 0)) ;

    ERR (GrB_Vector_extractElement_BOOL   (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_INT8   (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_UINT8  (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_INT16  (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_UINT16 (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_INT32  (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_UINT32 (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_INT64  (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_UINT64 (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_FP32   (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_FP64   (NULL, v, 0)) ;
    ERR (GrB_Vector_extractElement_UDT    (NULL, v, 0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_Vector_extractElement_BOOL   (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_INT8   (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_UINT8  (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_INT16  (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_UINT16 (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_INT32  (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_UINT32 (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_INT64  (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_UINT64 (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_FP32   (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_FP64   (NULL, vcrud, 0)) ;
    ERR (GrB_Vector_extractElement_UDT    (NULL, vcrud, 0)) ;

    expected = GrB_INVALID_INDEX ;

    ERR (GrB_Vector_extractElement_FP64_(&x_double, v, -1)) ;
    ERR (GrB_Vector_extractElement_FP64_(&x_double, v, 10)) ;

    OK (GrB_Vector_setElement_FP64 (v, 22.8, 2)) ;
    OK (GrB_Vector_setElement_FP64 (v, 44.9, 4)) ;

    expected = GrB_DOMAIN_MISMATCH ;
    ERR (GrB_Vector_extractElement_UDT ((void *) X, v, 2)) ;

    x_double = 404 ;
    OK (GrB_Vector_extractElement_FP64_(&x_double, v, 3)) ;
    CHECK (x_double == 404) ;
    CHECK (info == GrB_NO_VALUE) ;

    OK (GrB_Vector_setElement_FP64 (v, 77.3, 0)) ;

    OK (GrB_Vector_extractElement_FP64_(&x_double, v, 0)) ;

    CHECK (info == GrB_SUCCESS) ;
    CHECK (x_double == 77.3) ;

    OK (GrB_Vector_nvals (&n2, v)) ;
    OK (GrB_Vector_wait_(v, GrB_MATERIALIZE)) ;
    fprintf (f, "vector nvals: %d\n", (int) n2) ;
    CHECK (n2 == 3) ;

    // v is now a valid FP64 vector with 3 entries

    //--------------------------------------------------------------------------
    // Vector extractTuples
    //--------------------------------------------------------------------------

    printf ("GrB_Vector_extractTuples-------------------------------------\n") ;

    expected = GrB_NULL_POINTER ;
    OK (GrB_Vector_nvals (&n2, v)) ;
    nvals = n2 ;

    ERR (GrB_Vector_extractTuples_BOOL   (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_INT8   (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_UINT8  (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_INT16  (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_UINT16 (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_INT32  (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_UINT32 (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_INT64  (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_UINT64 (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_FP32   (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_FP64   (NULL, NULL, NULL, v)) ;
    ERR (GrB_Vector_extractTuples_UDT    (NULL, NULL, NULL, v)) ;

    ERR (GrB_Vector_extractTuples_BOOL   (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_INT8   (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_UINT8  (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_INT16  (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_UINT16 (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_INT32  (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_UINT32 (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_INT64  (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_UINT64 (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_FP32   (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_FP64   (NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Vector_extractTuples_UDT    (NULL, NULL, &nvals, NULL)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_Vector_extractTuples_BOOL   (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_INT8   (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_UINT8  (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_INT16  (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_UINT16 (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_INT32  (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_UINT32 (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_INT64  (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_UINT64 (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_FP32   (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_FP64   (NULL, NULL, &nvals, vcrud)) ;
    ERR (GrB_Vector_extractTuples_UDT    (NULL, NULL, &nvals, vcrud)) ;

    expected = GrB_INSUFFICIENT_SPACE ;

    nvals = n2-1 ;
    ERR (GrB_Vector_extractTuples_FP64_(I2, X2, &nvals, v)) ;
    nvals = n2 ;
    OK  (GrB_Vector_extractTuples_FP64_(I2, X2, &nvals, v)) ;

    for (int k = 0 ; k < n2 ; k++)
    {
        fprintf (f, "%d: v("GBu") = %g\n", k, I2 [k], X2 [k]) ;
    }
    fprintf (f, "\n") ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR (GrB_Vector_extractTuples_UDT_(I2, (void *) X2, &nvals, v)) ;

    GrB_Vector_free_(&v) ;
    CHECK (v == NULL) ;

    //--------------------------------------------------------------------------
    // basic Matrix methods
    //--------------------------------------------------------------------------

    printf ("GrB_Matrix---------------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Matrix_new (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_new (NULL, GrB_BOOL, 0, 0)) ;
    ERR (GrB_Matrix_new (&A, NULL, 0, 0)) ;
    CHECK (A == NULL) ;

    expected = GrB_INVALID_VALUE ;

    ERR (GrB_Matrix_new (&A, GrB_BOOL, UINT64_MAX, 0)) ; CHECK (A == NULL) ;
    ERR (GrB_Matrix_new (&A, GrB_BOOL, 0, UINT64_MAX)) ; CHECK (A == NULL) ;

    OK (GrB_Matrix_new (&Aempty, GrB_BOOL, 0, 0)) ;
    CHECK (Aempty != NULL) ;

    OK (GrB_Matrix_free_(&Aempty)) ;
    CHECK (Aempty == NULL) ;

    OK (GrB_Matrix_free_(&Aempty)) ;
    CHECK (Aempty == NULL) ;

    OK (GrB_Matrix_new (&Aempty, GrB_BOOL, 0, 0)) ;
    CHECK (Aempty != NULL) ;

    OK (GrB_Matrix_new (&Acrud, GrB_BOOL, 1, 1)) ;
    CHECK (Acrud != NULL) ;
    Acrud->magic = 1024 ;

    OK (GrB_Matrix_new (&A, GrB_UINT16, 32, 8)) ;
    CHECK (A != NULL) ;

    OK (GrB_Matrix_dup (&C, A)) ;
    CHECK (C != NULL) ;

    OK (GrB_Matrix_free_(&C)) ;
    CHECK (C == NULL) ;

    OK (GrB_Matrix_free_(&C)) ;
    CHECK (C == NULL) ;

    expected = GrB_NULL_POINTER ;
    ERR (GrB_Matrix_dup (NULL, NULL)) ;
    ERR (GrB_Matrix_dup (NULL, A)) ;
    ERR (GrB_Matrix_dup (NULL, Aempty)) ;
    ERR (GrB_Matrix_clear (NULL)) ;
    ERR (GrB_Matrix_dup (NULL, Acrud)) ;
    ERR (GrB_Matrix_dup (&C, NULL)) ;
    CHECK (C == NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;
    ERR (GrB_Matrix_nrows (&n, Acrud)) ;
    ERR (GrB_Matrix_ncols (&n, Acrud)) ;
    ERR (GrB_Matrix_nvals (&n, Acrud)) ;
    ERR (GrB_Matrix_dup (&C, Acrud)) ;
    CHECK (C == NULL) ;

    OK (GrB_Matrix_clear (A)) ;

    expected = GrB_NULL_POINTER ;
    n = 999 ;
    ERR (GrB_Matrix_nrows (NULL, NULL)) ;
    ERR (GrB_Matrix_nrows (NULL, A)) ;
    ERR (GrB_Matrix_nrows (NULL, Acrud)) ;
    ERR (GrB_Matrix_nrows (NULL, Aempty)) ;
    ERR (GrB_Matrix_nrows (&n, NULL)) ;
    CHECK (n == 999) ;

    ERR (GrB_Matrix_ncols (NULL, NULL)) ;
    ERR (GrB_Matrix_ncols (NULL, A)) ;
    ERR (GrB_Matrix_ncols (NULL, Acrud)) ;
    ERR (GrB_Matrix_ncols (NULL, Aempty)) ;
    ERR (GrB_Matrix_ncols (&n, NULL)) ;
    CHECK (n == 999) ;

    OK (GrB_Matrix_nrows (&n, A)) ;
    CHECK (n == 32) ;

    OK (GrB_Matrix_ncols (&n, A)) ;
    CHECK (n == 8) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_Matrix_dup (&C, Acrud)) ;

    OK (GrB_Matrix_setElement_INT32 (A, 21, 0, 2)) ;
    OK (GrB_Matrix_setElement_INT32 (A, 19, 3, 1)) ;

    expected = GrB_INVALID_INDEX ;
    ERR1 (A, GrB_Matrix_setElement_INT32 (A, 19, 3, 1000)) ;

    expected = GrB_NULL_POINTER ;

    n = 0 ;
    ERR (GrB_Matrix_nvals (NULL, NULL)) ;
    ERR (GrB_Matrix_nvals (NULL, A)) ;
    ERR (GrB_Matrix_nvals (NULL, Aempty)) ;
    ERR (GrB_Matrix_nvals (&nvals, NULL)) ;
    CHECK (n == 0) ;

    OK (GrB_Matrix_nvals (&nvals, A)) ;
    CHECK (nvals == 2) ;

    OK (GrB_Matrix_clear (A)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    CHECK (n == 0) ;

    expected = GrB_NULL_POINTER ;

    ERR (GxB_Matrix_type (NULL, NULL)) ;

    ERR (GxB_Matrix_type (NULL, A)) ;
    ERR (GxB_Matrix_type (NULL, Aempty)) ;
    ERR (GxB_Matrix_type (&T, NULL)) ;
    CHECK (T == NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_Matrix_nvals (NULL, Acrud)) ;
    ERR (GxB_Matrix_type (NULL, Acrud)) ;
    ERR (GxB_Matrix_type (&T, Acrud)) ;
    CHECK (T == NULL) ;

    nvals = 42 ;
    ERR (GrB_Matrix_nvals (&nvals, Acrud)) ;
    CHECK (nvals == 42) ;

    OK (GrB_Matrix_free_(&A)) ;
    CHECK (A == NULL) ;

    // Acrud is allocated but not initialized

    //--------------------------------------------------------------------------
    // Matrix build
    //--------------------------------------------------------------------------

    printf ("GrB_Matrix_build---------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Matrix_build_BOOL   (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_INT8   (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_UINT8  (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_INT16  (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_UINT16 (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_INT32  (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_UINT32 (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_INT64  (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_UINT64 (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_FP32   (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_FP64   (NULL, NULL, NULL, NULL, 0, NULL)) ;
    ERR (GrB_Matrix_build_UDT    (NULL, NULL, NULL, NULL, 0, NULL)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR1 (Acrud, GrB_Matrix_build_BOOL   (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_INT8   (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_UINT8  (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_INT16  (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_UINT16 (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_INT32  (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_UINT32 (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_INT64  (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_UINT64 (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_FP32   (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_FP64   (Acrud, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (Acrud, GrB_Matrix_build_UDT    (Acrud, NULL, NULL, NULL, 0, NULL)) ;

    expected = GrB_NULL_POINTER ;

    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 5)) ;
    ERR1 (A, GrB_Matrix_build_FP64 (A, I,    NULL, NULL, 0, NULL)) ;
    ERR1 (A, GrB_Matrix_build_FP64 (A, NULL, NULL, NULL, 0, NULL)) ;
    ERR1 (A, GrB_Matrix_build_FP64 (A, I,    J,    NULL, 0, NULL)) ;

    expected = GrB_INVALID_VALUE ;

    o2 = GrB_SECOND_FP64 ;
    ERR1 (A, GrB_Matrix_build_FP64_(A, GrB_ALL, J, X, 0, o2)) ;
    ERR1 (A, GrB_Matrix_build_FP64_(A, I, GrB_ALL, X, 0, o2)) ;
    ERR1 (A, GrB_Matrix_build_FP64_(A, I,       J, X, GB_NMAX+1, o2)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR1 (A, GrB_Matrix_build_FP64_(A, I, J, X, 5, op2crud)) ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR1 (A, GrB_Matrix_build_FP64_(A, I, J, X, 5, GrB_LE_FP64)) ;
    if (Complex == GxB_FC64)
    {
        OK (GrB_Matrix_build_FP64_(A, I, J, X, 5, Complex_plus)) ;
    }
    else
    {
        ERR1 (A, GrB_Matrix_build_FP64_(A, I, J, X, 5, Complex_plus)) ;
    }

    GrB_Matrix_free_(&A) ;
    OK (GrB_Matrix_new (&A, user_type, 10, 10)) ;
    ERR1 (A, GrB_Matrix_build_UDT_(A, I, J, (void *) X, 5, GrB_PLUS_FP64)) ;
    GrB_Matrix_error (&err, A) ;
    printf ("info %d expected error: %s\n", info, err) ;
    GrB_Type_free_(&user_type) ;

    GrB_Matrix_free_(&A) ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 5)) ;

    expected = GrB_OUTPUT_NOT_EMPTY ;

    OK  (GrB_Matrix_setElement_INT32 (A, 12, 0, 0)) ;
    ERR1 (A, GrB_Matrix_build_FP64_(A, I, J, X, 5, GrB_PLUS_FP64)) ;
    OK  (GrB_Matrix_clear (A)) ;
    OK  (GrB_Matrix_build_FP64_(A, I, J, X, 5, GrB_PLUS_FP64)) ;
    OK  (GrB_Matrix_clear (A)) ;

    expected = GrB_INDEX_OUT_OF_BOUNDS ;

    I [0] = 10 ;
    ERR1 (A, GrB_Matrix_build_FP64_(A, I, J, X, 5, GrB_PLUS_FP64)) ;

    expected = GrB_INVALID_OBJECT ;
    ERR (GrB_Matrix_nvals (&nvals, A)) ;

    OK (GrB_Matrix_free_(&A)) ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 5)) ;

    I [0] = -1 ;

    expected = GrB_INDEX_OUT_OF_BOUNDS ;
    ERR1 (A, GrB_Matrix_build_FP64_(A, I, J, X, 5, GrB_PLUS_FP64)) ;

    expected = GrB_INVALID_OBJECT ;
    ERR (GrB_Matrix_nvals (&nvals, A)) ;
    I [0] = 0 ;

    OK (GrB_Matrix_free_(&A)) ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 5)) ;

    // A is a valid 10-by-5 FP64 matrix with no entries

    //--------------------------------------------------------------------------
    // Matrix setElement
    //--------------------------------------------------------------------------

    printf ("GrB_Matrix_setElement----------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Matrix_setElement_BOOL   (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_INT8   (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_UINT8  (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_INT16  (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_UINT16 (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_INT32  (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_UINT32 (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_INT64  (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_UINT64 (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_FP32   (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_FP64   (NULL, 0, 0, 0)) ;
    ERR (GrB_Matrix_setElement_UDT    (NULL, 0, 0, 0)) ;
    ERR1 (A, GrB_Matrix_setElement_UDT    (A, NULL, 0, 0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR1 (Acrud, GrB_Matrix_setElement_BOOL   (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_INT8   (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_UINT8  (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_INT16  (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_UINT16 (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_INT32  (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_UINT32 (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_INT64  (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_UINT64 (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_FP32   (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_FP64   (Acrud, 0, 0, 0)) ;
    ERR1 (Acrud, GrB_Matrix_setElement_UDT    (Acrud, 0, 0, 0)) ;

    expected = GrB_INVALID_INDEX ;

    ERR1 (A, GrB_Matrix_setElement_INT32 (A, 0, -1, 0)) ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR1 (A, GrB_Matrix_setElement_UDT (A, (void *) X, 0, 0)) ;

    //--------------------------------------------------------------------------
    // Matrix extractElement
    //--------------------------------------------------------------------------

    printf ("GrB_Matrix_extractElement------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Matrix_extractElement_BOOL   (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT8   (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT8  (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT16  (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT16 (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT32  (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT32 (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT64  (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT64 (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_FP32   (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_FP64   (NULL, NULL, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UDT    (NULL, NULL, 0, 0)) ;

    ERR (GrB_Matrix_extractElement_BOOL   (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT8   (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT8  (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT16  (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT16 (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT32  (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT32 (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT64  (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT64 (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_FP32   (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_FP64   (NULL, A, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UDT    (NULL, A, 0, 0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_Matrix_extractElement_BOOL   (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT8   (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT8  (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT16  (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT16 (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT32  (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT32 (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_INT64  (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UINT64 (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_FP32   (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_FP64   (NULL, Acrud, 0, 0)) ;
    ERR (GrB_Matrix_extractElement_UDT    (NULL, Acrud, 0, 0)) ;

    OK (GrB_Matrix_setElement_FP64 (A, 22.8, 2, 0)) ;
    OK (GrB_Matrix_setElement_FP64 (A, 44.9, 4, 0)) ;

    expected = GrB_INVALID_INDEX ;
    OK (GxB_Matrix_Option_set (A, GxB_FORMAT, GxB_BY_ROW)) ;
    GxB_print (A, 3) ;
    ERR (GrB_Matrix_extractElement_FP64_(&x_double, A, -1, 0)) ;
    ERR (GrB_Matrix_extractElement_FP64_(&x_double, A, 10, 0)) ;
    ERR (GrB_Matrix_extractElement_FP64_(&x_double, A, 0, 911)) ;
    OK (GxB_Matrix_Option_set (A, GxB_FORMAT, GxB_BY_COL)) ;
    GxB_print (A, 3) ;
    ERR (GrB_Matrix_extractElement_FP64_(&x_double, A, -1, 0)) ;
    ERR (GrB_Matrix_extractElement_FP64_(&x_double, A, 10, 0)) ;
    ERR (GrB_Matrix_extractElement_FP64_(&x_double, A, 0, 911)) ;
    OK (GxB_Matrix_Option_set (A, GxB_FORMAT, GxB_BY_ROW)) ;

    expected = GrB_DOMAIN_MISMATCH ;
    ERR (GrB_Matrix_extractElement_UDT ((void *) X, A, 2, 0)) ;

    x_double = 404 ;
    OK (GrB_Matrix_extractElement_FP64_(&x_double, A, 3, 0)) ;
    CHECK (x_double == 404) ;
    CHECK (info == GrB_NO_VALUE) ;

    OK (GrB_Matrix_setElement_FP64 (A, 707.3, 0, 0)) ;

    OK (GrB_Matrix_extractElement_FP64_(&x_double, A, 0, 0)) ;

    CHECK (info == GrB_SUCCESS) ;
    CHECK (x_double == 707.3) ;

    OK (GrB_Matrix_nvals (&n2, A)) ;
    OK (GrB_Matrix_wait_(A, GrB_MATERIALIZE)) ;
    fprintf (f, "nvals: %d\n", (int) n2) ;

    // A is now a valid FP64 matrix with 3 entries

    //--------------------------------------------------------------------------
    // Matrix extractTuples
    //--------------------------------------------------------------------------

    printf ("GrB_Matrix_extractTuples-------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;
    OK (GrB_Matrix_nvals (&n2, A)) ;
    nvals = n2 ;

    ERR (GrB_Matrix_extractTuples_BOOL   (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_INT8   (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_UINT8  (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_INT16  (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_UINT16 (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_INT32  (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_UINT32 (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_INT64  (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_UINT64 (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_FP32   (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_FP64   (NULL, NULL, NULL, NULL, A)) ;
    ERR (GrB_Matrix_extractTuples_UDT    (NULL, NULL, NULL, NULL, A)) ;

    ERR (GrB_Matrix_extractTuples_BOOL   (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_INT8   (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_UINT8  (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_INT16  (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_UINT16 (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_INT32  (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_UINT32 (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_INT64  (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_UINT64 (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_FP32   (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_FP64   (NULL, NULL, NULL, &nvals, NULL)) ;
    ERR (GrB_Matrix_extractTuples_UDT    (NULL, NULL, NULL, &nvals, NULL)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GrB_Matrix_extractTuples_BOOL   (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_INT8   (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_UINT8  (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_INT16  (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_UINT16 (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_INT32  (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_UINT32 (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_INT64  (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_UINT64 (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_FP32   (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_FP64   (NULL, NULL, NULL, &nvals, Acrud)) ;
    ERR (GrB_Matrix_extractTuples_UDT    (NULL, NULL, NULL, &nvals, Acrud)) ;

    expected = GrB_INSUFFICIENT_SPACE ;

    nvals = n2-1 ;
    ERR (GrB_Matrix_extractTuples_FP64_(I2, J2, X2, &nvals, A)) ;
    nvals = n2 ;
    OK  (GrB_Matrix_extractTuples_FP64_(I2, J2, X2, &nvals, A)) ;

    for (int k = 0 ; k < n2 ; k++)
    {
        fprintf (f, "%d: A("GBu","GBu") = %g\n", k, I2 [k], J2 [k], X2 [k]) ;
    }
    fprintf (f, "\n") ;

    expected = GrB_DOMAIN_MISMATCH ;

    nvals = n2 ;
    ERR (GrB_Matrix_extractTuples_UDT_(I2, J2, (void *) X2, &nvals, A)) ;

    GrB_Matrix_free_(&A) ;
    CHECK (A == NULL) ;

    // The following are now allocated; keep them for the rest the tests:
    // Acrud, Tcrud, op1crud, op2crud, monoid_crud, semicrud, Aempty, vempty

    //--------------------------------------------------------------------------
    // Descriptor
    //--------------------------------------------------------------------------

    printf ("GrB_Descriptor-----------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_Descriptor_new (NULL)) ;

    OK (GrB_Descriptor_new (&dcrud)) ;
    CHECK (dcrud != NULL) ;
    dcrud->magic = 22309483 ;

    GrB_Descriptor dnull = NULL ;

    expected = GrB_INVALID_VALUE ;
    ERRD (dnull, GxB_Desc_set (dnull, 0, 0)) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Desc_get (dnull, 0, NULL)) ;

    expected = GrB_INVALID_VALUE ;
    ERRD (dnull, GxB_Desc_set_INT32 (dnull, 0, 0)) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Desc_get_INT32 (dnull, 0, NULL)) ;

//  ERRD (dnull, GxB_Desc_set_FP64 (dnull, 0, 0)) ;
//  ERR (GxB_Desc_get_FP64 (dnull, 0, NULL)) ;

    expected = GrB_INVALID_VALUE ;
    ERRD (dnull, GrB_Descriptor_set (dnull, 0, 0)) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Descriptor_get (NULL, dnull, 0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERRD (dcrud, GxB_Desc_set (dcrud, 0, 0)) ;
    ERR (GxB_Desc_get (dcrud, 0, &dval)) ;

    int32_t dval2 ;
    ERRD (dcrud, GxB_Desc_set_INT32 (dcrud, 0, 0)) ;
    ERR (GxB_Desc_get_INT32 (dcrud, 0, &dval2)) ;

    double dval3 ;

    ERRD (dcrud, GrB_Descriptor_set (dcrud, 0, 0)) ;
    ERR (GxB_Descriptor_get (&dval, dcrud, 0)) ;

    OK (GxB_Desc_get (dnull, 0, &dval)) ;
    CHECK (dval == GxB_DEFAULT) ;

    OK (GxB_Desc_get_INT32 (dnull, 0, &dval2)) ;
    CHECK (dval2 == GxB_DEFAULT) ;

    OK (GxB_Descriptor_get (&dval, dnull, 0)) ;
    CHECK (dval == GxB_DEFAULT) ;

    OK (GrB_Descriptor_new (&desc)) ;

    expected = GrB_INVALID_VALUE ;

    ERR (GxB_Desc_get (desc, -1, &dval)) ;
    ERR (GxB_Desc_get_INT32 (desc, -1, &dval2)) ;
    ERRD (desc, GxB_Desc_set (desc, -1, 0)) ;
    ERRD (desc, GxB_Desc_set_INT32 (desc, -1, 0)) ;
    ERRD (desc, GxB_Desc_set_FP64 (desc, -1, 0)) ;

    ERR (GxB_Descriptor_get (&dval, desc, -1)) ;
    ERRD (desc, GrB_Descriptor_set (desc, -1, 0)) ;

    ERRD (desc, GxB_Desc_set (desc, GrB_OUTP, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;
    ERRD (desc, GxB_Desc_set (desc, GrB_MASK, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;
    ERRD (desc, GxB_Desc_set (desc, GrB_INP0, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;
    ERRD (desc, GxB_Desc_set (desc, GrB_INP1, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;

    ERRD (desc, GxB_Desc_set_INT32 (desc, GrB_OUTP, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;
    ERRD (desc, GxB_Desc_set_INT32 (desc, GrB_MASK, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;
    ERRD (desc, GxB_Desc_set_INT32 (desc, GrB_INP0, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;
    ERRD (desc, GxB_Desc_set_INT32 (desc, GrB_INP1, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;

    ERRD (desc, GrB_Descriptor_set (desc, GxB_AxB_METHOD, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;

    ERRD (desc, GrB_Descriptor_set (desc, GrB_OUTP, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;
    ERRD (desc, GrB_Descriptor_set (desc, GrB_MASK, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;
    ERRD (desc, GrB_Descriptor_set (desc, GrB_INP0, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;
    ERRD (desc, GrB_Descriptor_set (desc, GrB_INP1, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;
    ERRD (desc, GrB_Descriptor_set (desc, GxB_AxB_METHOD, -1)) ;
    GrB_Descriptor_error_(&err, desc) ;
    CHECK (err != NULL) ;
    printf ("%s\n", err) ;

    dval = -1 ;
    OK (GxB_Desc_get (desc, GrB_OUTP, &dval)) ;
    CHECK (dval == GxB_DEFAULT) ;
    dval = -1 ;
    OK (GxB_Desc_get (desc, GrB_MASK, &dval)) ;
    CHECK (dval == GxB_DEFAULT) ;
    dval = -1 ;
    OK (GxB_Desc_get (desc, GrB_INP0, &dval)) ;
    CHECK (dval == GxB_DEFAULT) ;
    dval = -1 ;
    OK (GxB_Desc_get (desc, GrB_INP1, &dval)) ;
    CHECK (dval == GxB_DEFAULT) ;

    dval2 = -1 ;
    OK (GxB_Desc_get_INT32 (desc, GrB_OUTP, &dval2)) ;
    CHECK (dval2 == GxB_DEFAULT) ;
    dval2 = -1 ;
    OK (GxB_Desc_get_INT32 (desc, GrB_MASK, &dval2)) ;
    CHECK (dval2 == GxB_DEFAULT) ;
    dval2 = -1 ;
    OK (GxB_Desc_get_INT32 (desc, GrB_INP0, &dval2)) ;
    CHECK (dval2 == GxB_DEFAULT) ;
    dval2 = -1 ;
    OK (GxB_Desc_get_INT32 (desc, GrB_INP1, &dval2)) ;
    CHECK (dval2 == GxB_DEFAULT) ;

    OK (GxB_Descriptor_get (&dval, desc, GrB_OUTP)) ;
    CHECK (dval == GxB_DEFAULT) ;
    OK (GxB_Descriptor_get (&dval, desc, GrB_MASK)) ;
    CHECK (dval == GxB_DEFAULT) ;
    OK (GxB_Descriptor_get (&dval, desc, GrB_INP0)) ;
    CHECK (dval == GxB_DEFAULT) ;
    OK (GxB_Descriptor_get (&dval, desc, GrB_INP1)) ;
    CHECK (dval == GxB_DEFAULT) ;

    // The following are now allocated; keep them for the rest the tests:
    // Acrud, Tcrud, op1crud, op2crud, monoid_crud, semicrud, Aempty, vempty,
    // desc, dcrud

    #define FREE_DEEP_COPY ;
    #define GET_DEEP_COPY ;

    GrB_Descriptor d7 ;
    METHOD (GrB_Descriptor_new (&d7)) ;
    OK (GB_Descriptor_check (d7, "new descriptor", G3, NULL)) ;
    OK (GxB_Descriptor_fprint (d7, "new descriptor", G3, ff)) ;
    OK (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;

    #undef FREE_DEEP_COPY
    #undef GET_DEEP_COPY

    OK (GxB_Desc_set (d7, GxB_AxB_METHOD, GxB_DEFAULT)) ;
    OK (GB_Descriptor_check (d7, "new descriptor (default)", G3, NULL)) ;
    OK (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;

    OK (GxB_Desc_set (d7, GxB_AxB_METHOD, GxB_AxB_DOT)) ;
    OK (GB_Descriptor_check (d7, "new descriptor (dot)", G3, NULL)) ;
    OK (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;
    OK (GxB_Descriptor_get (&dval, d7, GxB_AxB_METHOD)) ;
    CHECK (dval == GxB_AxB_DOT) ;

    OK (GxB_Desc_set (d7, GxB_AxB_METHOD, GxB_AxB_GUSTAVSON)) ;
    OK (GB_Descriptor_check (d7, "new descriptor (Gustavson)", G3, NULL)) ;
    OK (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;

    expected = GrB_INVALID_VALUE ;
    ERRD (d7, GxB_Desc_set (d7, GxB_AxB_METHOD, 911911)) ;
    OK (GB_Descriptor_check (d7, "new descriptor (still Gustavson)", G3, NULL));
    OK (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;

    OK (GxB_Desc_set_INT32 (d7, GxB_AxB_METHOD, GxB_DEFAULT)) ;
    OK (GB_Descriptor_check (d7, "new descriptor (default)", G3, NULL)) ;
    OK (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;

    OK (GxB_Desc_set_INT32 (d7, GxB_AxB_METHOD, GxB_AxB_DOT)) ;
    OK (GB_Descriptor_check (d7, "new descriptor (dot)", G3, NULL)) ;
    OK (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;
    OK (GxB_Descriptor_get (&dval, d7, GxB_AxB_METHOD)) ;
    CHECK (dval == GxB_AxB_DOT) ;

    OK (GxB_Desc_set_INT32 (d7, GxB_AxB_METHOD, GxB_AxB_GUSTAVSON)) ;
    OK (GB_Descriptor_check (d7, "new descriptor (Gustavson)", G3, NULL)) ;
    OK (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;

    expected = GrB_INVALID_VALUE ;
    ERRD (d7, GxB_Desc_set_INT32 (d7, GxB_AxB_METHOD, 911911)) ;
    OK (GB_Descriptor_check (d7, "new descriptor (still Gustavson)", G3, NULL));
    OK (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;

    expected = GrB_INVALID_OBJECT ;

    d7->axb = 99 ;
    ERR (GB_Descriptor_check (d7, "invalid", G3, NULL)) ;
    ERR (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;
    d7->axb = GxB_DEFAULT ;

    d7->out = 99 ;
    ERR (GB_Descriptor_check (d7, "invalid", G3, NULL)) ;
    ERR (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;
    d7->out = GxB_DEFAULT ;

    d7->out = GxB_AxB_HASH ;
    ERR (GB_Descriptor_check (d7, "invalid", G3, NULL)) ;
    ERR (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;
    d7->out = GxB_DEFAULT ;

    d7->axb = GrB_TRAN ;
    ERR (GB_Descriptor_check (d7, "invalid", G3, NULL)) ;
    ERR (GxB_Descriptor_fprint (d7, "d7", G3, ff)) ;
    d7->out = GxB_DEFAULT ;

    OK (GrB_Descriptor_free_(&d7)) ;
    CHECK (d7 == NULL) ;

    //--------------------------------------------------------------------------
    // create some valid matrices and vectors
    //--------------------------------------------------------------------------

    printf ("create test matrices-----------------------------------------\n") ;

    OK (GB_mx_random_matrix (&A, false, false, 3, 4, 12, 0, false)) ;
    OK (GB_mx_random_matrix (&B, false, false, 4, 2,  6, 0, false)) ;
    OK (GB_mx_random_matrix (&C, false, false, 3, 2,  4, 0, false)) ;
    OK (GB_mx_random_matrix (&E, false, false, 3, 2,  4, 0, false)) ;
    OK (GB_mx_random_matrix (&F, false, false, 3, 2,  4, 0, false)) ;
    OK (GB_mx_random_matrix (&Z, false, false, 3, 2,  8, 0, true)) ; // complex

    OK (GrB_Matrix_wait_(A, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait_(B, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait_(C, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait_(E, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait_(F, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait_(Z, GrB_MATERIALIZE)) ;

    OK (GrB_Vector_new (&v, GrB_FP64, 5)) ;
    OK (GrB_Vector_new (&u, GrB_FP64, 5)) ;
    OK (GrB_Vector_wait_(v, GrB_MATERIALIZE)) ;
    OK (GrB_Vector_wait_(u, GrB_MATERIALIZE)) ;

    printf ("complex vector:\n") ;
    OK (GrB_Vector_new (&z, Complex, 5)) ;

    OK (GrB_Descriptor_new (&dnt)) ;
    OK (GxB_Desc_set (dnt, GrB_INP1, GrB_TRAN)) ;
    OK (GrB_Descriptor_wait_(dnt, GrB_MATERIALIZE)) ;

    OK (GrB_Descriptor_new (&dtn)) ;
    OK (GxB_Desc_set (dtn, GrB_INP0, GrB_TRAN)) ;
    OK (GrB_Descriptor_wait_(dtn, GrB_MATERIALIZE)) ;

    OK (GrB_Descriptor_new (&dtt)) ;
    OK (GxB_Desc_set (dtt, GrB_INP0, GrB_TRAN)) ;
    OK (GxB_Desc_set (dtt, GrB_INP1, GrB_TRAN)) ;
    OK (GrB_Descriptor_wait_(dtt, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // GrB_mxm, mxv, and vxm
    //--------------------------------------------------------------------------

    printf ("GrB_mxm------------------------------------------------------\n") ;
    s2 = GxB_MAX_PLUS_FP32 ;
    o2 = GrB_MAX_FP32 ;

    expected = GrB_NULL_POINTER ;
    ERR1 (Acrud, GrB_mxm (Acrud,  NULL , NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (C, GrB_mxm (C    ,  Acrud, NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (C, GrB_mxm (C    ,  C    , NULL   , NULL    , Acrud, NULL , NULL )) ;

    expected = GrB_UNINITIALIZED_OBJECT ;
    ERR1 (Acrud, GrB_mxm (Acrud, NULL , NULL, s2      , A    , A    , NULL )) ;
    ERR1 (C, GrB_mxm (C    ,  C    , NULL   , NULL    , A    , Acrud, NULL )) ;
    ERR1 (C, GrB_mxm (C    ,  C    , NULL   , NULL    , A    , A    , dcrud)) ;
    ERR1 (C, GrB_mxm (C    ,  C    , op2crud, NULL    , A    , A    , NULL )) ;
    ERR1 (C, GrB_mxm (C    ,  C    , o2     , semicrud, A    , A    , NULL )) ;

    expected = GrB_NULL_POINTER ;
    ERR1 (vcrud, GrB_vxm (vcrud,  NULL , NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (v, GrB_vxm (v    ,  vcrud, NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (v, GrB_vxm (v    ,  v    , NULL   , NULL    , vcrud, NULL , NULL )) ;

    expected = GrB_UNINITIALIZED_OBJECT ;
    ERR1 (vcrud, GrB_vxm (vcrud,  NULL, NULL, s2      , v    , A    , NULL )) ;
    ERR1 (v, GrB_vxm (v    ,  v    , NULL   , NULL    , v    , Acrud, NULL )) ;
    ERR1 (v, GrB_vxm (v    ,  v    , NULL   , NULL    , v    , A    , dcrud)) ;
    ERR1 (v, GrB_vxm (v    ,  v    , op2crud, NULL    , v    , A    , NULL )) ;
    ERR1 (v, GrB_vxm (v    ,  v    , o2     , semicrud, v    , A    , NULL )) ;

    expected = GrB_NULL_POINTER ;
    ERR1 (vcrud, GrB_mxv (vcrud,  NULL , NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (v, GrB_mxv (v    ,  vcrud, NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (v, GrB_mxv (v    ,  v    , NULL   , NULL    , Acrud, NULL , NULL )) ;

    expected = GrB_UNINITIALIZED_OBJECT ;
    ERR1 (vcrud, GrB_mxv (vcrud, NULL , NULL, s2      , A    , v    , NULL )) ;
    ERR1 (v, GrB_mxv (v    ,  v    , NULL   , NULL    , A    , vcrud, NULL )) ;
    ERR1 (v, GrB_mxv (v    ,  v    , NULL   , NULL    , A    , v    , dcrud)) ;
    ERR1 (v, GrB_mxv (v    ,  v    , op2crud, NULL    , A    , v    , NULL )) ;
    ERR1 (v, GrB_mxv (v    ,  v    , o2     , semicrud, A    , v    , NULL )) ;

    expected = GrB_NULL_POINTER ;

    ERR (GrB_mxm (NULL ,  NULL , NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (C, GrB_mxm (C    ,  NULL , NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (C, GrB_mxm (C    ,  NULL , NULL   , NULL    , A    , NULL , NULL )) ;
    ERR1 (C, GrB_mxm (C    ,  NULL , o2     , NULL    , A    , A    , NULL )) ;

    ERR (GrB_vxm (NULL ,  NULL , NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (v, GrB_vxm (v    ,  v    , NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (v, GrB_vxm (v    ,  v    , NULL   , NULL    , v    , NULL , NULL )) ;
    ERR1 (v, GrB_vxm (v    ,  v    , o2     , NULL    , v    , A    , NULL )) ;

    ERR (GrB_mxv (NULL ,  NULL , NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (v, GrB_mxv (v    ,  v    , NULL   , NULL    , NULL , NULL , NULL )) ;
    ERR1 (v, GrB_mxv (v    ,  v    , NULL   , NULL    , A    , NULL , NULL )) ;
    ERR1 (v, GrB_mxv (v    ,  v    , o2     , NULL    , A    , v    , NULL )) ;

    expected = GrB_DIMENSION_MISMATCH ;

    ERR1 (C, GrB_mxm (C   , NULL, NULL, s2  , B   , A   , NULL)) ;
    ERR1 (C, GrB_mxm (C   , A   , NULL, s2  , A   , B   , NULL)) ;

    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;
    ERR1 (C, GrB_mxm (C, NULL, NULL, s2, Z, B, NULL)) ;
    ERR1 (C, GrB_mxm (C, NULL, NULL, s2, B, Z, NULL)) ;
    if (Complex == GxB_FC64)
    {
        OK  (GrB_mxm (C, NULL, NULL, Complex_plus_times, A, B, NULL)) ;
        OK  (GrB_mxm (Z, NULL, NULL, s2, A, B, NULL)) ;
        OK  (GrB_mxm (C, Z   , NULL, s2, A, B, NULL)) ;
    }
    else
    {
        ERR1 (C, GrB_mxm (C, NULL, NULL, Complex_plus_times, A, B, NULL)) ;
        ERR1 (Z, GrB_mxm (Z, NULL, NULL, s2, A, B, NULL)) ;
        ERR1 (C, GrB_mxm (C, Z   , NULL, s2, A, B, NULL)) ;
    }

    GrB_Matrix_error_(&err, C) ;
    printf ("last error was [%s]\n", err) ;
    OK (GrB_mxm (C, NULL, o2 , s2, A, B, NULL)) ;

    // The following are now allocated; keep them for the rest the tests:
    // Acrud, Tcrud, op1crud, op2crud, monoid_crud, semicrud, Aempty, vempty,
    // desc, dcrud, A, B, C, E, F, Z, v, u

    //--------------------------------------------------------------------------
    // GrB_mxm dot product
    //--------------------------------------------------------------------------

    uint64_t huge = GB_NMAX ;
    GrB_Matrix HugeRow ;
    OK (GrB_Matrix_new (&HugeRow, GrB_FP64, 1, huge)) ;
    GB_Matrix_check (HugeRow, "huge row", G3, NULL) ;
    GxB_Matrix_fprint (HugeRow, "HugeRow", G3, ff) ;

    struct GB_Matrix_opaque HugeMatrix_header ;
    GrB_Matrix HugeMatrix = GB_clear_matrix_header (&HugeMatrix_header) ;

    OK (GB_AxB_dot2 (HugeMatrix, false, NULL, NULL, false, false,
        false, HugeRow, HugeRow, GxB_PLUS_TIMES_FP64, false, Werk)) ;

    GxB_Matrix_fprint (HugeMatrix, "HugeMatrix", G3, ff) ;
    GrB_Matrix_free_(&HugeMatrix) ;
    GrB_Matrix_free_(&HugeRow) ;

    //--------------------------------------------------------------------------
    // eWiseMult and eWiseAdd
    //--------------------------------------------------------------------------

    printf ("GrB_eWise ---------------------------------------------------\n") ;

    m2 = GxB_MIN_FP64_MONOID ;
    s2 = GxB_PLUS_ISEQ_FP32 ;

    expected = GrB_NULL_POINTER ;

    ERR1 (v0, GrB_Vector_eWiseMult_Semiring_(v0, NULL, NULL, s0 , v0, v0, d0)) ;  // vector semiring
    ERR1 (v0, GrB_Vector_eWiseMult_Semiring_(v0, NULL, NULL, s2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Semiring_(v , NULL, NULL, s2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Semiring_(v , NULL, NULL, s2 , v , v0, d0)) ;

    ERR1 (v0, GrB_Vector_eWiseMult_Monoid_(v0, NULL, NULL, m0 , v0, v0, d0)) ;  // vector monoid
    ERR1 (v0, GrB_Vector_eWiseMult_Monoid_(v0, NULL, NULL, m2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Monoid_(v , NULL, NULL, m2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Monoid_(v , NULL, NULL, m2 , v , v0, d0)) ;

    ERR1 (v0, GrB_Vector_eWiseMult_BinaryOp_(v0, NULL, NULL, op0, v0, v0, d0)) ;  // vector op
    ERR1 (v0, GrB_Vector_eWiseMult_BinaryOp_(v0, NULL, NULL, o2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_BinaryOp_(v , NULL, NULL, o2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_BinaryOp_(v , NULL, NULL, o2 , v , v0, d0)) ;

    ERR1 (A0, GrB_Matrix_eWiseMult_Semiring_(A0, NULL, NULL, s0 , A0, A0, d0)) ;  // matrix semiring
    ERR1 (A0, GrB_Matrix_eWiseMult_Semiring_(A0, NULL, NULL, s2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Semiring_(A , NULL, NULL, s2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Semiring_(A , NULL, NULL, s2 , A , A0, d0)) ;

    ERR1 (A0, GrB_Matrix_eWiseMult_Monoid_(A0, NULL, NULL, m0 , A0, A0, d0)) ;  // matrix monoid
    ERR1 (A0, GrB_Matrix_eWiseMult_Monoid_(A0, NULL, NULL, m2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Monoid_(A , NULL, NULL, m2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Monoid_(A , NULL, NULL, m2 , A , A0, d0)) ;

    ERR1 (A0, GrB_Matrix_eWiseMult_BinaryOp_(A0, NULL, NULL, op0, A0, A0, d0)) ;  // matrix op
    ERR1 (A0, GrB_Matrix_eWiseMult_BinaryOp_(A0, NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_BinaryOp_(A , NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_BinaryOp_(A , NULL, NULL, o2 , A , A0, d0)) ;

    ERR1 (v0, GrB_Vector_eWiseAdd_Semiring_(v0, NULL, NULL, s0 , v0, v0, d0)) ;  // vector semiring
    ERR1 (v0, GrB_Vector_eWiseAdd_Semiring_(v0, NULL, NULL, s2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Semiring_(v , NULL, NULL, s2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Semiring_(v , NULL, NULL, s2 , v , v0, d0)) ;

    ERR1 (v0, GrB_Vector_eWiseAdd_Monoid_(v0, NULL, NULL, m0 , v0, v0, d0)) ;  // vector monoid
    ERR1 (v0, GrB_Vector_eWiseAdd_Monoid_(v0, NULL, NULL, m2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Monoid_(v , NULL, NULL, m2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Monoid_(v , NULL, NULL, m2 , v , v0, d0)) ;

    ERR1 (v0, GrB_Vector_eWiseAdd_BinaryOp_(v0, NULL, NULL, op0, v0, v0, d0)) ;  // vector op
    ERR1 (v0, GrB_Vector_eWiseAdd_BinaryOp_(v0, NULL, NULL, o2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_BinaryOp_(v , NULL, NULL, o2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_BinaryOp_(v , NULL, NULL, o2 , v , v0, d0)) ;

    ERR1 (A0, GrB_Matrix_eWiseAdd_Semiring_(A0, NULL, NULL, s0 , A0, A0, d0)) ;  // matrix semiring
    ERR1 (A0, GrB_Matrix_eWiseAdd_Semiring_(A0, NULL, NULL, s2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Semiring_(A , NULL, NULL, s2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Semiring_(A , NULL, NULL, s2 , A , A0, d0)) ;

    ERR1 (A0, GrB_Matrix_eWiseAdd_Monoid_(A0, NULL, NULL, m0 , A0, A0, d0)) ;  // matrix monoid
    ERR1 (A0, GrB_Matrix_eWiseAdd_Monoid_(A0, NULL, NULL, m2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Monoid_(A , NULL, NULL, m2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Monoid_(A , NULL, NULL, m2 , A , A0, d0)) ;

    ERR1 (A0, GrB_Matrix_eWiseAdd_BinaryOp_(A0, NULL, NULL, op0, A0, A0, d0)) ;  // matrix op
    ERR1 (A0, GrB_Matrix_eWiseAdd_BinaryOp_(A0, NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, NULL, o2 , A , A0, d0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    m0 = monoid_crud ;
    s0 = semicrud ;
    v0 = vcrud ;
    A0 = Acrud ;
    op0 = op2crud ;

    ERR1 (v0, GrB_Vector_eWiseMult_Semiring_(v0, NULL, NULL, s0 , v0, v0, d0)) ;  // vector semiring
    ERR1 (v0, GrB_Vector_eWiseMult_Semiring_(v0, NULL, NULL, s2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Semiring_(v , NULL, NULL, s2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Semiring_(v , NULL, NULL, s2 , v , v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Semiring_(v , v0  , NULL, s2 , v , v , d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Semiring_(v , NULL, op0 , s2 , v , v , NULL)) ;

    ERR1 (v0, GrB_Vector_eWiseMult_Monoid_(v0, NULL, NULL, m0 , v0, v0, d0)) ;  // vector monoid
    ERR1 (v0, GrB_Vector_eWiseMult_Monoid_(v0, NULL, NULL, m2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Monoid_(v , NULL, NULL, m2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Monoid_(v , NULL, NULL, m2 , v , v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Monoid_(v , v0  , NULL, m2 , v , v , d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_Monoid_(v , NULL, op0 , m2 , v , v , NULL)) ;

    ERR1 (v0, GrB_Vector_eWiseMult_BinaryOp_(v0, NULL, NULL, op0, v0, v0, d0)) ;  // vector op
    ERR1 (v0, GrB_Vector_eWiseMult_BinaryOp_(v0, NULL, NULL, o2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_BinaryOp_(v , NULL, NULL, o2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_BinaryOp_(v , NULL, NULL, o2 , v , v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_BinaryOp_(v , v0  , NULL, o2 , v , v , d0)) ;
    ERR1 (v,  GrB_Vector_eWiseMult_BinaryOp_(v , NULL, op0 , o2 , v , v , NULL)) ;

    ERR1 (A0, GrB_Matrix_eWiseMult_Semiring_(A0, NULL, NULL, s0 , A0, A0, d0)) ;  // matrix semiring
    ERR1 (A0, GrB_Matrix_eWiseMult_Semiring_(A0, NULL, NULL, s2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Semiring_(A , NULL, NULL, s2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Semiring_(A , NULL, NULL, s2 , A , A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Semiring_(A , A0  , NULL, s2 , A , A , d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Semiring_(A , NULL, op0 , s2 , A , A , NULL)) ;

    ERR1 (A0, GrB_Matrix_eWiseMult_Monoid_(A0, NULL, NULL, m0 , A0, A0, d0)) ;  // matrix monoid
    ERR1 (A0, GrB_Matrix_eWiseMult_Monoid_(A0, NULL, NULL, m2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Monoid_(A , NULL, NULL, m2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Monoid_(A , NULL, NULL, m2 , A , A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Monoid_(A , A0  , NULL, m2 , A , A , d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_Monoid_(A , NULL, op0 , m2 , A , A , NULL)) ;

    ERR1 (A0, GrB_Matrix_eWiseMult_BinaryOp_(A0, NULL, NULL, op0, A0, A0, d0)) ;  // matrix op
    ERR1 (A0, GrB_Matrix_eWiseMult_BinaryOp_(A0, NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_BinaryOp_(A , NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_BinaryOp_(A , NULL, NULL, o2 , A , A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_BinaryOp_(A , A0  , NULL, o2 , A , A , d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseMult_BinaryOp_(A , NULL, op0 , o2 , A , A , NULL)) ;

    ERR1 (v0, GrB_Vector_eWiseAdd_Semiring_(v0, NULL, NULL, s0 , v0, v0, d0)) ;  // vector semiring
    ERR1 (v0, GrB_Vector_eWiseAdd_Semiring_(v0, NULL, NULL, s2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Semiring_(v , NULL, NULL, s2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Semiring_(v , NULL, NULL, s2 , v , v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Semiring_(v , v0  , NULL, s2 , v , v , d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Semiring_(v , NULL, op0 , s2 , v , v , NULL)) ;

    ERR1 (v0, GrB_Vector_eWiseAdd_Monoid_(v0, NULL, NULL, m0 , v0, v0, d0)) ;  // vector monoid
    ERR1 (v0, GrB_Vector_eWiseAdd_Monoid_(v0, NULL, NULL, m2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Monoid_(v , NULL, NULL, m2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Monoid_(v , NULL, NULL, m2 , v , v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Monoid_(v , v0  , NULL, m2 , v , v , d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_Monoid_(v , NULL, op0 , m2 , v , v , NULL)) ;

    ERR1 (v0, GrB_Vector_eWiseAdd_BinaryOp_(v0, NULL, NULL, op0, v0, v0, d0)) ;  // vector op
    ERR1 (v0, GrB_Vector_eWiseAdd_BinaryOp_(v0, NULL, NULL, o2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_BinaryOp_(v , NULL, NULL, o2 , v0, v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_BinaryOp_(v , NULL, NULL, o2 , v , v0, d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_BinaryOp_(v , v0  , NULL, o2 , v , v , d0)) ;
    ERR1 (v,  GrB_Vector_eWiseAdd_BinaryOp_(v , NULL, op0 , o2 , v , v , NULL)) ;

    ERR1 (A0, GrB_Matrix_eWiseAdd_Semiring_(A0, NULL, NULL, s0 , A0, A0, d0)) ;  // matrix semiring
    ERR1 (A0, GrB_Matrix_eWiseAdd_Semiring_(A0, NULL, NULL, s2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Semiring_(A , NULL, NULL, s2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Semiring_(A , NULL, NULL, s2 , A , A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Semiring_(A , A0  , NULL, s2 , A , A , d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Semiring_(A , NULL, op0 , s2 , A , A , NULL)) ;

    ERR1 (A0, GrB_Matrix_eWiseAdd_Monoid_(A0, NULL, NULL, m0 , A0, A0, d0)) ;  // matrix monoid
    ERR1 (A0, GrB_Matrix_eWiseAdd_Monoid_(A0, NULL, NULL, m2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Monoid_(A , NULL, NULL, m2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Monoid_(A , NULL, NULL, m2 , A , A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Monoid_(A , A0  , NULL, m2 , A , A , d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_Monoid_(A , NULL, op0 , m2 , A , A , NULL)) ;

    ERR1 (A0, GrB_Matrix_eWiseAdd_BinaryOp_(A0, NULL, NULL, op0, A0, A0, d0)) ;  // matrix op
    ERR1 (A0, GrB_Matrix_eWiseAdd_BinaryOp_(A0, NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, NULL, o2 , A , A0, d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , A0  , NULL, o2 , A , A , d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, op0 , o2 , A , A , NULL)) ;

    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, NULL, o2 , Z , A , d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, NULL, o2 , A , Z , d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, NULL, Complex_plus, Z , A , d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, NULL, Complex_plus, A , Z , d0)) ;
    ERR1 (A,  GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, NULL, Complex_plus, Z , Z , d0)) ;
    ERR1 (Z,  GrB_Matrix_eWiseAdd_BinaryOp_(Z , NULL, NULL, Complex_complex, A , A , d0)) ;

    if (Complex == GxB_FC64)
    {
        OK  (GrB_Matrix_eWiseAdd_BinaryOp_(Z , Z   , NULL, Complex_plus, Z , Z , d0)) ;
    }
    else
    {
        ERR1 (Z,  GrB_Matrix_eWiseAdd_BinaryOp_(Z , Z   , NULL, Complex_plus, Z , Z , d0)) ;
    }

    OK (GrB_BinaryOp_new (&op3, (GxB_binary_function) f3, Complex, Complex, GrB_FP64)) ;

    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;
    ERR1 (Z, GrB_Matrix_eWiseAdd_BinaryOp_(Z , NULL, NULL, op3, Z , A , d0)) ;
    ERR1 (Z, GrB_Matrix_eWiseAdd_BinaryOp_(Z , NULL, op3 , o2 , A , A , d0)) ;

    if (Complex == GxB_FC64)
    {
        OK  (GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, op3 , o2 , A , A , d0)) ;
        OK  (GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, Complex_complex, o2 , A , A , d0)) ;
    }
    else
    {
        ERR1 (A, GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, op3 , o2 , A , A , d0)) ;
        ERR1 (A, GrB_Matrix_eWiseAdd_BinaryOp_(A , NULL, Complex_complex, o2 , A , A , d0)) ;
    }

    expected = GrB_DIMENSION_MISMATCH ;

    ERR1 (C, GrB_Matrix_eWiseAdd_BinaryOp_(C , NULL, NULL, o2 , A , B , d0)) ;
    ERR1 (C, GrB_Matrix_eWiseAdd_BinaryOp_(C , NULL, NULL, o2 , A , B , dtn)) ;
    ERR1 (C, GrB_Matrix_eWiseAdd_BinaryOp_(C , NULL, NULL, o2 , A , B , dnt)) ;
    ERR1 (C, GrB_Matrix_eWiseAdd_BinaryOp_(C , NULL, NULL, o2 , A , B , dtt)) ;

    // The following are now allocated; keep them for the rest the tests:
    // Acrud, Tcrud, op1crud, op2crud, monoid_crud, semicrud, Aempty, vempty,
    // desc, dcrud, A, B, C, E, F, Z, v, u, dnt, dtn, dtt

    //--------------------------------------------------------------------------
    // GrB_kronecker
    //--------------------------------------------------------------------------

    printf ("GrB_kronecker -----------------------------------------------\n") ;

    m2 = GxB_MIN_FP64_MONOID ;
    s2 = GxB_PLUS_ISEQ_FP32 ;

    m0 = NULL ;
    s0 = NULL ;
    v0 = NULL ;
    A0 = NULL ;
    op0 = NULL ;

    expected = GrB_NULL_POINTER ;

    info = (GrB_Matrix_kronecker_BinaryOp_(A0, NULL, NULL, op0, A0, A0, d0)) ;  // matrix op
    ERR1 (A0, GrB_Matrix_kronecker_BinaryOp_(A0, NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_(A , NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_(A , NULL, NULL, o2 , A , A0, d0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    m0 = monoid_crud ;
    s0 = semicrud ;
    v0 = vcrud ;
    A0 = Acrud ;
    op0 = op2crud ;

    ERR1 (A0, GrB_Matrix_kronecker_BinaryOp_(A0, NULL, NULL, op0, A0, A0, d0)) ;  // matrix op
    ERR1 (A0, GrB_Matrix_kronecker_BinaryOp_(A0, NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_(A , NULL, NULL, o2 , A0, A0, d0)) ;
    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_(A , NULL, NULL, o2 , A , A0, d0)) ;
    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_(A , A0  , NULL, o2 , A , A , d0)) ;
    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_(A , NULL, op0 , o2 , A , A , NULL)) ;

    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;

    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_  (A , NULL, NULL, o2 , Z , A , d0)) ;
    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_  (A , NULL, NULL, o2 , A , Z , d0)) ;
    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_  (A , NULL, NULL, Complex_plus, Z , A , d0)) ;
    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_  (A , NULL, NULL, Complex_plus, A , Z , d0)) ;
    ERR1 (A, GrB_Matrix_kronecker_BinaryOp_  (A , NULL, NULL, Complex_plus, Z , Z , d0)) ;
    ERR1 (Z, GrB_Matrix_kronecker_BinaryOp_  (Z , Z   , NULL, Complex_plus, Z , Z , d0)) ;

    expected = GrB_DIMENSION_MISMATCH ;

    ERR1 (C, GrB_Matrix_kronecker_BinaryOp_  (C , NULL, NULL, o2 , A , B , d0)) ;
    ERR1 (C, GrB_Matrix_kronecker_BinaryOp_  (C , NULL, NULL, o2 , A , B , dtn)) ;
    ERR1 (C, GrB_Matrix_kronecker_BinaryOp_  (C , NULL, NULL, o2 , A , B , dnt)) ;
    ERR1 (C, GrB_Matrix_kronecker_BinaryOp_  (C , NULL, NULL, o2 , A , B , dtt)) ;

    //--------------------------------------------------------------------------
    // extract
    //--------------------------------------------------------------------------

    printf ("GrB_extract -------------------------------------------------\n") ;

    expected = GrB_NULL_POINTER ;

    m0 = NULL ;
    s0 = NULL ;
    v0 = NULL ;
    A0 = NULL ;
    op0 = NULL ;

    ERR1 (v0, GrB_Vector_extract_(v0, NULL, NULL, v0, I0, 0,    d0)) ;     // vector extract
    ERR1 (v,  GrB_Vector_extract_(v , NULL, NULL, v0, I0, 0,    d0)) ;
    ERR1 (v,  GrB_Vector_extract_(v , NULL, NULL, u , I0, 0,    d0)) ;

    ERR1 (v0, GrB_Col_extract_(v0, NULL, NULL, A0, I0, 0, 0, d0)) ;     // column extract
    ERR1 (v,  GrB_Col_extract_(v , NULL, NULL, A0, I0, 0, 0, d0)) ;
    ERR1 (v,  GrB_Col_extract_(v , NULL, NULL, A , I0, 0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_extract_(A0, NULL, NULL, A0, I0, 0, J0, 0, d0)) ; // matrix extract
    ERR1 (A,  GrB_Matrix_extract_(A , NULL, NULL, A0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_extract_(A , NULL, NULL, A , I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_extract_(A , NULL, NULL, A , I , 0, J0, 0, d0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    v0 = vcrud ;
    A0 = Acrud ;
    d0 = dcrud ;
    op0 = op2crud ;

    ERR1 (v0, GrB_Vector_extract_(v0, NULL, NULL, v0, I0, 0,    d0)) ;     // vector extract
    ERR1 (v,  GrB_Vector_extract_(v , v0  , NULL, v0, I0, 0,    d0)) ;
    ERR1 (v,  GrB_Vector_extract_(v , v   , NULL, v0, I0, 0,    d0)) ;
    ERR1 (v,  GrB_Vector_extract_(v , v   , NULL, v , I , 1,    d0)) ;
    ERR1 (v,  GrB_Vector_extract_(v , v   , op0 , v , I , 1,    NULL)) ;

    ERR1 (v0, GrB_Col_extract_(v0, NULL, NULL, A0, I0, 0, 0, d0)) ;     // column extract
    ERR1 (v,  GrB_Col_extract_(v , v0  , NULL, A0, I0, 0, 0, d0)) ;
    ERR1 (v,  GrB_Col_extract_(v , v   , NULL, A0, I0, 0, 0, d0)) ;
    ERR1 (v,  GrB_Col_extract_(v , v   , NULL, A , I , 1, 0, d0)) ;
    ERR1 (v,  GrB_Col_extract_(v , v   , op0 , A , I , 1, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_extract_(A0, NULL, NULL, A0, I0, 0, J0, 0, d0)) ; // matrix extract
    ERR1 (A,  GrB_Matrix_extract_(A , A0  , NULL, A0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_extract_(A , A   , NULL, A0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_extract_(A , A   , NULL, A0, I , 1, J , 1, d0)) ;
    ERR1 (A,  GrB_Matrix_extract_(A , A   , op0 , A , I , 1, J , 1, NULL)) ;

    v0 = NULL ;
    A0 = NULL ;
    d0 = NULL ;
    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;

    ERR1 (v,  GrB_Vector_extract_(v, z, NULL, u, I, 0, d0)) ;              // vector extract
    ERR1 (v,  GrB_Vector_extract_(v, NULL, Complex_plus, u, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_extract_(v, NULL, Complex_plus, z, I, 0, d0)) ;
    ERR1 (z,  GrB_Vector_extract_(z, NULL, o2 , u, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_extract_(v, NULL, o2 , z, I, 0, d0)) ;

    ERR1 (v,  GrB_Col_extract_(v, z, NULL, A, I, 0, 0, d0)) ;           // column extract
    ERR1 (v,  GrB_Col_extract_(v, NULL, Complex_plus, A, I, 0, 0, d0)) ;
    ERR1 (v,  GrB_Col_extract_(v, NULL, Complex_plus, Z, I, 0, 0, d0)) ;
    ERR1 (z,  GrB_Col_extract_(z, NULL, o2 , A, I, 0, 0, d0)) ;
    ERR1 (v,  GrB_Col_extract_(v, NULL, o2 , Z, I, 0, 0, d0)) ;

    ERR1 (A,  GrB_Matrix_extract_(A, Z, NULL, A, I, 0, J, 0, d0)) ;        // matrix extract
    ERR1 (A,  GrB_Matrix_extract_(A, NULL, Complex_plus, A, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_extract_(A, NULL, Complex_plus, Z, I, 0, J, 0, d0)) ;
    ERR1 (Z,  GrB_Matrix_extract_(Z, NULL, o2 , A, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_extract_(A, NULL, o2 , Z, I, 0, J, 0, d0)) ;

    expected = GrB_DIMENSION_MISMATCH ;

    ERR1 (A,  GrB_Matrix_extract_(A, NULL, NULL, A, I, 1, J, 2, d0)) ;
    ERR1 (A,  GrB_Matrix_extract_(A, NULL, NULL, A, I, 1, J, 2, dtn)) ;

    expected = GrB_INVALID_INDEX ;

    OK (GrB_Vector_new (&h, GrB_FP64, 1)) ;

    OK  (GrB_Col_extract_(h, NULL, NULL, A, I, 1,   0, d0)) ;  // column extract

    ERR1 (h, GrB_Col_extract_(h, NULL, NULL, A, I, 1, 911, d0)) ;  // column extract
    GrB_Vector II = NULL ;
    ERR1 (h, GxB_Col_extract_Vector_(h, NULL, NULL, A, II, 911, d0)) ;  // column extract

    expected = GrB_INDEX_OUT_OF_BOUNDS ;

    OK (GrB_Matrix_new (&H, GrB_FP64, 1, 1)) ;

    I [0] = 911 ;
    ERR1 (H, GrB_Matrix_extract (H, NULL, NULL, A, I, 1, J, 1, d0)) ;
    I [0] = 0 ;

    J [0] = 911 ;
    ERR1 (H, GrB_Matrix_extract (H, NULL, NULL, A, I, 1, J, 1, d0)) ;
    J [0] = 4 ;

    //--------------------------------------------------------------------------
    // subassign
    //--------------------------------------------------------------------------

    printf ("GxB_subassign -----------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    // GxB_Vector_subassign   (w,mask,acc,u,I,ni,d)
    // GxB_Matrix_subassign   (C,Mask,acc,A,I,ni,J,nj,d)
    // GxB_Col_subassign      (C,mask,acc,u,I,ni,j,d)
    // GxB_Row_subassign      (C,mask,acc,u,i,J,nj,d)
    // GxB_Vector_subassign_T (w,mask,acc,x,I,ni,d)
    // GxB_Matrix_subassign_T (C,Mask,acc,x,I,ni,J,nj,d)

    ERR1 (v0, GxB_Vector_subassign_(v0, NULL, NULL, v0, I0, 0, d0)) ;       // vector assign
    ERR1 (v,  GxB_Vector_subassign_(v , NULL, NULL, v0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_(v , NULL, NULL, v , I0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_(A0, NULL, NULL, A0, I0, 0, J0, 0, d0)) ;// matrix assign
    ERR1 (A,  GxB_Matrix_subassign_(A , NULL, NULL, A0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_(A , NULL, NULL, A , I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_(A , NULL, NULL, A , I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Col_subassign_(A0, NULL, NULL, v0, I0, 0,  0, d0)) ;   // column assign
    ERR1 (A,  GxB_Col_subassign_(A , NULL, NULL, v0, I0, 0,  0, d0)) ;
    ERR1 (A,  GxB_Col_subassign_(A , NULL, NULL, v , I0, 0,  0, d0)) ;

    ERR1 (A0, GxB_Row_subassign_(A0, NULL, NULL, v0,  0, J0, 0, d0)) ;   // row assign
    ERR1 (A,  GxB_Row_subassign_(A , NULL, NULL, v0,  0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Row_subassign_(A , NULL, NULL, v ,  0, J0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_FP64_(v0, NULL, NULL,  x, I0, 0, d0)) ;       // vector scalar
    ERR1 (v,  GxB_Vector_subassign_FP64_(v , NULL, NULL,  x, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_BOOL_(v0, NULL, NULL,  (bool) 0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_BOOL_(v , NULL, NULL,  (bool) 0, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_INT8_(v0, NULL, NULL,  (int8_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT8_(v , NULL, NULL,  (int8_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_UINT8_(v0, NULL, NULL,  (uint8_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT8_(v , NULL, NULL,  (uint8_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_INT16_(v0, NULL, NULL,  (int16_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT16_(v , NULL, NULL,  (int16_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_UINT16_(v0, NULL, NULL,  (uint16_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT16_(v , NULL, NULL,  (uint16_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_INT32_(v0, NULL, NULL,  (int32_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT32_(v , NULL, NULL,  (int32_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_UINT32_(v0, NULL, NULL,  (uint32_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT32_(v , NULL, NULL,  (uint32_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_INT64_(v0, NULL, NULL,  (int64_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT64_(v , NULL, NULL,  (int64_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_UINT64_(v0, NULL, NULL,  (uint64_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT64_(v , NULL, NULL,  (uint64_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_FP32_(v0, NULL, NULL,  (float) 0, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_FP32_(v , NULL, NULL,  (float) 0, I0, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_UDT_(v0, NULL, NULL,  (void *) X, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UDT_(v , NULL, NULL,  (void *) X, I0, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UDT_(v , NULL, NULL,  (void *) NULL, I, 0, d0)) ;


    ERR1 (A0, GxB_Matrix_subassign_FP64_(A0, NULL, NULL,  x, I0, 0, J0, 0, d0)) ;// matrix scalar
    ERR1 (A,  GxB_Matrix_subassign_FP64_(A , NULL, NULL,  x, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_FP64_(A , NULL, NULL,  x, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_BOOL_(A0, NULL, NULL,  (bool) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_BOOL_(A , NULL, NULL,  (bool) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_BOOL_(A , NULL, NULL,  (bool) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_INT8_(A0, NULL, NULL,  (int8_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT8_(A , NULL, NULL,  (int8_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT8_(A , NULL, NULL,  (int8_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_UINT8_(A0, NULL, NULL,  (uint8_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT8_(A , NULL, NULL,  (uint8_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT8_(A , NULL, NULL,  (uint8_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_INT16_(A0, NULL, NULL,  (int16_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT16_(A , NULL, NULL,  (int16_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT16_(A , NULL, NULL,  (int16_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_UINT16_(A0, NULL, NULL,  (uint16_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT16_(A , NULL, NULL,  (uint16_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT16_(A , NULL, NULL,  (uint16_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_INT32_(A0, NULL, NULL,  (int32_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT32_(A , NULL, NULL,  (int32_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT32_(A , NULL, NULL,  (int32_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_UINT32_(A0, NULL, NULL,  (uint32_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT32_(A , NULL, NULL,  (uint32_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT32_(A , NULL, NULL,  (uint32_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_INT64_(A0, NULL, NULL,  (int64_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT64_(A , NULL, NULL,  (int64_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT64_(A , NULL, NULL,  (int64_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_UINT64_(A0, NULL, NULL,  (uint64_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT64_(A , NULL, NULL,  (uint64_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT64_(A , NULL, NULL,  (uint64_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_FP32_(A0, NULL, NULL,  (float) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_FP32_(A , NULL, NULL,  (float) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_FP32_(A , NULL, NULL,  (float) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GxB_Matrix_subassign_UDT_(A0, NULL, NULL,  (void *) X, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UDT_(A , NULL, NULL,  (void *) X, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UDT_(A , NULL, NULL,  (void *) X, I , 0, J0, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UDT_(A , NULL, NULL,  (void *) NULL, I , 0, J, 0, d0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    v0 = vcrud ;
    A0 = Acrud ;
    d0 = dcrud ;
    op0 = op2crud ;

    ERR1 (v0, GxB_Vector_subassign_(v0, NULL, NULL, v0, I, 0, d0)) ;        // vector assign
    ERR1 (v,  GxB_Vector_subassign_(v , v0  , NULL, v0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_(v , v   , NULL, v0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_(v , v   , NULL, v , I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_(v , v   , op0 , v , I, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_(A0, NULL, NULL, A0, I, 0, J, 0, d0)) ;  // matrix assign
    ERR1 (A,  GxB_Matrix_subassign_(A , A0  , NULL, A0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_(A , A   , NULL, A0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_(A , A   , NULL, A , I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_(A , A   , op0 , A , I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Col_subassign_(A0, NULL, NULL, v0, I, 0,  0, d0)) ;    // column assign
    ERR1 (A,  GxB_Col_subassign_(A , v0  , NULL, v0, I, 0,  0, d0)) ;
    ERR1 (A,  GxB_Col_subassign_(A , v   , NULL, v0, I, 0,  0, d0)) ;
    ERR1 (A,  GxB_Col_subassign_(A , v   , NULL, v , I, 0,  0, d0)) ;
    ERR1 (A,  GxB_Col_subassign_(A , v   , op0 , v , I, 0,  0, NULL)) ;

    ERR1 (A0, GxB_Row_subassign_(A0, NULL, NULL, v0,  0, J, 0, d0)) ;    // row assign
    ERR1 (A,  GxB_Row_subassign_(A , v0  , NULL, v0,  0, J, 0, d0)) ;
    ERR1 (A,  GxB_Row_subassign_(A , v   , NULL, v0,  0, J, 0, d0)) ;
    ERR1 (A,  GxB_Row_subassign_(A , v   , NULL, v ,  0, J, 0, d0)) ;
    ERR1 (A,  GxB_Row_subassign_(A , NULL, op0 , v ,  0, J, 0, NULL)) ;

    ERR1 (v0, GxB_Vector_subassign_FP64 (v0, NULL, NULL,  x, I, 0, d0)) ;       // vector scalar
    ERR1 (v,  GxB_Vector_subassign_FP64 (v , v0  , NULL,  x, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_FP64 (v , v   , NULL,  x, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_FP64 (v , v   , op0 ,  x, I, 0, NULL)) ;

    ERR1 (v0, GxB_Vector_subassign_BOOL (v0, NULL, NULL,  (bool) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_BOOL (v , v0  , NULL,  (bool) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_BOOL (v , v   , NULL,  (bool) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_BOOL (v , v   , op0 ,  (bool) 0, I, 0, NULL)) ;

    ERR1 (v0, GxB_Vector_subassign_INT8 (v0, NULL, NULL,  (int8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT8 (v , v0  , NULL,  (int8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT8 (v , v   , NULL,  (int8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT8 (v , v   , op0 ,  (int8_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GxB_Vector_subassign_UINT8 (v0, NULL, NULL,  (uint8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT8 (v , v0  , NULL,  (uint8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT8 (v , v   , NULL,  (uint8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT8 (v , v   , NULL,  (uint8_t) 0, I, 0, d0)) ;

    ERR1 (v0, GxB_Vector_subassign_INT16 (v0, NULL, NULL,  (int16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT16 (v , v0  , NULL,  (int16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT16 (v , v   , NULL,  (int16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT16 (v , v   , op0 ,  (int16_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GxB_Vector_subassign_UINT16 (v0, NULL, NULL,  (uint16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT16 (v , v0  , NULL,  (uint16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT16 (v , v   , NULL,  (uint16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT16 (v , v   , op0 ,  (uint16_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GxB_Vector_subassign_INT32 (v0, NULL, NULL,  (int32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT32 (v , v0  , NULL,  (int32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT32 (v , v   , NULL,  (int32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT32 (v , v   , op0 ,  (int32_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GxB_Vector_subassign_UINT32 (v0, NULL, NULL,  (uint32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT32 (v , v0  , NULL,  (uint32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT32 (v , v   , NULL,  (uint32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT32 (v , v   , op0 ,  (uint32_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GxB_Vector_subassign_INT64 (v0, NULL, NULL,  (int64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT64 (v , v0  , NULL,  (int64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT64 (v , v   , NULL,  (int64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_INT64 (v , v   , op0 ,  (int64_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GxB_Vector_subassign_UINT64 (v0, NULL, NULL,  (uint64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT64 (v , v0  , NULL,  (uint64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT64 (v , v   , NULL,  (uint64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UINT64 (v , v   , op0 ,  (uint64_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GxB_Vector_subassign_FP32 (v0, NULL, NULL,  (float) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_FP32 (v , v0  , NULL,  (float) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_FP32 (v , v   , NULL,  (float) 0, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_FP32 (v , v   , op0 ,  (float) 0, I, 0, NULL)) ;

    ERR1 (v,  GxB_Vector_subassign_UDT (v0, NULL, NULL,  (void *) X, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UDT (v , v0  , NULL,  (void *) X, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UDT (v , v   , NULL,  (void *) X, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UDT (v , v   , op0 ,  (void *) X, I, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_FP64_(A0, NULL, NULL,  x, I, 0, J, 0, d0)) ;  // matrix scalar
    ERR1 (A,  GxB_Matrix_subassign_FP64_(A , A0  , NULL,  x, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_FP64_(A , A   , NULL,  x, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_FP64_(A , A   , op0 ,  x, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_BOOL_(A0, NULL, NULL,  (bool) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_BOOL_(A , A0  , NULL,  (bool) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_BOOL_(A , A   , NULL,  (bool) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_BOOL_(A , A   , op0 ,  (bool) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_INT8_(A0, NULL, NULL,  (int8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT8_(A , A0  , NULL,  (int8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT8_(A , A   , NULL,  (int8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT8_(A , A   , op0 ,  (int8_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_UINT8_(A0, NULL, NULL,  (uint8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT8_(A , A0  , NULL,  (uint8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT8_(A , A   , NULL,  (uint8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT8_(A , A   , op0 ,  (uint8_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_INT16_(A0, NULL, NULL,  (int16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT16_(A , A0  , NULL,  (int16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT16_(A , A   , NULL,  (int16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT16_(A , A   , op0 ,  (int16_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_UINT16_(A0, NULL, NULL,  (uint16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT16_(A , A0  , NULL,  (uint16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT16_(A , A   , NULL,  (uint16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT16_(A , A   , op0 ,  (uint16_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_INT32_(A0, NULL, NULL,  (int32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT32_(A , A0  , NULL,  (int32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT32_(A , A   , NULL,  (int32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT32_(A , A   , op0 ,  (int32_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_UINT32_(A0, NULL, NULL,  (uint32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT32_(A , A0  , NULL,  (uint32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT32_(A , A   , NULL,  (uint32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT32_(A , A   , op0 ,  (uint32_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_INT64_(A0, NULL, NULL,  (int64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT64_(A , A0  , NULL,  (int64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT64_(A , A   , NULL,  (int64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_INT64_(A , A   , op0 ,  (int64_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_UINT64_(A0, NULL, NULL,  (uint64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT64_(A , A0  , NULL,  (uint64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT64_(A , A   , NULL,  (uint64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UINT64_(A , A   , op0 ,  (uint64_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_FP32_(A0, NULL, NULL,  (float) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_FP32_(A , A0  , NULL,  (float) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_FP32_(A , A   , NULL,  (float) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_FP32_(A , A   , op0 ,  (float) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GxB_Matrix_subassign_UDT_(A0, NULL, NULL,  (void *) X, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UDT_(A , A0  , NULL,  (void *) X, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UDT_(A , A   , NULL,  (void *) X, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_UDT_(A , A   , op0 ,  (void *) X, I, 0, J, 0, NULL)) ;

    v0 = NULL ;
    A0 = NULL ;
    d0 = NULL ;
    op0 = Complex_plus ;
    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;

    ERR1 (v,  GxB_Vector_subassign_(v, z , NULL, v, I, 0, d0)) ;            // vector assign
    ERR1 (v,  GxB_Vector_subassign_(v, v0, op0 , v, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_(v, v0, op0 , z, I, 0, d0)) ;
    ERR1 (z,  GxB_Vector_subassign_(z, v0, o2  , v, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_(v, v0, o2  , z, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_(v, v0, NULL, z, I, 0, d0)) ;

    ERR1 (A,  GxB_Matrix_subassign_(A, Z , NULL, A, I, 0, J, 0, d0)) ;      // matrix assign
    ERR1 (A,  GxB_Matrix_subassign_(A, A0, op0 , A, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_(A, A0, op0 , Z, I, 0, J, 0, d0)) ;
    ERR1 (Z,  GxB_Matrix_subassign_(Z, A0, o2  , A, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_(A, A0, o2  , Z, I, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Matrix_subassign_(A, A0, NULL, Z, I, 0, J, 0, d0)) ;

    ERR1 (A,  GxB_Col_subassign_(A, z , NULL, v, I, 0, 0, d0)) ;         // column assign
    ERR1 (A,  GxB_Col_subassign_(A, v0, op0 , v, I, 0, 0, d0)) ;
    ERR1 (A,  GxB_Col_subassign_(A, v0, op0 , z, I, 0, 0, d0)) ;
    ERR1 (Z,  GxB_Col_subassign_(Z, v0, o2  , v, I, 0, 0, d0)) ;
    ERR1 (A,  GxB_Col_subassign_(A, v0, o2  , z, I, 0, 0, d0)) ;
    ERR1 (A,  GxB_Col_subassign_(A, v0, NULL, z, I, 0, 0, d0)) ;

    ERR1 (A,  GxB_Row_subassign_(A, z , NULL, v, 0, J, 0, d0)) ;         // row assign
    ERR1 (A,  GxB_Row_subassign_(A, v0, op0 , v, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Row_subassign_(A, v0, op0 , z, 0, J, 0, d0)) ;
    ERR1 (Z,  GxB_Row_subassign_(Z, v0, o2  , v, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Row_subassign_(A, v0, o2  , z, 0, J, 0, d0)) ;
    ERR1 (A,  GxB_Row_subassign_(A, v0, NULL, z, 0, J, 0, d0)) ;

    ERR1 (v,  GxB_Vector_subassign_FP64_(v, z , NULL, x, I, 0, d0)) ;            // vector scalar

    if (Complex == GxB_FC64)
    {
        OK (GxB_Vector_subassign_FP64_(v, v0, op0 , x, I, 0, d0)) ;
    }
    else
    {
        ERR1 (v,  GxB_Vector_subassign_FP64_(v, v0, op0 , x, I, 0, d0)) ;
    }

    expected = GrB_DOMAIN_MISMATCH ;

    ERR1 (v,  GxB_Vector_subassign_UDT_(v, v0, op0 ,(void *) &c, I, 0, d0)) ;

    if (Complex == GxB_FC64)
    {
        OK (GxB_Vector_subassign_FP64_(z, v0, o2  , x, I, 0, d0)) ;
    }
    else
    {
        ERR1 (z,  GxB_Vector_subassign_FP64_(z, v0, o2  , x, I, 0, d0)) ;
    }

    ERR1 (v,  GxB_Vector_subassign_UDT_(v, v0, o2  ,(void *) &c, I, 0, d0)) ;
    ERR1 (v,  GxB_Vector_subassign_UDT_(v, v0, NULL,(void *) &c, I, 0, d0)) ;

    // matrix scalar
    if (Complex == GxB_FC64)
    {
        expected = GrB_DIMENSION_MISMATCH ;
        OK (GxB_Matrix_subassign_FP64_(A, A0, op0 , x, I, 0, J, 0, d0)) ;
        OK (GxB_Matrix_subassign_FP64_(Z, A0, o2  , x, I, 0, J, 0, d0)) ;
    }
    else
    {
        expected = GrB_DOMAIN_MISMATCH ;
        ERR1 (A, GxB_Matrix_subassign_FP64_(A, A0, op0 , x, I, 0, J, 0, d0)) ;
        ERR1 (Z, GxB_Matrix_subassign_FP64_(Z, A0, o2  , x, I, 0, J, 0, d0)) ;
    }

    ERR1 (A, GxB_Matrix_subassign_FP64_(A, Z , NULL, x, I, 0, J, 0, d0)) ;

    expected = GrB_DOMAIN_MISMATCH ;
    ERR1 (A, GxB_Matrix_subassign_UDT_(A, A0, op0 ,(void *) &c , I, 0, J, 0, d0)) ;
    ERR1 (A, GxB_Matrix_subassign_UDT_(A, A0, o2  ,(void *) &c , I, 0, J, 0, d0)) ;
    ERR1 (A, GxB_Matrix_subassign_UDT_(A, A0, NULL,(void *) &c , I, 0, J, 0, d0)) ;

    expected = GrB_DIMENSION_MISMATCH ;

    ERR1 (A, GxB_Matrix_subassign_(A, NULL, NULL, A, I, 2, J, 3, d0)) ;
    ERR1 (A, GxB_Matrix_subassign_(A, NULL, NULL, A, I, 2, J, 3, dtn)) ;
    ERR1 (A, GxB_Row_subassign_(A , v   , NULL, v ,  0, J, 0, NULL)) ;

    fprintf (ff, "test for indices out of bounds:\n") ;
    OK (GxB_Matrix_fprint (A, "A", GxB_COMPLETE, ff)) ;
    OK (GxB_Matrix_fprint (C, "C", GxB_COMPLETE, ff)) ;
    for (int k = 0 ; k < 3 ; k++)
    {
        fprintf (ff, "I [%d] = %g\n", k, (double) I [k]) ;
    }
    for (int k = 0 ; k < 2 ; k++)
    {
        fprintf (ff, "J [%d] = %g\n", k, (double) J [k]) ;
    }
    expected = GrB_INDEX_OUT_OF_BOUNDS ;

    OK (GrB_Matrix_dup (&A4, A)) ;
    ERR1 (A4, GxB_Matrix_subassign_(A4, NULL, GrB_PLUS_FP64, C, I, 3, J, 2, NULL)) ;
    GrB_Matrix_error_(&err, A4) ;
    fprintf (ff, "done bounds test: error returned:\n%s\n", err) ;
    OK (GrB_Matrix_free_(&A4)) ;

    uint64_t I3 [5] = { 0,   1,   2,   3,    4 } ;      // OK
    uint64_t J3 [5] = { 0,   1,   2,   3,    4 } ;      // OK

    OK (GxB_Matrix_fprint_(A, GxB_COMPLETE, NULL)) ;
    OK (GxB_Matrix_fprint_(A, GxB_COMPLETE, ff)) ;
    OK (GxB_Matrix_subassign_(A, NULL, GrB_PLUS_FP64, C, I3, 3, J3, 2, NULL)) ;

    OK (GxB_Matrix_subassign_(C, C, GrB_PLUS_FP64, C, I3, 3, J3, 2, NULL)) ;

    J3 [0] = 999 ;
    OK (GrB_Matrix_dup (&C4, C)) ;
    ERR1 (C4, GxB_Matrix_subassign_(C4, C4, GrB_PLUS_FP64, C4, I3, 3, J3, 2, NULL)) ;
    OK (GrB_Matrix_free_(&C4)) ;

    expected = GrB_INVALID_INDEX ;
    OK (GrB_Matrix_dup (&A4, A)) ;
    ERR1 (A4, GxB_Matrix_subassign_FP64_(A4, NULL, GrB_PLUS_FP64, x_double, I3, 1, J3, 1, NULL));
    OK (GrB_Matrix_free_(&A4)) ;

    J3 [0] = 0 ;
    I3 [0] = 999 ;

    OK (GrB_Matrix_dup (&A4, A)) ;
    ERR1 (A4, GxB_Matrix_subassign_FP64_(A4, NULL, GrB_PLUS_FP64, x_double, I3, 1, J3, 1, NULL));
    OK (GrB_Matrix_free_(&A4)) ;

    //--------------------------------------------------------------------------
    // assign
    //--------------------------------------------------------------------------

    printf ("GrB_assign---------------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    // GrB_Vector_assign   (w,mask,acc,u,I,ni,d)
    // GrB_Matrix_assign   (C,Mask,acc,A,I,ni,J,nj,d)
    // GrB_Col_assign      (C,mask,acc,u,I,ni,j,d)
    // GrB_Row_assign      (C,mask,acc,u,i,J,nj,d)
    // GrB_Vector_assign_T (w,mask,acc,x,I,ni,d)
    // GrB_Matrix_assign_T (C,Mask,acc,x,I,ni,J,nj,d)

    ERR1 (v0, GrB_Vector_assign_(v0, NULL, NULL, v0, I0, 0, d0)) ;          // vector assign
    ERR1 (v,  GrB_Vector_assign_(v , NULL, NULL, v0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_(v , NULL, NULL, v , I0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_(A0, NULL, NULL, A0, I0, 0, J0, 0, d0)) ;   // matrix assign
    ERR1 (A,  GrB_Matrix_assign_(A , NULL, NULL, A0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_(A , NULL, NULL, A , I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_(A , NULL, NULL, A , I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Col_assign_(A0, NULL, NULL, v0, I0, 0,  0, d0)) ;      // column assign
    ERR1 (A,  GrB_Col_assign_(A , NULL, NULL, v0, I0, 0,  0, d0)) ;
    ERR1 (A,  GrB_Col_assign_(A , NULL, NULL, v , I0, 0,  0, d0)) ;

    ERR1 (A0, GrB_Row_assign_(A0, NULL, NULL, v0,  0, J0, 0, d0)) ;      // row assign
    ERR1 (A,  GrB_Row_assign_(A , NULL, NULL, v0,  0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Row_assign_(A , NULL, NULL, v ,  0, J0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_FP64_(v0, NULL, NULL,  x, I0, 0, d0)) ;          // vector scalar
    ERR1 (v,  GrB_Vector_assign_FP64_(v , NULL, NULL,  x, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_BOOL_(v0, NULL, NULL,  (bool) 0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_BOOL_(v , NULL, NULL,  (bool) 0, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_INT8_(v0, NULL, NULL,  (int8_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT8_(v , NULL, NULL,  (int8_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_UINT8_(v0, NULL, NULL,  (uint8_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT8_(v , NULL, NULL,  (uint8_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_INT16_(v0, NULL, NULL,  (int16_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT16_(v , NULL, NULL,  (int16_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_UINT16_(v0, NULL, NULL,  (uint16_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT16_(v , NULL, NULL,  (uint16_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_INT32_(v0, NULL, NULL,  (int32_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT32_(v , NULL, NULL,  (int32_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_UINT32_(v0, NULL, NULL,  (uint32_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT32_(v , NULL, NULL,  (uint32_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_INT64_(v0, NULL, NULL,  (int64_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT64_(v , NULL, NULL,  (int64_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_UINT64_(v0, NULL, NULL,  (uint64_t) 0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT64_(v , NULL, NULL,  (uint64_t) 0, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_FP32_(v0, NULL, NULL,  (float) 0, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_FP32_(v , NULL, NULL,  (float) 0, I0, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_UDT_(v0, NULL, NULL,  (void *) X, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UDT_(v , NULL, NULL,  (void *) X, I0, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UDT_(v , NULL, NULL,  (void *) NULL, I, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_FP64_(A0, NULL, NULL,  x, I0, 0, J0, 0, d0)) ;   // matrix scalar
    ERR1 (A,  GrB_Matrix_assign_FP64_(A , NULL, NULL,  x, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_FP64_(A , NULL, NULL,  x, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_BOOL_(A0, NULL, NULL,  (bool) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_BOOL_(A , NULL, NULL,  (bool) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_BOOL_(A , NULL, NULL,  (bool) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_INT8_(A0, NULL, NULL,  (int8_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT8_(A , NULL, NULL,  (int8_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT8_(A , NULL, NULL,  (int8_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_UINT8_(A0, NULL, NULL,  (uint8_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT8_(A , NULL, NULL,  (uint8_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT8_(A , NULL, NULL,  (uint8_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_INT16_(A0, NULL, NULL,  (int16_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT16_(A , NULL, NULL,  (int16_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT16_(A , NULL, NULL,  (int16_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_UINT16_(A0, NULL, NULL,  (uint16_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT16_(A , NULL, NULL,  (uint16_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT16_(A , NULL, NULL,  (uint16_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_INT32_(A0, NULL, NULL,  (int32_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT32_(A , NULL, NULL,  (int32_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT32_(A , NULL, NULL,  (int32_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_UINT32_(A0, NULL, NULL,  (uint32_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT32_(A , NULL, NULL,  (uint32_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT32_(A , NULL, NULL,  (uint32_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_INT64_(A0, NULL, NULL,  (int64_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT64_(A , NULL, NULL,  (int64_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT64_(A , NULL, NULL,  (int64_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_UINT64_(A0, NULL, NULL,  (uint64_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT64_(A , NULL, NULL,  (uint64_t) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT64_(A , NULL, NULL,  (uint64_t) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_FP32_(A0, NULL, NULL,  (float) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_FP32_(A , NULL, NULL,  (float) 0, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_FP32_(A , NULL, NULL,  (float) 0, I , 0, J0, 0, d0)) ;

    ERR1 (A0, GrB_Matrix_assign_UDT_(A0, NULL, NULL,  (void *) X, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UDT_(A , NULL, NULL,  (void *) X, I0, 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UDT_(A , NULL, NULL,  (void *) X, I , 0, J0, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UDT_(A , NULL, NULL,  (void *) NULL, I , 0, J, 0, d0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    v0 = vcrud ;
    A0 = Acrud ;
    d0 = dcrud ;
    op0 = op2crud ;

    ERR1 (v0, GrB_Vector_assign_(v0, NULL, NULL, v0, I, 0, d0)) ;          // vector assign
    ERR1 (v,  GrB_Vector_assign_(v , v0  , NULL, v0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_(v , v   , NULL, v0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_(v , v   , NULL, v , I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_(v , v   , op0 , v , I, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_(A0, NULL, NULL, A0, I, 0, J, 0, d0)) ;   // matrix assign
    ERR1 (A,  GrB_Matrix_assign_(A , A0  , NULL, A0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_(A , A   , NULL, A0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_(A , A   , NULL, A , I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_(A , A   , op0 , A , I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Col_assign_(A0, NULL, NULL, v0, I, 0,  0, d0)) ;      // column assign
    ERR1 (A,  GrB_Col_assign_(A , v0  , NULL, v0, I, 0,  0, d0)) ;
    ERR1 (A,  GrB_Col_assign_(A , v   , NULL, v0, I, 0,  0, d0)) ;
    ERR1 (A,  GrB_Col_assign_(A , v   , NULL, v , I, 0,  0, d0)) ;
    ERR1 (A,  GrB_Col_assign_(A , v   , op0 , v , I, 0,  0, NULL)) ;

    ERR1 (A0, GrB_Row_assign_(A0, NULL, NULL, v0,  0, J, 0, d0)) ;      // row assign
    ERR1 (A,  GrB_Row_assign_(A , v0  , NULL, v0,  0, J, 0, d0)) ;
    ERR1 (A,  GrB_Row_assign_(A , v   , NULL, v0,  0, J, 0, d0)) ;
    ERR1 (A,  GrB_Row_assign_(A , v   , NULL, v ,  0, J, 0, d0)) ;
    ERR1 (A,  GrB_Row_assign_(A , NULL, op0 , v ,  0, J, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_FP64_(v0, NULL, NULL,  x, I, 0, d0)) ;          // vector scalar
    ERR1 (v,  GrB_Vector_assign_FP64_(v , v0  , NULL,  x, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_FP64_(v , v   , NULL,  x, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_FP64_(v , v   , op0 ,  x, I, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_BOOL_(v0, NULL, NULL,  (bool) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_BOOL_(v , v0  , NULL,  (bool) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_BOOL_(v , v   , NULL,  (bool) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_BOOL_(v , v   , op0 ,  (bool) 0, I, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_INT8_(v0, NULL, NULL,  (int8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT8_(v , v0  , NULL,  (int8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT8_(v , v   , NULL,  (int8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT8_(v , v   , op0 ,  (int8_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_UINT8_(v0, NULL, NULL,  (uint8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT8_(v , v0  , NULL,  (uint8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT8_(v , v   , NULL,  (uint8_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT8_(v , v   , NULL,  (uint8_t) 0, I, 0, d0)) ;

    ERR1 (v0, GrB_Vector_assign_INT16_(v0, NULL, NULL,  (int16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT16_(v , v0  , NULL,  (int16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT16_(v , v   , NULL,  (int16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT16_(v , v   , op0 ,  (int16_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_UINT16_(v0, NULL, NULL,  (uint16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT16_(v , v0  , NULL,  (uint16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT16_(v , v   , NULL,  (uint16_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT16_(v , v   , op0 ,  (uint16_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_INT32_(v0, NULL, NULL,  (int32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT32_(v , v0  , NULL,  (int32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT32_(v , v   , NULL,  (int32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT32_(v , v   , op0 ,  (int32_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_UINT32_(v0, NULL, NULL,  (uint32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT32_(v , v0  , NULL,  (uint32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT32_(v , v   , NULL,  (uint32_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT32_(v , v   , op0 ,  (uint32_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_INT64_(v0, NULL, NULL,  (int64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT64_(v , v0  , NULL,  (int64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT64_(v , v   , NULL,  (int64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_INT64_(v , v   , op0 ,  (int64_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_UINT64_(v0, NULL, NULL,  (uint64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT64_(v , v0  , NULL,  (uint64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT64_(v , v   , NULL,  (uint64_t) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UINT64_(v , v   , op0 ,  (uint64_t) 0, I, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_FP32_(v0, NULL, NULL,  (float) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_FP32_(v , v0  , NULL,  (float) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_FP32_(v , v   , NULL,  (float) 0, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_FP32_(v , v   , op0 ,  (float) 0, I, 0, NULL)) ;

    ERR1 (v0, GrB_Vector_assign_UDT_(v0, NULL, NULL,  (void *) X, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UDT_(v , v0  , NULL,  (void *) X, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UDT_(v , v   , NULL,  (void *) X, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UDT_(v , v   , op0 ,  (void *) X, I, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_FP64_(A0, NULL, NULL,  x, I, 0, J, 0, d0)) ;   // matrix scalar
    ERR1 (A,  GrB_Matrix_assign_FP64_(A , A0  , NULL,  x, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_FP64_(A , A   , NULL,  x, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_FP64_(A , A   , op0 ,  x, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_BOOL_(A0, NULL, NULL,  (bool) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_BOOL_(A , A0  , NULL,  (bool) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_BOOL_(A , A   , NULL,  (bool) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_BOOL_(A , A   , op0 ,  (bool) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_INT8_(A0, NULL, NULL,  (int8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT8_(A , A0  , NULL,  (int8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT8_(A , A   , NULL,  (int8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT8_(A , A   , op0 ,  (int8_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_UINT8_(A0, NULL, NULL,  (uint8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT8_(A , A0  , NULL,  (uint8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT8_(A , A   , NULL,  (uint8_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT8_(A , A   , op0 ,  (uint8_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_INT16_(A0, NULL, NULL,  (int16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT16_(A , A0  , NULL,  (int16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT16_(A , A   , NULL,  (int16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT16_(A , A   , op0 ,  (int16_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_UINT16_(A0, NULL, NULL,  (uint16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT16_(A , A0  , NULL,  (uint16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT16_(A , A   , NULL,  (uint16_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT16_(A , A   , op0 ,  (uint16_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_INT32_(A0, NULL, NULL,  (int32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT32_(A , A0  , NULL,  (int32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT32_(A , A   , NULL,  (int32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT32_(A , A   , op0 ,  (int32_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_UINT32_(A0, NULL, NULL,  (uint32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT32_(A , A0  , NULL,  (uint32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT32_(A , A   , NULL,  (uint32_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT32_(A , A   , op0 ,  (uint32_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_INT64_(A0, NULL, NULL,  (int64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT64_(A , A0  , NULL,  (int64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT64_(A , A   , NULL,  (int64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_INT64_(A , A   , op0 ,  (int64_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_UINT64_(A0, NULL, NULL,  (uint64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT64_(A , A0  , NULL,  (uint64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT64_(A , A   , NULL,  (uint64_t) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UINT64_(A , A   , op0 ,  (uint64_t) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_FP32_(A0, NULL, NULL,  (float) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_FP32_(A , A0  , NULL,  (float) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_FP32_(A , A   , NULL,  (float) 0, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_FP32_(A , A   , op0 ,  (float) 0, I, 0, J, 0, NULL)) ;

    ERR1 (A0, GrB_Matrix_assign_UDT_(A0, NULL, NULL,  (void *) X, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UDT_(A , A0  , NULL,  (void *) X, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UDT_(A , A   , NULL,  (void *) X, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UDT_(A , A   , op0 ,  (void *) X, I, 0, J, 0, NULL)) ;

    v0 = NULL ;
    A0 = NULL ;
    d0 = NULL ;
    op0 = Complex_plus ;

    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;

    ERR1 (v,  GrB_Vector_assign_(v, z , NULL, v, I, 0, d0)) ;               // vector assign
    ERR1 (v,  GrB_Vector_assign_(v, v0, op0 , v, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_(v, v0, op0 , z, I, 0, d0)) ;
    ERR1 (z,  GrB_Vector_assign_(z, v0, o2  , v, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_(v, v0, o2  , z, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_(v, v0, NULL, z, I, 0, d0)) ;

    ERR1 (A,  GrB_Matrix_assign_(A, Z , NULL, A, I, 0, J, 0, d0)) ;         // matrix assign
    ERR1 (A,  GrB_Matrix_assign_(A, A0, op0 , A, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_(A, A0, op0 , Z, I, 0, J, 0, d0)) ;
    ERR1 (Z,  GrB_Matrix_assign_(Z, A0, o2  , A, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_(A, A0, o2  , Z, I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_(A, A0, NULL, Z, I, 0, J, 0, d0)) ;

    ERR1 (A,  GrB_Col_assign_(A, z , NULL, v, I, 0, 0, d0)) ;            // column assign
    ERR1 (A,  GrB_Col_assign_(A, v0, op0 , v, I, 0, 0, d0)) ;
    ERR1 (A,  GrB_Col_assign_(A, v0, op0 , z, I, 0, 0, d0)) ;
    ERR1 (Z,  GrB_Col_assign_(Z, v0, o2  , v, I, 0, 0, d0)) ;
    ERR1 (A,  GrB_Col_assign_(A, v0, o2  , z, I, 0, 0, d0)) ;
    ERR1 (A,  GrB_Col_assign_(A, v0, NULL, z, I, 0, 0, d0)) ;

    ERR1 (A,  GrB_Row_assign_(A, z , NULL, v, 0, J, 0, d0)) ;            // row assign
    ERR1 (A,  GrB_Row_assign_(A, v0, op0 , v, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Row_assign_(A, v0, op0 , z, 0, J, 0, d0)) ;
    ERR1 (Z,  GrB_Row_assign_(Z, v0, o2  , v, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Row_assign_(A, v0, o2  , z, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Row_assign_(A, v0, NULL, z, 0, J, 0, d0)) ;

    // vector scalar and matrix-scalar
    if (Complex == GxB_FC64)
    {
        OK (GrB_Vector_assign_FP64_(v, z , NULL, x, I, 0, d0)) ;
        OK (GrB_Vector_assign_FP64_(v, v0, op0 , x, I, 0, d0)) ;
        OK (GrB_Vector_assign_FP64_(z, v0, o2  , x, I, 0, d0)) ;
        OK (GrB_Matrix_assign_FP64_(A, A0, op0 , x, I, 0, J, 0, d0)) ;
        OK (GrB_Matrix_assign_FP64_(Z, A0, o2  , x, I, 0, J, 0, d0)) ;
    }
    else
    {
        ERR1 (v,  GrB_Vector_assign_FP64_(v, z , NULL, x, I, 0, d0)) ;
        ERR1 (v,  GrB_Vector_assign_FP64_(v, v0, op0 , x, I, 0, d0)) ;
        ERR1 (z,  GrB_Vector_assign_FP64_(z, v0, o2  , x, I, 0, d0)) ;
        ERR1 (A,  GrB_Matrix_assign_FP64_(A, A0, op0 , x, I, 0, J, 0, d0)) ;
        ERR1 (Z,  GrB_Matrix_assign_FP64_(Z, A0, o2  , x, I, 0, J, 0, d0)) ;
    }

    expected = GrB_DOMAIN_MISMATCH ;
    ERR1 (v,  GrB_Vector_assign_UDT_(v, v0, op0 ,(void *) &c, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UDT_(v, v0, o2  ,(void *) &c, I, 0, d0)) ;
    ERR1 (v,  GrB_Vector_assign_UDT_(v, v0, NULL,(void *) &c, I, 0, d0)) ;

    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;

    ERR1 (A,  GrB_Matrix_assign_FP64_(A, Z , NULL, x, I, 0, J, 0, d0)) ;

    expected = GrB_DOMAIN_MISMATCH ;
    ERR1 (A,  GrB_Matrix_assign_UDT_(A, A0, op0 ,(void *) &c , I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UDT_(A, A0, o2  ,(void *) &c , I, 0, J, 0, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_UDT_(A, A0, NULL,(void *) &c , I, 0, J, 0, d0)) ;

    expected = GrB_DIMENSION_MISMATCH ;

    ERR1 (A,  GrB_Matrix_assign_(A, NULL, NULL, A, I, 2, J, 3, d0)) ;
    ERR1 (A,  GrB_Matrix_assign_(A, NULL, NULL, A, I, 2, J, 3, dtn)) ;
    ERR1 (A,  GrB_Row_assign_(A , v   , NULL, v ,  0, J, 0, NULL)) ;

    GrB_Vector v5 ;
    OK (GrB_Vector_new (&v5, GrB_BOOL, 5)) ;
    GxB_Matrix_fprint_(A, G3, NULL) ;
    GxB_Vector_fprint_(v5, G3, NULL) ;
    GxB_Vector_fprint_(v, G3, NULL) ;
    ERR1 (A,  GrB_Col_assign_(A, v5 , NULL, v, GrB_ALL, 0, 0, NULL)) ; // column assign
    GrB_Matrix_error_(&err, A) ;
    printf ("mask wrong size:\n%s\n", err) ;
    OK (GrB_Vector_free_(&v5)) ;

    // matrix assign, mask wrong size
    GrB_Matrix A5 ;
    OK (GrB_Matrix_new (&A5, GrB_BOOL, 5, 5)) ;
    GB_Matrix_check (A, "A", G3, NULL) ;
    GB_Matrix_check (A5, "A5", G3, NULL) ;
    ERR1 (A,  GrB_Matrix_assign_(A, A5, NULL, A, GrB_ALL, 0, GrB_ALL, 0, NULL)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("mask wrong size:\n%s\n", err) ;
    OK (GrB_Matrix_free_(&A5)) ;

    // change of op-2nd type
    int64_t I_0 = 0 ;
    int64_t J_0 = 0 ;
    double X_0 = 42 ;
    OK (GrB_Matrix_new (&A5, GrB_FP32, 5, 5)) ;
    OK (GrB_Matrix_assign_FP32 (A5, NULL, GrB_SECOND_FP32, 42,
        GrB_ALL, 0, GrB_ALL, 0, NULL)) ;
    GB_Matrix_check (A5, "A5 with 2nd:fp32", G3, NULL) ;
    OK (GrB_Matrix_assign_FP32 (A5, NULL, GrB_SECOND_BOOL, 42,
        GrB_ALL, 0, GrB_ALL, 0, NULL)) ;
    GB_Matrix_check (A5, "A5 with 2nd:bool", G3, NULL) ;
    OK (GrB_Matrix_nvals (&nvals, A5)) ;
    OK (GrB_Matrix_wait_(A5, GrB_MATERIALIZE)) ;
    CHECK (nvals == 25) ;
    GB_Matrix_check (A5, "A5 done", G3, NULL) ;

    OK (GrB_Matrix_free_(&A5)) ;

    //--------------------------------------------------------------------------
    // apply
    //--------------------------------------------------------------------------

    printf ("GrB_apply----------------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR1 (v0, GrB_Vector_apply (v0, NULL, NULL, NULL, v0, d0)) ;
    ERR1 (v,  GrB_Vector_apply (v , NULL, NULL, NULL, v0, d0)) ;
    ERR1 (v,  GrB_Vector_apply (v , NULL, NULL, NULL, v , d0)) ;

    ERR1 (A0, GrB_Matrix_apply (A0, NULL, NULL, NULL, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_apply (A , NULL, NULL, NULL, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_apply (A , NULL, NULL, NULL, A , d0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    v0 = vcrud ;
    A0 = Acrud ;
    d0 = dcrud ;
    op0 = op2crud ;

    ERR1 (v0, GrB_Vector_apply_(v0, NULL, NULL, op1crud, v0, d0)) ;
    ERR1 (v,  GrB_Vector_apply_(v , v0  , NULL, op1crud, v0, d0)) ;
    ERR1 (v,  GrB_Vector_apply_(v , v   , NULL, op1crud, v0, d0)) ;
    ERR1 (v,  GrB_Vector_apply_(v , v   , NULL, op1crud, v , d0)) ;
    ERR1 (v,  GrB_Vector_apply_(v , v   , op0 , op1crud, v , NULL)) ;
    ERR1 (v,  GrB_Vector_apply_(v , v   , NULL, op1crud, v , NULL)) ;

    ERR1 (A0, GrB_Matrix_apply_(A0, NULL, NULL, op1crud, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_apply_(A , A0  , NULL, op1crud, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_apply_(A , A   , NULL, op1crud, A0, d0)) ;
    ERR1 (A,  GrB_Matrix_apply_(A , A   , NULL, op1crud, A , d0)) ;
    ERR1 (A,  GrB_Matrix_apply_(A , A   , op0 , op1crud, A , NULL)) ;
    ERR1 (A,  GrB_Matrix_apply_(A , A   , NULL, op1crud, A , NULL)) ;

    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;

    o2  = Complex_plus ;
    ERR1 (A,  GrB_Matrix_apply_(A, Z   , NULL, GrB_AINV_FP64, A, NULL)) ;

    if (Complex == GxB_FC64)
    {
        OK (GrB_Matrix_apply_(A, NULL, o2  , GrB_AINV_FP64, A, NULL)) ;
        OK (GrB_Matrix_apply_(Z, NULL, NULL, GrB_AINV_FP64, Z, NULL)) ;
        OK (GrB_Matrix_apply_BinaryOp1st_INT32_(Z, NULL, NULL, o2, 1, Z, NULL)) ;
        OK (GrB_Matrix_apply_BinaryOp2nd_INT32_(Z, NULL, NULL, o2, Z, 1, NULL)) ;
    }
    else
    {
        ERR1 (A,  GrB_Matrix_apply_(A, NULL, o2  , GrB_AINV_FP64, A, NULL)) ;
        ERR1 (Z,  GrB_Matrix_apply_(Z, NULL, NULL, GrB_AINV_FP64, Z, NULL)) ;
        ERR1 (Z,  GrB_Matrix_apply_BinaryOp1st_INT32_(Z, NULL, NULL, o2, 1, Z, NULL)) ;
        ERR1 (Z,  GrB_Matrix_apply_BinaryOp2nd_INT32_(Z, NULL, NULL, o2, Z, 1, NULL)) ;
    }

    ERR1 (A,  GrB_Matrix_apply_(A, NULL, o2  , GrB_AINV_FP64, Z, NULL)) ;
    ERR1 (A,  GrB_Matrix_apply_(A, NULL, NULL, GrB_AINV_FP64, Z, NULL)) ;
    ERR1 (Z,  GrB_Matrix_apply_(Z, NULL, NULL, GrB_AINV_FP64, A, NULL)) ;

    ERR1 (Z,  GrB_Matrix_apply_BinaryOp1st_INT32_(Z, NULL, NULL, o2, 1, A, NULL)) ;
    ERR1 (Z,  GrB_Matrix_apply_BinaryOp2nd_INT32_(Z, NULL, NULL, o2, A, 1, NULL)) ;

    v0 = NULL ;
    A0 = NULL ;
    d0 = NULL ;
    op0 = NULL ;

    expected = GrB_DIMENSION_MISMATCH ;

    ERR1 (A,  GrB_Matrix_apply_(A , NULL, NULL, GrB_AINV_FP64, C , d0)) ;

    //--------------------------------------------------------------------------
    // reduce to scalar
    //--------------------------------------------------------------------------

    printf ("GrB_reduce (to scalar)---------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    o2 = GrB_PLUS_FP32 ;
    m2 = GxB_TIMES_FP64_MONOID ;

    // matrix to scalar

    ERR (GrB_Matrix_reduce_BOOL_((bool     *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_BOOL_((bool     *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_BOOL_((bool     *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_INT8_((int8_t   *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_INT8_((int8_t   *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_INT8_((int8_t   *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_UINT8_((uint8_t  *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_UINT8_((uint8_t  *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_UINT8_((uint8_t  *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_INT16_((int16_t  *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_INT16_((int16_t  *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_INT16_((int16_t  *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_UINT16_((uint16_t *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_UINT16_((uint16_t *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_UINT16_((uint16_t *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_INT32_((int32_t  *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_INT32_((int32_t  *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_INT32_((int32_t  *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_UINT32_((uint32_t *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_UINT32_((uint32_t *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_UINT32_((uint32_t *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_INT64_((int64_t  *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_INT64_((int64_t  *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_INT64_((int64_t  *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_UINT64_((uint64_t *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_UINT64_((uint64_t *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_UINT64_((uint64_t *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_FP32_((float    *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_FP32_((float    *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_FP32_((float    *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_FP64_((double   *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_FP64_((double   *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_FP64_((double   *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_UDT_((void     *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_UDT_((void     *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_UDT_((void     *) NULL, op0, m2, A , d0)) ;

    // vector to scalar

    ERR (GrB_Vector_reduce_BOOL_((bool     *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_BOOL_((bool     *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_BOOL_((bool     *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_INT8_((int8_t   *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_INT8_((int8_t   *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_INT8_((int8_t   *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_UINT8_((uint8_t  *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_UINT8_((uint8_t  *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_UINT8_((uint8_t  *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_INT16_((int16_t  *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_INT16_((int16_t  *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_INT16_((int16_t  *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_UINT16_((uint16_t *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_UINT16_((uint16_t *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_UINT16_((uint16_t *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_INT32_((int32_t  *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_INT32_((int32_t  *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_INT32_((int32_t  *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_UINT32_((uint32_t *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_UINT32_((uint32_t *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_UINT32_((uint32_t *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_INT64_((int64_t  *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_INT64_((int64_t  *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_INT64_((int64_t  *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_UINT64_((uint64_t *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_UINT64_((uint64_t *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_UINT64_((uint64_t *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_FP32_((float    *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_FP32_((float    *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_FP32_((float    *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_FP64_((double   *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_FP64_((double   *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_FP64_((double   *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_UDT_((void     *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_UDT_((void     *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_UDT_((void     *) NULL, op0, m2, v , d0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    m0 = monoid_crud ;
    v0 = vcrud ;
    A0 = Acrud ;
    op0 = op2crud ;

    // matrix to scalar

    ERR (GrB_Matrix_reduce_BOOL_((bool     *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_BOOL_((bool     *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_BOOL_((bool     *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_INT8_((int8_t   *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_INT8_((int8_t   *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_INT8_((int8_t   *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_UINT8_((uint8_t  *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_UINT8_((uint8_t  *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_UINT8_((uint8_t  *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_INT16_((int16_t  *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_INT16_((int16_t  *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_INT16_((int16_t  *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_UINT16_((uint16_t *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_UINT16_((uint16_t *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_UINT16_((uint16_t *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_INT32_((int32_t  *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_INT32_((int32_t  *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_INT32_((int32_t  *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_UINT32_((uint32_t *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_UINT32_((uint32_t *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_UINT32_((uint32_t *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_INT64_((int64_t  *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_INT64_((int64_t  *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_INT64_((int64_t  *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_UINT64_((uint64_t *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_UINT64_((uint64_t *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_UINT64_((uint64_t *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_FP32_((float    *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_FP32_((float    *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_FP32_((float    *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_FP64_((double   *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_FP64_((double   *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_FP64_((double   *) NULL, op0, m2, A , d0)) ;

    ERR (GrB_Matrix_reduce_UDT_((void     *) NULL, op0, m0, A0, d0)) ;
    ERR (GrB_Matrix_reduce_UDT_((void     *) NULL, op0, m0, A , d0)) ;
    ERR (GrB_Matrix_reduce_UDT_((void     *) NULL, op0, m2, A , d0)) ;

    // vector to scalar

    ERR (GrB_Vector_reduce_BOOL_((bool     *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_BOOL_((bool     *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_BOOL_((bool     *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_INT8 ((int8_t   *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_INT8 ((int8_t   *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_INT8_((int8_t   *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_UINT8_((uint8_t  *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_UINT8_((uint8_t  *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_UINT8_((uint8_t  *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_INT16_((int16_t  *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_INT16_((int16_t  *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_INT16_((int16_t  *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_UINT16_((uint16_t *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_UINT16_((uint16_t *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_UINT16_((uint16_t *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_INT32_((int32_t  *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_INT32_((int32_t  *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_INT32_((int32_t  *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_UINT32_((uint32_t *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_UINT32_((uint32_t *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_UINT32_((uint32_t *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_INT64_((int64_t  *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_INT64_((int64_t  *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_INT64_((int64_t  *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_UINT64_((uint64_t *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_UINT64_((uint64_t *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_UINT64_((uint64_t *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_FP32_((float    *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_FP32_((float    *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_FP32_((float    *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_FP64_((double   *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_FP64_((double   *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_FP64_((double   *) NULL, op0, m2, v , d0)) ;

    ERR (GrB_Vector_reduce_UDT_((void     *) NULL, op0, m0, v0, d0)) ;
    ERR (GrB_Vector_reduce_UDT_((void     *) NULL, op0, m0, v , d0)) ;
    ERR (GrB_Vector_reduce_UDT_((void     *) NULL, op0, m2, v , d0)) ;

    m0 = NULL ;
    v0 = NULL ;
    A0 = NULL ;
    op0 = NULL ;
    expected = GrB_DOMAIN_MISMATCH ;
    o2  = Complex_plus ;

    if (Complex == GxB_FC64)
    {
        OK  (GrB_Matrix_reduce_FP64 (&x, op0 , GxB_PLUS_FP64_MONOID , Z , d0)) ;
        OK  (GrB_Matrix_reduce_FP64 (&x, op0 , Complex_plus_monoid  , Z , d0)) ;
        OK  (GrB_Matrix_reduce_FP64 (&x, op0 , Complex_plus_monoid  , A , d0)) ;
        OK  (GrB_Matrix_reduce_FP64 (&x, o2  , Complex_plus_monoid  , Z , d0)) ;
        OK  (GrB_Matrix_reduce_UDT  (&c, o2  , Complex_plus_monoid  , A , d0)) ;
    }
    else
    {
        ERR (GrB_Matrix_reduce_FP64 (&x, op0 , GxB_PLUS_FP64_MONOID , Z , d0)) ;
        ERR (GrB_Matrix_reduce_FP64 (&x, op0 , Complex_plus_monoid  , Z , d0)) ;
        ERR (GrB_Matrix_reduce_FP64 (&x, op0 , Complex_plus_monoid  , A , d0)) ;
        ERR (GrB_Matrix_reduce_FP64 (&x, o2  , Complex_plus_monoid  , Z , d0)) ;
        ERR (GrB_Matrix_reduce_UDT  (&c, o2  , Complex_plus_monoid  , A , d0)) ;
    }

    //--------------------------------------------------------------------------
    // reduce to vector
    //--------------------------------------------------------------------------

    printf ("GrB_reduce (to vector)---------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    o2 = GrB_PLUS_FP64 ;
    m2 = GxB_TIMES_FP64_MONOID ;

    ERR1 (v0, GrB_Matrix_reduce_BinaryOp_(v0, NULL, NULL, op0, A0, d0)) ;    // reduce via op
    ERR1 (v0, GrB_Matrix_reduce_BinaryOp_(v0, NULL, NULL, o2 , A0, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_BinaryOp_(v , NULL, NULL, o2 , A0, d0)) ;

    ERR1 (v0, GrB_Matrix_reduce_Monoid_(v0, NULL, NULL, m0 , A0, d0)) ;    // reduce via monoid
    ERR1 (v0, GrB_Matrix_reduce_Monoid_(v0, NULL, NULL, m2 , A0, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_Monoid_(v , NULL, NULL, m2 , A0, d0)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    m0 = monoid_crud ;
    v0 = vcrud ;
    A0 = Acrud ;
    op0 = op2crud ;
    d0 = dcrud ;

    ERR1 (v0, GrB_Matrix_reduce_BinaryOp_(v0, v0  , op0 , op0, A0, d0)) ;    // reduce via op
    ERR1 (v0, GrB_Matrix_reduce_BinaryOp_(v0, v0  , op0 , o2 , A0, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_BinaryOp_(v , v0  , op0 , o2 , A0, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_BinaryOp_(v , v   , op0 , o2 , A0, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_BinaryOp_(v , v   , o2  , o2 , A0, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_BinaryOp_(v , v   , o2  , o2 , A , d0)) ;

    ERR1 (v0, GrB_Matrix_reduce_Monoid_(v0, v0  , op0 , m0 , A0, d0)) ;    // reduce via monoid
    ERR1 (v0, GrB_Matrix_reduce_Monoid_(v0, v0  , op0 , m2 , A0, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_Monoid_(v , v0  , op0 , m2 , A0, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_Monoid_(v , v   , op0 , m2 , A0, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_Monoid_(v , v   , o2  , m2 , A0, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_Monoid_(v , v   , o2  , m2 , A , d0)) ;

    m0 = NULL ;
    v0 = NULL ;
    A0 = NULL ;
    op0 = NULL ;
    d0 = NULL ;

    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;

    o2 = Complex_plus ;
    ERR1 (v,  GrB_Matrix_reduce_BinaryOp_(v, z   , NULL, GrB_PLUS_FP64, A, d0)) ;
    ERR1 (z,  GrB_Matrix_reduce_BinaryOp_(z, NULL, NULL, GrB_PLUS_FP64, A, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_BinaryOp_(v, NULL, o2  , GrB_PLUS_FP64, A, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_BinaryOp_(v, NULL, NULL, GrB_PLUS_FP64, Z, d0)) ;

    ERR1 (v,  GrB_Matrix_reduce_Monoid_(v, z   , NULL, GxB_PLUS_FP64_MONOID, A, d0)) ;
    ERR1 (z,  GrB_Matrix_reduce_Monoid_(z, NULL, NULL, GxB_PLUS_FP64_MONOID, A, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_Monoid_(v, NULL, o2  , GxB_PLUS_FP64_MONOID, A, d0)) ;
    ERR1 (v,  GrB_Matrix_reduce_Monoid_(v, NULL, NULL, GxB_PLUS_FP64_MONOID, Z, d0)) ;

    expected = GrB_DIMENSION_MISMATCH ;

    ERR1 (v,  GrB_Matrix_reduce_Monoid_(v, NULL, NULL, GxB_PLUS_FP64_MONOID, A, dtn)) ;
    ERR1 (v,  GrB_Matrix_reduce_Monoid_(v, NULL, NULL, GxB_PLUS_FP64_MONOID, A, d0)) ;

    //--------------------------------------------------------------------------
    // transpose
    //--------------------------------------------------------------------------

    printf ("GrB_transpose------------------------------------------------\n") ;
    expected = GrB_NULL_POINTER ;

    ERR (GrB_transpose (NULL, NULL, NULL, NULL, NULL)) ;
    ERR1 (A, GrB_transpose (A   , NULL, NULL, NULL, NULL)) ;

    expected = GrB_NULL_POINTER ;
    ERR1 (Acrud, GrB_transpose (Acrud, NULL , NULL   , NULL , NULL )) ;
    ERR1 (A,  GrB_transpose (A    , Acrud, NULL   , NULL , NULL )) ;
    ERR1 (A,  GrB_transpose (A    , A    , op2crud, NULL , NULL )) ;

    expected = GrB_UNINITIALIZED_OBJECT ;
    ERR1 (A,  GrB_transpose (A    , NULL , NULL   , Acrud, NULL )) ;
    ERR1 (A,  GrB_transpose (A    , NULL , NULL   , A    , dcrud)) ;

    expected = (Complex == GxB_FC64) ? GrB_DIMENSION_MISMATCH : GrB_DOMAIN_MISMATCH ;

    o2 = Complex_plus ;
    ERR1 (A,  GrB_transpose (A   , Z   , NULL, A, NULL)) ;
    ERR1 (A,  GrB_transpose (A   , NULL, NULL, Z, NULL)) ;
    ERR1 (A,  GrB_transpose (A   , NULL, NULL, Z, NULL)) ;
    ERR1 (Z,  GrB_transpose (Z   , NULL, NULL, A, NULL)) ;
    ERR1 (A,  GrB_transpose (A   , NULL, o2  , A, NULL)) ;
    ERR1 (A,  GrB_transpose (A   , NULL, o2  , Z, NULL)) ;
    ERR1 (Z,  GrB_transpose (Z   , NULL, o2  , A, NULL)) ;

    expected = GrB_DIMENSION_MISMATCH ;

    ERR1 (A,  GrB_transpose (A   , NULL, NULL, A, NULL)) ;
    ERR1 (C,  GrB_transpose (C   , NULL, NULL, A, dtn )) ;

    //==========================================================================
    //=== internal functions ===================================================
    //==========================================================================

    //--------------------------------------------------------------------------
    // Entry print
    //--------------------------------------------------------------------------

    printf ("\n-------------- GB_entry_check:\n") ;

    Werk->where = "GB_entry_check (type, x, pr, f, &string, &string_size)" ;

    expected = GrB_NULL_POINTER ;

    ERR (GB_entry_check (NULL, NULL, 5, NULL, NULL, NULL)) ;
    ERR (GB_entry_check (NULL, X, 5, NULL, NULL, NULL)) ;
    OK (GB_entry_check (GrB_FP64, X, 5, NULL, NULL, NULL)) ;
    printf ("\n") ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GB_entry_check (Tcrud, X, 5, NULL, NULL, NULL)) ;
    printf ("\nAll GB_entry_check tests passed (errors expected)\n") ;

    //--------------------------------------------------------------------------
    // Type check
    //--------------------------------------------------------------------------

    printf ("\n-------------- GB_Type_check:\n") ;

    Werk->where = "GB_Type_check" ;

    info = GB_Type_check (NULL, "null type", G1, ff) ;
    CHECK (info == GrB_NULL_POINTER) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    ERR (GB_Type_check (Tcrud, "Tcrud", G1, ff)) ;

    CHECK (T == NULL) ;
    OK (GrB_Type_new (&T, sizeof (int))) ;

    Werk->where = "GB_Type_check" ;
    OK (GB_Type_check (T, "T ok (via function)", G3, ff)) ;

    T->magic = GB_FREED ;
    ERR (GB_Type_check (T, "T freed", G1, ff)) ;
    T->magic = GB_MAGIC ;

    expected = GrB_INVALID_OBJECT ;

    T->code = 99 ;
    ERR (GB_Type_check (T, "T bad code", G1, ff)) ;
    T->code = GB_UDT_code ;
    T->magic = GB_MAGIC ;
    T->size = 0 ;
    ERR (GB_Type_check (T, "T bad size", G1, ff)) ;
    T->size = sizeof (int) ;

    const char *e = GB_code_string (9999) ;
    printf ("unknown code: [%s]\n", e) ;
    CHECK (strcmp (e, "unknown type!") == 0) ;

    e = GB_status_code (9999) ;
    printf ("unknown info: [%s]\n", e) ;
    CHECK (strcmp (e, "unknown GrB_Info value!") == 0) ;

    OK (GB_Type_check (T, "type ok", G1, ff)) ;
    printf ("\nAll GB_Type_check tests passed (errors expected)\n") ;

    //--------------------------------------------------------------------------
    // UnaryOp check
    //--------------------------------------------------------------------------

    printf ("\n-------------- GB_UnaryOp_check:\n") ;

    Werk->where = "GB_UnaryOp_check" ;

    info = GB_UnaryOp_check (NULL, "null unary op", G3, ff) ;
    CHECK (info == GrB_NULL_POINTER) ;

    CHECK (op1b == NULL) ;
    OK (GrB_UnaryOp_new (&op1b, (GxB_unary_function) f1, GrB_FP64, GrB_UINT32)) ;
    CHECK (op1b != NULL) ;
    OK (GrB_UnaryOp_wait_(op1b, GrB_MATERIALIZE)) ;

    Werk->where = "GB_UnaryOp_check" ;
    OK (GB_UnaryOp_check (op1b, "op1b ok (via function)", G3, ff)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    op1b->magic = GB_FREED ;
    ERR (GB_UnaryOp_check (op1b, "op1b freed", G1, ff)) ;
    op1b->magic = GB_MAGIC ;

    expected = GrB_INVALID_OBJECT ;

    op1b->unop_function = NULL ;
    ERR (GB_UnaryOp_check (op1b, "op1b null func", G1, ff)) ;
    op1b->unop_function = (GxB_unary_function) f1 ;

    op1b->opcode = 1024 ;
    ERR (GB_UnaryOp_check (op1b, "op1b invalid opcode", G1, ff)) ;
    op1b->opcode = GB_USER_unop_code ;

    op1b->ztype = NULL ;
    ERR (GB_UnaryOp_check (op1b, "op1b invalid ztype", G1, ff)) ;
    op1b->ztype = GrB_FP64 ;

    op1b->xtype = NULL ;
    ERR (GB_UnaryOp_check (op1b, "op1b invalid xtype", G1, ff)) ;
    op1b->xtype = GrB_UINT32 ;

    printf ("\nAll GB_UnaryOp_check tests passed (errors expected)\n") ;

    //--------------------------------------------------------------------------
    // BinaryOp check
    //--------------------------------------------------------------------------

    printf ("\n-------------- GB_BinaryOp_check:\n") ;

    Werk->where = "GB_BinaryOp_check" ;

    info = GB_BinaryOp_check (NULL, "null unary op", G3, ff) ;
    CHECK (info == GrB_NULL_POINTER) ;

    CHECK (op2b == NULL) ;
    OK (GrB_BinaryOp_new (&op2b, (GxB_binary_function) f2, GrB_INT32, GrB_UINT8, GrB_INT16)) ;
    CHECK (op2b != NULL) ;

    Werk->where = "GB_BinaryOp_check" ;
    OK (GB_BinaryOp_check (op2b, "op2b ok (via function)", G3, ff)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    op2b->magic = GB_FREED ;
    ERR (GB_BinaryOp_check (op2b, "op2b freed", G1, ff)) ;
    op2b->magic = GB_MAGIC ;

    expected = GrB_INVALID_OBJECT ;

    op2b->binop_function = NULL ;
    ERR (GB_BinaryOp_check (op2b, "op2b null func", G1, ff)) ;
    op2b->binop_function = (GxB_binary_function) f2 ;

    op2b->opcode = 1024 ;
    ERR (GB_BinaryOp_check (op2b, "op2b invalid opcode", G1, ff)) ;
    op2b->opcode = GB_USER_binop_code ;

    op2b->ztype = NULL ;
    ERR (GB_BinaryOp_check (op2b, "op2b invalid ztype", G1, ff)) ;
    op2b->ztype = GrB_INT32 ;

    op2b->xtype = NULL ;
    ERR (GB_BinaryOp_check (op2b, "op2b invalid xtype", G1, ff)) ;
    op2b->xtype = GrB_UINT8 ;

    op2b->ytype = NULL ;
    ERR (GB_BinaryOp_check (op2b, "op2b invalid ytype", G1, ff)) ;
    op2b->ytype = GrB_UINT16 ;

    printf ("\nAll GB_BinaryOp_check tests passed (errors expected)\n") ;

    //--------------------------------------------------------------------------
    // Monoid check
    //--------------------------------------------------------------------------

    printf ("\n-------------- GB_Monoid_check:\n") ;

    Werk->where = "GB_Monoid_check" ;

    info = GB_Monoid_check (NULL, "null monoid", G3, ff, false) ;
    CHECK (info == GrB_NULL_POINTER) ;

    CHECK (monoidb == NULL) ;
    OK (GrB_Monoid_new_INT32 (&monoidb, GrB_TIMES_INT32, (int) 1)) ;
    CHECK (monoidb != NULL) ;

    Werk->where = "GB_Monoid_check" ;
    OK (GB_Monoid_check (monoidb, "monoidb ok", G3, ff, false)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    monoidb->magic = GB_FREED ;
    ERR (GB_Monoid_check (monoidb, "monoidb freed", G1, ff, false)) ;
    monoidb->magic = GB_MAGIC ;

    expected = GrB_INVALID_OBJECT ;

    monoidb->op = NULL ;
    ERR (GB_Monoid_check (monoidb, "monoidb invalid op", G1, ff, false)) ;
    monoidb->op = GrB_TIMES_INT32 ;

    monoidb->op = GrB_EQ_INT32 ;
    ERR (GB_Monoid_check (monoidb, "monoidb invalid op domains", G1, ff,
        false)) ;
    monoidb->op = GrB_TIMES_INT32 ;

    OK (GB_Monoid_check (Complex_plus_monoid, "complex plus monoid", G3, ff,
        false)) ;
    OK (GB_Monoid_check (Complex_times_monoid, "complex times monoid", G3, ff,
        false)) ;

    printf ("\nAll GB_Monoid_check tests passed (errors expected)\n") ;

    //--------------------------------------------------------------------------
    // Semiring check
    //--------------------------------------------------------------------------

    printf ("\n-------------- GB_Semiring_check:\n") ;

    Werk->where = "GB_Semiring_check" ;

    info = GB_Semiring_check (NULL, "null semiring", G3, ff) ;
    CHECK (info == GrB_NULL_POINTER) ;

    CHECK (semiring2 == NULL) ;
    OK (GrB_Semiring_new (&semiring2, GxB_MAX_FP32_MONOID, GrB_TIMES_FP32)) ;
    CHECK (semiring2 != NULL) ;

    Werk->where = "GB_Semiring_check" ;
    OK (GB_Semiring_check (semiring2, "semiring2 ok", G3, ff)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    semiring2->magic = GB_FREED ;
    ERR (GB_Semiring_check (semiring2, "semiring2 freed", G1, ff)) ;
    semiring2->magic = GB_MAGIC ;

    expected = GrB_INVALID_OBJECT ;

    semiring2->add = NULL ;
    ERR (GB_Semiring_check (semiring2, "semiring2 invalid add monoid", G1, ff)) ;
    semiring2->add = GxB_MAX_FP32_MONOID ;

    semiring2->multiply = NULL ;
    ERR (GB_Semiring_check (semiring2, "semiring2 invalid mult", G1, ff)) ;
    semiring2->multiply = GrB_TIMES_FP32 ;

    semiring2->multiply = GrB_TIMES_INT32 ;
    ERR (GB_Semiring_check (semiring2, "semiring2 invalid mix", G1, ff)) ;
    semiring2->multiply = GrB_TIMES_FP32 ;

    printf ("\nAll GB_Semiring_check tests passed (errors expected)\n") ;

    //--------------------------------------------------------------------------
    // Descriptor check
    //--------------------------------------------------------------------------

    printf ("\n-------------- GB_Descriptor_check:\n") ;

    Werk->where = "GB_Descriptor_check" ;

    info = GB_Descriptor_check (NULL, "null descriptor", G3, ff) ;
    CHECK (info == GrB_NULL_POINTER) ;

    CHECK (descb == NULL) ;
    OK (GrB_Descriptor_new (&descb)) ;
    CHECK (descb != NULL) ;

    Werk->where = "GB_Descriptor_check" ;
    OK (GB_Descriptor_check (descb, "descb ok", G3, ff)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    descb->magic = GB_FREED ;
    ERR (GB_Descriptor_check (descb, "descb freed", G1, ff)) ;
    descb->magic = GB_MAGIC ;

    expected = GrB_INVALID_OBJECT ;

    descb->out = 42 ;
    ERR (GB_Descriptor_check (descb, "descb invalid", G1, ff)) ;
    descb->out = GxB_DEFAULT ;

    printf ("\nAll GB_Descriptor_check tests passed (errors expected)\n") ;

    //--------------------------------------------------------------------------
    // Vector check
    //--------------------------------------------------------------------------

    printf ("\n-------------- GB_Vector_check:\n") ;
    OK (GxB_Vector_fprint_(v, G3, ff)) ;

    OK (GrB_Vector_free_(&v)) ;
    CHECK (v == NULL) ;

    Werk->where = "GB_Vector_check" ;

    info = GB_Vector_check (NULL, "null vector", G3, ff) ;
    CHECK (info == GrB_NULL_POINTER) ;

    CHECK (v == NULL) ;
    OK (GrB_Vector_new (&v, GrB_FP64, 10)) ;
    CHECK (v != NULL) ;
    CHECK (v->h == NULL) ;

    Werk->where = "GB_Vector_check" ;
    OK (GB_Vector_check (v, "v ok", G3, ff)) ;

    OK (GrB_Vector_setElement_INT32 (v, 990, 0)) ;
    OK (GrB_Vector_setElement_INT32 (v, 991, 1)) ;
    OK (GrB_Vector_nvals (&nvals, v)) ;
    OK (GrB_Vector_wait_(v, GrB_MATERIALIZE)) ;
    CHECK (nvals == 2) ;
    OK (GxB_Vector_fprint (v, "v ok (might be bitmap)", G3, ff)) ;
    OK (GxB_Vector_Option_set (v, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GxB_Vector_fprint (v, "v ok (sparse)", G3, ff)) ;

    expected = GrB_INVALID_OBJECT ;
    CHECK (!GB_IS_FULL (v)) ;
    GB_MDECL (Vi, , ) ;
    Vi = v->i ;
    GB_IPTR (Vi, v->i_is_32) ;
    GB_ISET (Vi, 0, 1) ;        // Vi [0] = 1 ;
    GB_ISET (Vi, 1, 0) ;        // Vi [1] = 0 ;
    ERR (GxB_Vector_fprint (v, "v jumbled", G3, ff)) ;
    GB_ISET (Vi, 0, 0) ;        // Vi [0] = 0 ;
    GB_ISET (Vi, 1, 1) ;        // Vi [1] = 1 ;
    OK (GxB_Vector_fprint (v, "v fixed", G3, ff)) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    v->magic = GB_FREED ;
    ERR (GB_Vector_check (v, "v freed", G1, ff)) ;
    v->magic = GB_MAGIC ;

    expected = GrB_INVALID_OBJECT ;

    v->vdim = 2 ;
    void *psave = v->p ;
    size_t p_size_save = v->p_size ;
    printf ("test0, remove v->p from memtable %d\n", v->p) ;
    GB_Global_memtable_remove (v->p) ;
    v->p_size = 3 * sizeof (int64_t) ;
    v->p = mxMalloc (v->p_size) ;
    memset (v->p, 0, v->p_size) ;
    printf ("test0, add v->p to memtable %d\n", v->p) ;
    GB_Global_memtable_add (v->p, v->p_size) ;
    ERR (GB_Vector_check (v, "v invalid", G1, ff)) ;
    v->vdim = 1 ;

    CHECK (!GB_IS_FULL (v)) ;
    GB_MDECL (Vp, , ) ;
    Vp = v->p ;
    GB_IPTR (Vp, v->p_is_32) ;
    GB_ISET (Vp, 0, 1) ;        // Vp [0] = 1 ;
    ERR (GB_Vector_check (v, "v p[0] invalid", G1, ff)) ;

    mxFree (v->p) ;
    printf ("test0, remove v->p from memtable %d\n", v->p) ;
    GB_Global_memtable_remove (v->p) ;
    v->p = psave ;
    v->p_size = p_size_save ;
    printf ("test0, add v->p to memtable %d\n", v->p) ;
    GB_Global_memtable_add (v->p, v->p_size) ;
    psave = NULL ;
    OK (GB_Vector_check (v, "v OK now", G1, ff)) ;

    printf ("\nAll GB_Vector_check tests passed (errors expected)\n") ;

    //--------------------------------------------------------------------------
    // Matrix check
    //--------------------------------------------------------------------------

    printf ("\n-------------- GB_Matrix_check:\n") ;

    OK (GrB_Matrix_free_(&A)) ;
    CHECK (A == NULL) ;

    Werk->where = "GB_Matrix_check" ;

    info = GB_Matrix_check (NULL, "null matrix", G3, ff) ;
    CHECK (info == GrB_NULL_POINTER) ;

    CHECK (A == NULL) ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 4)) ;
    CHECK (A != NULL) ;

    Werk->where = "GB_Matrix_check" ;
    OK (GB_Matrix_check (A, "A ok", G3, ff)) ;
    CHECK (A->h != NULL) ;

    CHECK (A->i == NULL) ;

    expected = GrB_UNINITIALIZED_OBJECT ;

    A->magic = GB_FREED ;
    ERR (GB_Matrix_check (A, "A freed", G1, ff)) ;
    A->magic = GB_MAGIC ;

    expected = GrB_INVALID_OBJECT ;

    CHECK (!GB_IS_FULL (A)) ;
    GB_MDECL (Ap, , ) ;
    Ap = A->p ;
    GB_IPTR (Ap, A->p_is_32) ;
    GB_ISET (Ap, 0, 1) ;        // Ap [0] = 1 ;
    ERR (GB_Matrix_check (A, "p[0] invalid", G1, ff)) ;
    GB_ISET (Ap, 0, 0) ;        // Ap [0] = 0 ;

    A->vlen = -1 ;
    ERR (GB_Matrix_check (A, "invalid dimensions", G1, ff)) ;
    A->vlen = 10 ;

    A->type = NULL ;
    ERR (GB_Matrix_check (A, "invalid type", G1, ff)) ;
    A->type = GrB_FP64 ;

    psave = A->p ;
    A->p = NULL ;
    ERR (GB_Matrix_check (A, "NULL Ap", G1, ff)) ;
    A->p = psave ;

    CHECK (A->i == NULL) ;

    OK (GrB_Matrix_free_(&A)) ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 4)) ;

    uint64_t I00 [1] = { 0 } ;     // OK
    uint64_t J00 [1] = { 0 } ;     // OK
    OK (GrB_Matrix_setElement_FP64 (A, 3.14159, 0, 0)) ;
    OK (GrB_Matrix_assign_BOOL (A, NULL, GrB_SECOND_FP64, true, I00, 1, J00, 1,
        NULL)) ;
    OK (GB_Matrix_check (A, "valid pending pi", G3, NULL)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    OK (GB_Matrix_check (A, "valid pending pi again", G3, NULL)) ;
    printf ("nvals %lu\n", nvals) ;
    OK (GrB_Matrix_wait_(A, GrB_MATERIALIZE)) ;
    CHECK (nvals == 1) ;

    printf ("\n========================================== valid pi\n") ;
    OK (GB_Matrix_check (A, "valid pi", G3, NULL)) ;
    printf ("\n===================================================\n") ;

    OK (GrB_Matrix_free_(&A)) ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 4)) ;
    OK (GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GB_Matrix_check (A, "A empty", G3, NULL)) ;

    // change the type of the pending tuples, forcing a wait
    OK (GrB_Matrix_assign_BOOL (A, NULL, GrB_SECOND_FP64, (bool) true,
        I00, 1, J00, 1, NULL)) ;
    OK (GB_Matrix_check (A, "with bool pending", G3, NULL)) ;

    OK (GrB_Matrix_setElement_FP64 (A, 3.14159, 3, 3)) ;
    OK (GB_Matrix_check (A, "with pi pending", G3, NULL)) ;

    A_Pending = A->Pending ;
    CHECK (A_Pending != NULL) ;
    CHECK (A_Pending->n == 1) ;
    CHECK (A_Pending->type == GrB_FP64) ;

    OK (GrB_Matrix_setElement_FP64 (A, 9.0909, 2, 1)) ;

    A_Pending = A->Pending ;
    CHECK (A_Pending != NULL) ;
    CHECK (A_Pending->n == 2) ;
    CHECK (A_Pending->type == GrB_FP64) ;

    OK (GB_Matrix_check (A, "with pi and 9.0909 pending", G3, NULL)) ;

    OK (GrB_Matrix_nvals (&nvals, A)) ;
    OK (GrB_Matrix_wait_(A, GrB_MATERIALIZE)) ;
    CHECK (nvals == 3) ;

    Werk->where = "GB_Matrix_check" ;

    psave = A->i ;
    A->i = NULL ;
    ERR (GB_Matrix_check (A, "NULL Ai", G1, NULL)) ;
    A->i = psave ;
    OK (GB_Matrix_check (A, "valid pi", G0, NULL)) ;

    CHECK (!GB_IS_FULL (A)) ;
    Ap = A->p ;
    GB_IPTR (Ap, A->p_is_32) ;
    GB_ISET (Ap, 0, 1) ;        // Ap [0] = 1
    ERR (GB_Matrix_check (A, "Ap[0] invalid", G1, NULL)) ;
    GB_ISET (Ap, 0, 0) ;        // Ap [0] = 0

    int64_t isave = GB_IGET (Ap, 1) ;
    GB_ISET (Ap, 1, -1) ;       // Ap [1] = -1 ;
    ERR (GB_Matrix_check (A, "Ap[1] invalid", G1, NULL)) ;
    GB_ISET (Ap, 1, isave) ;    // Ap [1] = isave ;

    isave = GB_IGET (Ap, 4) ;
    GB_ISET (Ap, 4, 999) ;      // Ap [4] += 999 ;
    ERR (GB_Matrix_check (A, "Ap[ncols] invalid", G1, NULL)) ;
    GB_ISET (Ap, 4, isave) ;    // Ap [4] = isave ;

    isave = A->nzombies ;
    A->nzombies = -1 ;
    ERR (GB_Matrix_check (A, "negative zombies", G1, NULL)) ;
    A->nzombies = isave ;

    isave = A->nzombies ;
    A->nzombies = 1000 ;
    ERR (GB_Matrix_check (A, "too many zombies", G1, NULL)) ;
    A->nzombies = isave ;

    Ai = A->i ;
    GB_IPTR (Ai, A->i_is_32) ;
    isave = GB_IGET (Ai, 0) ;
    GB_ISET (Ai, 0, -1) ;       // Ai [0] = -1
    ERR (GB_Matrix_check (A, "row index invalid", G3, NULL)) ;
    GB_ISET (Ai, 0, isave) ;    // Ai [0] = isave ;

    isave = A->nzombies ;
    A->nzombies = 1 ;
    ERR (GB_Matrix_check (A, "bad zombies", G3, NULL)) ;
    A->nzombies = isave ;

    A_Pending = A->Pending ;
    CHECK (A_Pending == NULL) ;
    OK (GB_Matrix_check (A, "so far", G3, NULL)) ;

    printf ("\n========================================== valid [pi 7.1]\n") ;
    OK (GrB_Matrix_setElement_FP64 (A, 7.1, 1, 0)) ;
    OK (GB_Matrix_check (A, "valid pending [pi 7.1]", G3, NULL)) ;
    printf ("\n===================================================\n") ;

    Werk->where = "GB_Matrix_check" ;

    A_Pending = A->Pending ;
    CHECK (A_Pending != NULL) ;
    isave = A_Pending->n ;
    A_Pending->n = -1 ;
    ERR (GB_Matrix_check (A, "negative pending", G1, NULL)) ;
    A_Pending->n = isave ;

    A_Pending = A->Pending ;
    CHECK (A_Pending != NULL) ;
    psave = A_Pending->i ;
    A_Pending->i = NULL ;
    ERR (GB_Matrix_check (A, "missing pending", G3, NULL)) ;
    A_Pending->i = psave ;

    OK (GB_Matrix_check (A, "valid pending [pi 7.1]", G0, NULL)) ;

    A_Pending = A->Pending ;
    CHECK (A_Pending != NULL) ;
    GB_MDECL (Pending_j, , u) ;
    GB_GET_PENDINGj_PTR (Pending_j, A) ;
    CHECK (Pending_j != NULL) ;
    isave = GB_IGET (Pending_j, 0) ;
    GB_ISET (Pending_j, 0, 1070) ;
    ERR (GB_Matrix_check (A, "bad pending tuple", G3, NULL)) ;
    GB_ISET (Pending_j, 0, isave) ;
    OK (GB_Matrix_check (A, "valid pending [pi 7.1]", G0, NULL)) ;

    printf ("\n====================================== valid [pi 7.1 11.4]\n") ;
    OK (GrB_Matrix_setElement_FP64 (A, 11.4, 0, 1)) ;
    OK (GB_Matrix_check (A, "valid pending [pi 7.1 11.4]", G3, NULL)) ;
    printf ("\n=========================================================\n") ;

    Werk->where = "GB_Matrix_check" ;

    A_Pending = A->Pending ;
    CHECK (A_Pending != NULL) ;
    A_Pending->sorted = !(A_Pending->sorted) ;
    printf ("matrix check with jumbled pending tuples:\n") ;
    ERR (GxB_Matrix_fprint (A, "jumbled pending tuples", G3, ff)) ;
    ERR (GB_Matrix_check (A, "jumbled pending tuples", G3, ff)) ;
    A_Pending->sorted = !(A_Pending->sorted) ;
    OK (GB_Matrix_check (A, "valid pending [pi 7.1 11.4]", G0, ff)) ;

    A_Pending = A->Pending ;
    CHECK (A_Pending != NULL) ;
    CHECK (A_Pending->op == NULL) ;
    A_Pending->op = op2crud ;
    ERR (GB_Matrix_check (A, "invalid operator", G3, NULL)) ;
    A_Pending->op = NULL ;

    OK (GB_Matrix_check (A, "valid pending [pi 7.1 11.4]", G3, NULL)) ;
    printf ("\n=========================================================\n") ;

    printf ("\n###### get nvals; assemble the pending tuples ##### \n") ;

    OK (GrB_Matrix_nvals (&nvals, A)) ;
    printf ("nvals: %lu\n", nvals) ;
    OK (GrB_Matrix_wait_(A, GrB_MATERIALIZE)) ;

    Werk->where = "GB_Matrix_check" ;
    printf ("\n====================================== valid [pi 7.1 11.4]\n") ;
    OK (GB_Matrix_check (A, "valid [pi 7 11.4]", G3, NULL)) ;
    printf ("\n=========================================================\n") ;
    CHECK (nvals == 5) ;

    expected = GrB_INDEX_OUT_OF_BOUNDS ;

    CHECK (!GB_IS_FULL (A)) ;
    CHECK (!GB_IS_BITMAP (A)) ;
    Ai = A->i ;
    GB_IPTR (Ai, A->i_is_32) ;
    GB_ISET (Ai, 0, 1) ;    // Ai [0] = 1 ;
    GB_ISET (Ai, 1, 0) ;    // Ai [1] = 0 ;

    info = GB_Matrix_check (A, "jumbled", G3, NULL) ;
    printf ("jumbled info %d\n", info) ;
    CHECK (info == GrB_INDEX_OUT_OF_BOUNDS) ;

    info = GxB_Matrix_fprint (A, "jumbled", G3, ff) ;
    printf ("jumbled info %d\n", info) ;
    CHECK (info == GrB_INVALID_OBJECT) ;

    CHECK (!GB_IS_FULL (A)) ;
    CHECK (!GB_IS_BITMAP (A)) ;
    Ai = A->i ;
    GB_IPTR (Ai, A->i_is_32) ;
    GB_ISET (Ai, 0, 0) ;    // Ai [0] = 0 ;
    GB_ISET (Ai, 1, 1) ;    // Ai [1] = 1 ;
    OK (GB_Matrix_check (A, "OK", G3, NULL)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    OK (GrB_Matrix_wait_(A, GrB_MATERIALIZE)) ;
    CHECK (nvals == 5) ;

    A_Pending = A->Pending ;
    CHECK (A_Pending == NULL) ;
    CHECK (A->nzombies == 0) ;
    OK (GrB_Matrix_new (&Empty1, GrB_FP64, 1, 1)) ;
    I [0] = 0 ;
    J [0] = 0 ;
    OK (GxB_Matrix_subassign_(A, NULL, NULL, Empty1, I, 1, J, 1, NULL)) ;
    OK (GB_Matrix_check (A, "valid zombie", G3, NULL)) ;
    OK (GrB_Matrix_setElement_INT32 (A, 99099, 0, 0)) ;
    OK (GB_Matrix_check (A, "no more zombie", G3, NULL)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    OK (GrB_Matrix_wait_(A, GrB_MATERIALIZE)) ;
    CHECK (nvals == 5) ;

    OK (GxB_Matrix_subassign_(A, NULL, NULL, Empty1, I, 1, J, 1, NULL)) ;
    OK (GB_Matrix_check (A, "valid zombie", G3, NULL)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    OK (GrB_Matrix_wait_(A, GrB_MATERIALIZE)) ;
    CHECK (nvals == 4) ;
    OK (GB_Matrix_check (A, "again no more zombie", G3, NULL)) ;
    OK (A->Pending == NULL && A->nzombies == 0) ;

    expected = GrB_INVALID_OBJECT ;

    OK (GB_Matrix_check (A, "valid, no pending", G3, NULL)) ;

    // #define FREE_DEEP_COPY ;
    // #define GET_DEEP_COPY ;

    CHECK (GB_IS_SPARSE (A)) ;
    OK (GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
    CHECK (GB_IS_HYPERSPARSE (A)) ;
    OK (GB_Matrix_check (A, "A now hyper", G3, NULL)) ;
    CHECK (A->h != NULL) ;

    CHECK (GB_IS_HYPERSPARSE (A)) ;
    OK (GxB_Matrix_Option_set_INT32 (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    CHECK (GB_IS_SPARSE (A)) ;
    OK (GB_Matrix_check (A, "A now sparse", G3, NULL)) ;
    CHECK (A->h == NULL) ;

    CHECK (GB_IS_SPARSE (A)) ;
    OK (GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
    CHECK (GB_IS_HYPERSPARSE (A)) ;
    OK (GB_Matrix_check (A, "A now hyper", G3, NULL)) ;
    CHECK (A->h != NULL) ;

    OK (GxB_Matrix_Option_set_(A, GxB_HYPER_SWITCH, GxB_NEVER_HYPER)) ;
    OK (GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    CHECK (A->h == NULL) ;
    bool A_is_hyper ;
    OK (GxB_Matrix_Option_get_(A, GxB_IS_HYPER, &A_is_hyper)) ; // historical
    CHECK (!A_is_hyper) ;
    int32_t A_is_hyperw ;
    OK (GxB_Matrix_Option_get_(A, GxB_IS_HYPER, &A_is_hyper)) ; // historical

    OK (GxB_Matrix_Option_set_(A, GxB_HYPER_SWITCH, GxB_ALWAYS_HYPER)) ;
    OK (GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
    CHECK (A->h != NULL) ;
    OK (GxB_Matrix_Option_get_(A, GxB_IS_HYPER, &A_is_hyper)) ; // historical
    CHECK (A_is_hyper) ;

    // make sure A->nvec_nonempty is valid
    CHECK (GB_IS_HYPERSPARSE (A)) ;
    GB_nvec_nonempty_update (A) ;

    // now make invalid.  GB_Matrix_check requires it to be -1, or correct value
    CHECK (GB_IS_HYPERSPARSE (A)) ;

    Ap = A->p ;
    GB_IPTR (Ap, A->p_is_32) ;
    isave = GB_IGET (Ap, 1) ;
    GB_ISET (Ap, 1, 0) ;        // Ap [1] = 0 ;
    expected = GrB_INDEX_OUT_OF_BOUNDS ;
    ERR (GB_Matrix_check (A, "A with bad nvec_nonempty", G1, NULL)) ;
    expected = GrB_INVALID_OBJECT ;
    ERR (GxB_Matrix_fprint (A, "A", G1, ff)) ;
    CHECK (GB_IS_HYPERSPARSE (A)) ;
    GB_ISET (Ap, 1, isave) ;    // Ap [1] = isave ;
    OK (GB_Matrix_check (A, "A fixed", G0, NULL)) ;

    double hratio = 0.5;
    OK (GxB_Matrix_Option_set_(A, GxB_HYPER_SWITCH, hratio)) ;

    double hratio2 = 0 ;
    OK (GxB_Matrix_Option_get_(A, GxB_HYPER_SWITCH, &hratio2)) ;
    CHECK (hratio == hratio2) ;

    hratio = 0.75 ;
    OK (GxB_Matrix_Option_set_FP64 (A, GxB_HYPER_SWITCH, hratio)) ;

    hratio2 = 0 ;
    OK (GxB_Matrix_Option_get_FP64 (A, GxB_HYPER_SWITCH, &hratio2)) ;
    printf ("hratio: %g %g %g\n", hratio, hratio2, hratio-hratio2) ;
    CHECK (hratio == hratio2) ;

    double bswitch [GxB_NBITMAP_SWITCH] ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        bswitch [k] = (double) k / 16.0 ;
    }
    OK (GxB_Global_Option_set_(GxB_BITMAP_SWITCH, bswitch)) ;

    double bswitch2 [GxB_NBITMAP_SWITCH] ;
    OK (GxB_Global_Option_get_(GxB_BITMAP_SWITCH, bswitch2)) ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        CHECK (fabs (bswitch [k] - bswitch2 [k]) < 1e-5) ;
    }

    OK (GxB_Global_Option_set_FP64_ARRAY (GxB_BITMAP_SWITCH, NULL)) ;
    OK (GxB_Global_Option_set_FP64_ARRAY (GxB_BITMAP_SWITCH, bswitch)) ;
    OK (GxB_Global_Option_get_(GxB_BITMAP_SWITCH, bswitch2)) ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        CHECK (fabs (bswitch [k] - bswitch2 [k]) < 1e-5) ;
    }

    double bswitch3 [GxB_NBITMAP_SWITCH] ;
    OK (GxB_Global_Option_get_FP64 (GxB_BITMAP_SWITCH, bswitch3)) ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        CHECK (bswitch2 [k] == bswitch3 [k]) ;
    }

    OK (GxB_Global_Option_set_(GxB_BITMAP_SWITCH, NULL)) ;
    OK (GxB_Global_Option_get_(GxB_BITMAP_SWITCH, bswitch)) ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        printf ("default bswitch [%d] = %g\n", k, bswitch [k]) ;
    }

    OK (GxB_Global_Option_set_FP64_ARRAY (GxB_BITMAP_SWITCH, NULL)) ;
    OK (GxB_Global_Option_get_ (GxB_BITMAP_SWITCH, bswitch2)) ;
    OK (GxB_Global_Option_get_FP64 (GxB_BITMAP_SWITCH, bswitch3)) ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        CHECK (bswitch2 [k] == bswitch3 [k]) ;
    }

    OK (GxB_Matrix_Option_set_(A, GxB_FORMAT, GxB_BY_COL)) ;
    CHECK (GB_IS_HYPERSPARSE (A)) ;
    CHECK (A->is_csc) ;

    OK (GxB_Matrix_Option_set_INT32 (A, GxB_FORMAT, GxB_BY_ROW)) ;
    CHECK (GB_IS_HYPERSPARSE (A)) ;
    CHECK (!A->is_csc) ;

    OK (GxB_Matrix_Option_set_(A, GxB_FORMAT, GxB_BY_COL)) ;
    CHECK (GB_IS_HYPERSPARSE (A)) ;
    CHECK (A->is_csc) ;

    int format = 0;
    OK (GxB_Matrix_Option_get_(A, GxB_FORMAT, &format)) ;
    CHECK (format == GxB_BY_COL) ;

    int f2 = 99 ;
    OK (GxB_Matrix_Option_get_INT32 (A, GxB_FORMAT, &f2)) ;
    CHECK (f2 == GxB_BY_COL) ;

    OK (GxB_Matrix_Option_set_(A, GxB_FORMAT, GxB_BY_ROW)) ;
    CHECK (!A->is_csc) ;

    OK (GxB_Matrix_Option_set_INT32 (A, GxB_FORMAT, GxB_BY_COL)) ;
    CHECK (A->is_csc) ;

    OK (GxB_Matrix_Option_set_INT32 (A, GxB_FORMAT, GxB_BY_ROW)) ;
    CHECK (!A->is_csc) ;

    OK (GxB_Matrix_Option_get_(A, GxB_FORMAT, &format)) ;
    CHECK (format == GxB_BY_ROW) ;

    OK (GxB_Matrix_Option_get_INT32 (A, GxB_FORMAT, &f2)) ;
    CHECK (f2 == GxB_BY_ROW) ;

    OK (GxB_Matrix_Option_set_(A, GxB_FORMAT, GxB_BY_COL)) ;
    CHECK (A->is_csc) ;

    OK (GxB_Matrix_Option_set_INT32(A, GxB_FORMAT, GxB_BY_ROW)) ;
    CHECK (!A->is_csc) ;

    OK (GxB_Matrix_Option_set_INT32(A, GxB_FORMAT, GxB_BY_COL)) ;
    CHECK (A->is_csc) ;

    OK (GxB_Global_Option_set_(GxB_FORMAT, GxB_BY_ROW)) ;
    format = 99 ;
    OK (GxB_Global_Option_get_(GxB_FORMAT, &format)) ;
    CHECK (format == 0) ;
    int32_t format2 = 999 ;
    OK (GxB_Global_Option_get_INT32 (GxB_FORMAT, &format2)) ;
    CHECK (format2 == 0) ;

    OK (GxB_Global_Option_set_INT32 (GxB_FORMAT, GxB_BY_COL)) ;
    OK (GxB_Global_Option_get_INT32 (GxB_FORMAT, &format2)) ;
    CHECK (format2 == 1) ;

    OK (GxB_Global_Option_set_INT32 (GxB_FORMAT, GxB_BY_ROW)) ;
    OK (GxB_Global_Option_get_INT32 (GxB_FORMAT, &format2)) ;
    CHECK (format2 == 0) ;

    OK (GxB_Global_Option_set_(GxB_HYPER_SWITCH, 77.33f)) ;
    OK (GxB_Global_Option_get_(GxB_HYPER_SWITCH, &hratio)) ;
    printf ("%g\n", hratio) ;
    CHECK (hratio == 77.33f) ;
    hratio2 = 88 ;
    OK (GxB_Global_Option_get_FP64 (GxB_HYPER_SWITCH, &hratio2)) ;
    CHECK (hratio == hratio2) ;

    OK (GxB_Global_Option_set_FP64 (GxB_HYPER_SWITCH, 88.22f)) ;
    OK (GxB_Global_Option_get_FP64 (GxB_HYPER_SWITCH, &hratio)) ;
    CHECK (hratio == 88.22f) ;

    OK (GxB_Global_Option_set_(GxB_HYPER_SWITCH, GxB_HYPER_DEFAULT)) ;
    OK (GxB_Global_Option_get_(GxB_HYPER_SWITCH, &hratio)) ;
    CHECK (hratio == GxB_HYPER_DEFAULT) ;

    OK (GxB_Global_Option_set_(GxB_HYPER_SWITCH, 12.34f)) ;
    OK (GxB_Global_Option_set_FP64 (GxB_HYPER_SWITCH, GxB_HYPER_DEFAULT)) ;
    OK (GxB_Global_Option_get_FP64 (GxB_HYPER_SWITCH, &hratio)) ;
    CHECK (hratio == GxB_HYPER_DEFAULT) ;

    hratio2 = 22 ;
    OK (GxB_Global_Option_get_FP64 (GxB_HYPER_SWITCH, &hratio2)) ;
    CHECK (hratio == hratio2) ;

    expected = GrB_NULL_POINTER ;
    GrB_Matrix O_NULL = NULL ;
    ERR1 (O_NULL, GxB_Matrix_Option_set_(O_NULL, GxB_FORMAT, GxB_BY_COL)) ;
    ERR1 (O_NULL, GxB_Matrix_Option_set_INT32 (O_NULL, GxB_FORMAT, GxB_BY_COL)) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Global_Option_get_(GxB_FORMAT, NULL)) ;
    ERR (GxB_Global_Option_get_INT32 (GxB_FORMAT, NULL)) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Matrix_Option_get_(A, GxB_FORMAT, NULL)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected (A format null):%s\n", err) ;

    ERR (GxB_Matrix_Option_get_INT32 (A, GxB_FORMAT, NULL)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected (A format null):%s\n", err) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Matrix_Option_get_(A, GxB_HYPER_SWITCH, NULL)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected:%s\n", err) ;

    ERR (GxB_Matrix_Option_get_FP64 (A, GxB_HYPER_SWITCH, NULL)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected:%s\n", err) ;

    ERR (GxB_Matrix_Option_get_(A, GxB_BITMAP_SWITCH, NULL)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected:%s\n", err) ;

    ERR (GxB_Matrix_Option_get_FP64 (A, GxB_BITMAP_SWITCH, NULL)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected:%s\n", err) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Global_Option_get_(GxB_HYPER_SWITCH, NULL)) ;
    ERR (GxB_Global_Option_get_(GxB_BITMAP_SWITCH, NULL)) ;
    ERR (GxB_Global_Option_get_FP64 (GxB_HYPER_SWITCH, NULL)) ;
    ERR (GxB_Global_Option_get_FP64 (GxB_BITMAP_SWITCH, NULL)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Global_Option_get_(-1, NULL)) ;

    int32_t ignore2 = 999 ;
    ERR (GxB_Global_Option_get_INT32 (-1, &ignore2)) ;
    CHECK (ignore2 == 999) ;

    int64_t ignore3 = 999 ;
    ERR (GxB_Global_Option_get_INT64 (-1, &ignore3)) ;
    CHECK (ignore3 == 999) ;

    double ignore4 = 999 ;
    ERR (GxB_Global_Option_get_FP64 (-1, &ignore4)) ;
    CHECK (ignore4 == 999) ;

    void *ignore5 = NULL ;
    ERR (GxB_Global_Option_get_FUNCTION (-1, &ignore5)) ;
    CHECK (ignore5 == NULL) ;

    const char *ignore6 = NULL ;
    ERR (GxB_Global_Option_get_CHAR (-1, &ignore6)) ;
    CHECK (ignore6 == NULL) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Matrix_Option_get_(A, 999, NULL)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected (bad field):%s\n", err) ;

    int ii ;
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Matrix_Option_get_INT32 (A, 999, &ii)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected (bad field):%s\n", err) ;

    double xx ;
    ERR (GxB_Matrix_Option_get_FP64 (A, 999, &xx)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected (bad field):%s\n", err) ;

    ERR1 (A, GxB_Matrix_Option_set_(A, 999, GxB_BY_ROW)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected:%s\n", err) ;

    ERR1 (A, GxB_Matrix_Option_set_INT32 (A, 999, GxB_BY_ROW)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected:%s\n", err) ;

    ERR (GxB_Global_Option_set_(999, GxB_BY_ROW)) ;
    ERR (GxB_Global_Option_set_INT32 (999, GxB_BY_ROW)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Global_Option_set_(GxB_FORMAT, 9999)) ;
    ERR (GxB_Global_Option_set_INT32 (GxB_FORMAT, 9999)) ;

    ERR1 (A, GxB_Matrix_Option_set_(A, 999, GxB_BY_ROW)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected:%s\n", err) ;

    ERR1 (A, GxB_Matrix_Option_set_INT32 (A, 999, GxB_BY_ROW)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected:%s\n", err) ;

    ERR1 (A, GxB_Matrix_Option_set_(A, GxB_FORMAT, 909090)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected:%s\n", err) ;

    ERR1 (A, GxB_Matrix_Option_set_INT32 (A, GxB_FORMAT, 909090)) ;
    GrB_Matrix_error_(&err, A) ;
    printf ("error expected:%s\n", err) ;

    CHECK (A != NULL) ;
    CHECK (A->is_csc) ;

    OK (GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;

    expected = GrB_INVALID_OBJECT ;

    void *Ah_save = A->h ;
    A->h = NULL ;
    ERR (GB_Matrix_check (A, "h invalid", G3, NULL)) ;
    A->h = Ah_save ;
    OK (GB_Matrix_check (A, "h restored", G1, NULL)) ;

    uint64_t nvec = A->nvec ;
    A->nvec = -1 ;
    ERR (GB_Matrix_check (A, "nvec invalid", G1, NULL)) ;
    A->nvec = nvec ;
    OK (GB_Matrix_check (A, "nvec restored", G1, NULL)) ;

    CHECK (!GB_IS_FULL (A)) ;
    CHECK (A->h != NULL) ;
    GB_MDECL (Ah, , ) ;
    Ah = A->h ;
    GB_IPTR (Ah, A->j_is_32) ;
    int64_t jsave = GB_IGET (Ah, 0) ;
    GB_ISET (Ah, 0, -1) ;       // Ah [0] = -1 ;
    ERR (GB_Matrix_check (A, "h[0] invalid", G1, NULL)) ;
    GB_ISET (Ah, 0, jsave) ;    // Ah [0] = jsave ;
    OK (GB_Matrix_check (A, "h[0] restored", G1, NULL)) ;

    GrB_Matrix Eleven ;
    OK (GrB_Matrix_new (&Eleven, GrB_BOOL, 11, 11)) ;
    I [0] = 0 ;
    OK (GrB_Matrix_assign_BOOL (Eleven, NULL, NULL, (bool) true, I, 1, GrB_ALL, 0, NULL)) ;
    OK (GB_Matrix_check (Eleven, "Eleven", G2, NULL)) ;

    if (!GB_IS_FULL (Eleven)) GB_wait (Eleven, "Eleven", Werk) ;

    for (int pr = -4 ; pr <= 3 ; pr++)
    {
        OK (GxB_Matrix_fprint (Eleven, "Eleven", pr, NULL)) ;
        OK (GxB_Matrix_fprint (Eleven, "Eleven is OK", pr, ff)) ;
        OK (GxB_Matrix_fprint (Eleven, "Eleven", pr, ff)) ;
    }

    OK (GB_convert_hyper_to_sparse (Eleven, true)) ;
    int64_t nothing = 42 ;
    printf ("\nEleven invalid hypersparse:\n") ;
    GB_free_memory ((void **) &(Eleven->h), Eleven->h_size) ;
    Eleven->h = &nothing ;
    Eleven->h_size = sizeof (int64_t) ;
    printf ("test0, add Eleven->h to memtable %d\n", Eleven->h) ;
    GB_Global_memtable_add (Eleven->h, 1 * sizeof (int64_t)) ;
    ERR (GB_Matrix_check (Eleven, "Eleven invalid", G2, NULL)) ;
    ERR (GxB_Matrix_fprint (Eleven, "Eleven", G2, NULL)) ;
    ERR (GxB_Matrix_fprint (Eleven, "Eleven invalid", G2, ff)) ;
    printf ("test0, remove Eleven->h from memtable %d\n", Eleven->h) ;
    GB_Global_memtable_remove (Eleven->h) ;
    Eleven->h = NULL ;

    OK (GrB_Matrix_free_(&Eleven)) ;

    OK (GrB_Matrix_new (&Eleven, GrB_BOOL, 11, 11)) ;
    for (int64_t i = 0 ; i < 11 ; i++)
    {
        for (int64_t j = 0 ; j < 11 ; j++)
        {
            GrB_Matrix_setElement_BOOL (Eleven, true, i, j) ;
        }
    }
    OK (GrB_Matrix_nvals (&nvals, Eleven)) ;
    OK (GrB_Matrix_wait_(Eleven, GrB_MATERIALIZE)) ;
    CHECK (nvals == 121) ;
    OK (GB_Matrix_check (Eleven, "Eleven", G2, NULL)) ;
    OK (GrB_Matrix_free_(&Eleven)) ;

    printf ("\nAll GB_Matrix_check tests passed (errors expected)\n") ;

    //--------------------------------------------------------------------------
    // blocking vs non-blocking mode
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_setElement_FP64 (A, 32.4, 3, 2)) ;
    OK (GB_Matrix_check (A, "A with one pending", G3, NULL)) ;
    A_Pending = A->Pending ;
    CHECK (A_Pending != NULL) ;
    CHECK (A_Pending->n == 1 && A->nzombies == 0) ;
    GB_Global_mode_set (GrB_BLOCKING) ;
    OK (GB_block (A, Werk)) ;
    OK (GB_Matrix_check (A, "A with no pending", G3, NULL)) ;
    A_Pending = A->Pending ;
    CHECK (A_Pending == NULL) ;
    CHECK (A->nzombies == 0) ;
    OK (GrB_Matrix_setElement_FP64 (A, 99.4, 3, 3)) ;
    OK (GB_Matrix_check (A, "A blocking mode", G3, NULL)) ;
    GB_Global_mode_set (GrB_NONBLOCKING) ;
    A_Pending = A->Pending ;
    CHECK (A_Pending == NULL) ;
    CHECK (A->nzombies == 0) ;

    printf ("\nAll blocking/nonblocking mode tests passed\n") ;

    //--------------------------------------------------------------------------
    // restore all 'crud' objects so they can be freed
    //--------------------------------------------------------------------------

    expected = GrB_UNINITIALIZED_OBJECT ;

    Werk->where = "GB *_check" ;

    ERR (GB_Type_check (Tcrud, "", G0, NULL)) ;
    ERR (GB_UnaryOp_check (op1crud, "", G0, NULL)) ;
    ERR (GB_BinaryOp_check (op2crud, "", G0, NULL)) ;
    ERR (GB_Monoid_check (monoid_crud, "", G0, NULL, false)) ;
    ERR (GB_Semiring_check (semicrud, "", G0, NULL)) ;
    ERR (GB_Vector_check (vcrud, "", G0, NULL)) ;
    ERR (GB_Matrix_check (Acrud, "", G0, NULL)) ;
    ERR (GB_Descriptor_check (dcrud, "", G0, NULL)) ;

    ERR (GxB_Type_fprint (Tcrud, "crud", G0, ff)) ;
    ERR (GxB_UnaryOp_fprint (op1crud, "crud", G0, ff)) ;
    ERR (GxB_BinaryOp_fprint (op2crud, "crud", G0, ff)) ;
    ERR (GxB_Monoid_fprint (monoid_crud, "crud", G0, ff)) ;
    ERR (GxB_Semiring_fprint (semicrud, "crud", G0, ff)) ;
    ERR (GxB_Vector_fprint (vcrud, "crud", G0, ff)) ;
    ERR (GxB_Matrix_fprint (Acrud, "crud", G0, ff)) ;
    ERR (GxB_Descriptor_fprint (dcrud, "crud", G0, ff)) ;

    #define REMAGIC(p) if (p != NULL) p->magic = GB_MAGIC ;
    REMAGIC (Tcrud)
    REMAGIC (op1crud)
    REMAGIC (op2crud)
    REMAGIC (monoid_crud)
    REMAGIC (semicrud)
    REMAGIC (vcrud)
    REMAGIC (Acrud)
    REMAGIC (dcrud)
    #undef REMAGIC

    OK (GB_Type_check (Tcrud, "", G0, NULL)) ;
    OK (GB_UnaryOp_check (op1crud, "", G0, NULL)) ;
    OK (GB_BinaryOp_check (op2crud, "", G0, NULL)) ;
    OK (GB_Monoid_check (monoid_crud, "", G0, NULL, false)) ;
    OK (GB_Semiring_check (semicrud, "", G0, NULL)) ;
    OK (GB_Vector_check (vcrud, "", G0, NULL)) ;
    OK (GB_Matrix_check (Acrud, "", G0, NULL)) ;
    OK (GB_Descriptor_check (dcrud, "", G0, NULL)) ;

    OK (GxB_Type_fprint_(Tcrud, G0, ff)) ;
    OK (GxB_UnaryOp_fprint_(op1crud, G0, ff)) ;
    OK (GxB_BinaryOp_fprint_(op2crud, G0, ff)) ;
    OK (GxB_Monoid_fprint_(monoid_crud, G0, ff)) ;
    OK (GxB_Semiring_fprint_(semicrud, G0, ff)) ;
    OK (GxB_Vector_fprint_(vcrud, G0, ff)) ;
    OK (GxB_Matrix_fprint_(Acrud, G0, ff)) ;
    OK (GxB_Descriptor_fprint_(dcrud, G0, ff)) ;

    //--------------------------------------------------------------------------
    // GB_Descriptor_get
    //--------------------------------------------------------------------------

    expected = GrB_INVALID_OBJECT ;
    dcrud->out = 999 ;
    x_bool = false ;
    Werk->where = "GB_Descriptor_get" ;
    ERR (GB_Descriptor_get (dcrud, &x_bool, NULL, NULL, NULL, NULL, NULL,
        NULL)) ;
    CHECK (x_bool == false) ;
    dcrud->out = GxB_DEFAULT ;

    //--------------------------------------------------------------------------
    // malloc wrappers
    //--------------------------------------------------------------------------

    size_t nbytes ;
    pp = (GB_void *) &x ;
    pp = GB_malloc_memory (UINT64_MAX, 1, &nbytes) ;
    CHECK (pp == NULL) ;
    CHECK (nbytes == 0) ;

    pp = (GB_void *) &x ;
    pp = GB_calloc_memory (UINT64_MAX, 1, &nbytes) ;
    CHECK (pp == NULL) ;
    CHECK (nbytes == 0) ;

    ok = true ;
    pp = GB_realloc_memory (UINT64_MAX, 1, NULL, &nbytes, &ok) ;
    CHECK (!ok) ;
    CHECK (nbytes == 0) ;

    s = 1 ;
    ok = GB_size_t_multiply (&s, UINT64_MAX, 0) ;
    CHECK (ok) ;
    CHECK (s == 0) ;

    s = 911 ;
    ok = GB_size_t_multiply (&s, UINT64_MAX/2, UINT64_MAX/2) ;
    CHECK (!ok) ;
    CHECK (s == 0) ;

    ok = GB_size_t_multiply (&s,
        ((size_t) UINT32_MAX)+2,
        ((size_t) UINT32_MAX)+1) ;
    CHECK (!ok) ;

    ok = GB_size_t_multiply (&s,
        ((size_t) UINT32_MAX)+1,
        ((size_t) UINT32_MAX)+1) ;
    CHECK (!ok) ;

    ok = GB_size_t_multiply (&s,
        ((size_t) UINT32_MAX)/4,
        ((size_t) UINT32_MAX)/4) ;
    CHECK (ok) ;
    CHECK (s == (((size_t) UINT32_MAX)/4) * (((size_t) UINT32_MAX)/4)) ;
    Werk->where = "GrB_error" ;

    n = 1 ;
    ok = GB_int64_multiply (&n, INT64_MAX, 0) ;
    CHECK (ok) ;
    CHECK (n == 0) ;

    n = 911 ;
    ok = GB_int64_multiply (&n, -1, -1) ;
    CHECK (!ok) ;
    CHECK (n == 0) ;

    n = 911 ;
    ok = GB_int64_multiply (&n, 1, GB_NMAX+1) ;
    CHECK (!ok) ;
    CHECK (n == 0) ;

    ok = GB_int64_multiply (&n, ((uint64_t) GB_NMAX)+1, ((uint64_t) GB_NMAX)+1);
    CHECK (!ok) ;

    ok = GB_int64_multiply (&n, ((uint64_t) GB_NMAX), ((uint64_t) GB_NMAX)) ;
    CHECK (!ok) ;

    a = (uint64_t) 16777216/2 ;     // (2^24)/2
    b = (uint64_t) 16777216 ;
    ok = GB_int64_multiply (&n, a, b) ;
    // printf ("%lld %lld n\n", n, a*b) ;
    CHECK (ok) ;
    CHECK (n == a*b)

    //--------------------------------------------------------------------------
    // internal GB * routines
    //--------------------------------------------------------------------------

    Werk->where = "GB_ix_realloc" ;

    CHECK (A != NULL) ;
    OK (GxB_Matrix_fprint (A, "A for ix_realloc", G3, NULL)) ;
    info = GB_ix_realloc (A, GB_NMAX+1) ;
    CHECK (info == GrB_OUT_OF_MEMORY) ;

    OK (GB_Matrix_check (A, "A pattern 1", G3, NULL)) ;
    OK (GB_ix_realloc (A, 20)) ;
    CHECK (info == GrB_SUCCESS) ;
    OK (GB_Matrix_check (A, "A pattern 2", G3, NULL)) ;

    GB_bix_free (NULL) ;
    GB_phy_free (NULL) ;

    GrB_Matrix_free_(&C) ;
    GrB_Matrix_free_(&B) ;
    CHECK (C == NULL) ;
    CHECK (B == NULL) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, 1, 1)) ;
    OK (GB_Matrix_check (A, "A for shallow op", G3, NULL)) ;
    Werk->where = "GB_shallow_op" ;

    struct GB_Matrix_opaque Q_header ;
    GrB_Matrix Q = GB_clear_matrix_header (&Q_header) ;
    OK (GB_shallow_op (Q, true, (GB_Operator) GrB_AINV_FP32, NULL, NULL,
        false, C, Werk)) ;
    OK (GB_Matrix_check (Q, "Q empty, float", G3, NULL)) ;
    GrB_Matrix_free_(&Q) ;

    bool b1, b2 ;
    int64_t imin, imax ;

    OK (GB_op_is_second (GrB_SECOND_FP64, NULL)) ;

    //--------------------------------------------------------------------------
    // check for inputs aliased with outputs
    //--------------------------------------------------------------------------

    printf ("free A:\n") ; GrB_Matrix_free_(&A) ;
    printf ("free B:\n") ; GrB_Matrix_free_(&B) ;
    printf ("free C:\n") ; GrB_Matrix_free_(&C) ;
    printf ("free E:\n") ; GrB_Matrix_free_(&E) ;
    printf ("free F:\n") ; GrB_Matrix_free_(&F) ;
    printf ("free v:\n") ; GrB_Vector_free_(&v) ;
    printf ("free u:\n") ; GrB_Vector_free_(&u) ;
    printf ("free z:\n") ; GrB_Vector_free_(&z) ;

    #define NWHAT 12
    n = NWHAT ;
    nvals = 40 ;
    uvals = 4 ;
    uint64_t ilist [NWHAT] = { 8, 9, 0, 1, 5, 6, 11, 3, 2, 10, 7, 4 } ; // OK
    uint64_t jlist [NWHAT] = { 0, 11, 1, 7, 8, 4, 2, 3, 5, 6, 10, 9 } ; // OK

    OK (GB_mx_random_matrix (&A, false, false, n, n, nvals, 0, false)) ;
    OK (GrB_Vector_new (&u, GrB_FP64, n)) ;
    OK (GrB_Vector_setElement_FP64 (u, (double) 3.4, 0)) ;

    E = A ;
    GrB_Matrix_dup (&A, A) ;
    CHECK (GB_mx_isequal (A, E, 0)) ;
    GrB_Matrix_free_(&E) ;

    z = u ;
    GrB_Vector_dup (&u, u) ;
    CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) z, 0)) ;
    GrB_Vector_free_(&z) ;

    for (int what = 0 ; what <= 2 ; what++)
    {

        GrB_Matrix Amask ;
        GrB_Vector umask ;
        switch (what)
        {
            case 0: Amask = NULL ; umask = NULL ; break ;
            case 1: Amask = A    ; umask = u    ; break ;
            case 2:
            OK (GB_mx_random_matrix (&Amask, false, false, n, n, nvals, 0,
                false)) ;
            OK (GB_mx_random_matrix (&F,false, false, n, 1, uvals, 0, false)) ;
            // vectors cannot be hypersparse
            OK (GxB_Matrix_Option_set_(F, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
            // vectors cannot be CSC: this is a hack just for brutal testing
            OK (GxB_Matrix_Option_set_(F, GxB_FORMAT, GxB_BY_COL)) ;
            umask = (GrB_Vector) F ;
            F = NULL ;
            default:
            break ;
        }

        CHECK (umask == NULL || GB_VECTOR_OK (umask)) ;

        //----------------------------------------------------------------------
        // GrB_mxm, GrB_vxm, and GrB_mxv
        //----------------------------------------------------------------------

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_mxm (B, Amask, NULL, GxB_PLUS_TIMES_FP64, A, A, NULL)) ;
        OK (GrB_mxm (A, Amask, NULL, GxB_PLUS_TIMES_FP64, A, A, NULL)) ;
        OK (GxB_Matrix_fprint (A, "A ok", G3, ff)) ;
        OK (GxB_Matrix_fprint (B, "B ok", G3, ff)) ;
        if (Amask != NULL)
        {
            OK (GxB_Matrix_fprint (Amask, "Amask ok", G3, ff)) ;
        }

        CHECK (GB_mx_isequal (A, B, 1e-14)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_vxm (v, umask, NULL, GxB_PLUS_TIMES_FP64, u, A, NULL)) ;

        OK (GrB_vxm (u, umask, NULL, GxB_PLUS_TIMES_FP64, u, A, NULL)) ;
        OK (GxB_Vector_fprint (u, "u ok", G3, ff)) ;
        OK (GxB_Vector_fprint (v, "v ok", G3, ff)) ;
        if (umask != NULL)
        {
            OK (GxB_Vector_fprint (umask, "umask ok", G3, ff)) ;
        }

        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 1e-14)) ;
        GrB_Vector_free_(&v) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_mxv (v, umask, NULL, GxB_PLUS_TIMES_FP64, A, u, NULL)) ;
        OK (GrB_mxv (u, umask, NULL, GxB_PLUS_TIMES_FP64, A, u, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 1e-14)) ;
        GrB_Vector_free_(&v) ;

        //----------------------------------------------------------------------
        // GrB_eWiseMult
        //----------------------------------------------------------------------

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_Vector_eWiseMult_Semiring_(v, umask, NULL, GxB_PLUS_TIMES_FP64,  u, u, NULL)) ;
        OK (GrB_Vector_eWiseMult_Semiring_(u, umask, NULL, GxB_PLUS_TIMES_FP64,  u, u, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 1e-14)) ;
        GrB_Vector_free_(&v) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_Vector_eWiseMult_Monoid_(v, umask, NULL, GxB_PLUS_FP64_MONOID, u, u, NULL)) ;
        OK (GrB_Vector_eWiseMult_Monoid_(u, umask, NULL, GxB_PLUS_FP64_MONOID, u, u, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 1e-14)) ;
        GrB_Vector_free_(&v) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_Vector_eWiseMult_BinaryOp_(v, umask, NULL, GrB_PLUS_FP64,        u, u, NULL)) ;
        OK (GrB_Vector_eWiseMult_BinaryOp_(u, umask, NULL, GrB_PLUS_FP64,        u, u, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 1e-14)) ;
        GrB_Vector_free_(&v) ;

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_Matrix_eWiseMult_Semiring_(B, Amask, NULL, GxB_PLUS_TIMES_FP64,  A, A, NULL)) ;
        OK (GrB_Matrix_eWiseMult_Semiring_(A, Amask, NULL, GxB_PLUS_TIMES_FP64,  A, A, NULL)) ;
        CHECK (GB_mx_isequal (A, B, 1e-14)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_Matrix_eWiseMult_Monoid_(B, Amask, NULL, GxB_PLUS_FP64_MONOID, A, A, NULL)) ;
        OK (GrB_Matrix_eWiseMult_Monoid_(A, Amask, NULL, GxB_PLUS_FP64_MONOID, A, A, NULL)) ;
        CHECK (GB_mx_isequal (A, B, 1e-14)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_Matrix_eWiseMult_BinaryOp_(B, Amask, NULL, GrB_PLUS_FP64,        A, A, NULL)) ;
        OK (GrB_Matrix_eWiseMult_BinaryOp_(A, Amask, NULL, GrB_PLUS_FP64,        A, A, NULL)) ;
        CHECK (GB_mx_isequal (A, B, 1e-14)) ;
        GrB_Matrix_free_(&B) ;

        //----------------------------------------------------------------------
        // GrB_eWiseAdd
        //----------------------------------------------------------------------

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_Vector_eWiseAdd_Semiring_(v, umask, NULL, GxB_PLUS_TIMES_FP64,  u, u, NULL)) ;
        OK (GrB_Vector_eWiseAdd_Semiring_(u, umask, NULL, GxB_PLUS_TIMES_FP64,  u, u, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 1e-14)) ;
        GrB_Vector_free_(&v) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_Vector_eWiseAdd_Monoid_(v, umask, NULL, GxB_PLUS_FP64_MONOID, u, u, NULL)) ;
        OK (GrB_Vector_eWiseAdd_Monoid_(u, umask, NULL, GxB_PLUS_FP64_MONOID, u, u, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 1e-14)) ;
        GrB_Vector_free_(&v) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_Vector_eWiseAdd_BinaryOp_(v, umask, NULL, GrB_PLUS_FP64,        u, u, NULL)) ;
        OK (GrB_Vector_eWiseAdd_BinaryOp_(u, umask, NULL, GrB_PLUS_FP64,        u, u, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 1e-14)) ;
        GrB_Vector_free_(&v) ;

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_Matrix_eWiseAdd_Semiring_(B, Amask, NULL, GxB_PLUS_TIMES_FP64,  A, A, NULL)) ;
        OK (GrB_Matrix_eWiseAdd_Semiring_(A, Amask, NULL, GxB_PLUS_TIMES_FP64,  A, A, NULL)) ;
        CHECK (GB_mx_isequal (A, B, 1e-14)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_Matrix_eWiseAdd_Monoid_(B, Amask, NULL, GxB_PLUS_FP64_MONOID, A, A, NULL)) ;
        OK (GrB_Matrix_eWiseAdd_Monoid_(A, Amask, NULL, GxB_PLUS_FP64_MONOID, A, A, NULL)) ;
        CHECK (GB_mx_isequal (A, B, 1e-14)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_Matrix_eWiseAdd_BinaryOp_(B, Amask, NULL, GrB_PLUS_FP64,        A, A, NULL)) ;
        OK (GrB_Matrix_eWiseAdd_BinaryOp_(A, Amask, NULL, GrB_PLUS_FP64,        A, A, NULL)) ;
        CHECK (GB_mx_isequal (A, B, 1e-14)) ;
        GrB_Matrix_free_(&B) ;

        //----------------------------------------------------------------------
        // GrB_extract
        //----------------------------------------------------------------------

        printf ("\nGrB_extract ============================================\n");

        OK (GrB_Vector_dup (&v, u)) ;
        GB_Vector_check (u, "start u ", G3, NULL) ;
        GB_Vector_check (v, "start v ", G3, NULL) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 0)) ;

        OK (GrB_Vector_extract_(u, umask, NULL, u, GrB_ALL, n, NULL)) ;
        GB_Vector_check (u, "u to check", G3, NULL) ;
        GB_Vector_check (v, "v to check", G3, NULL) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 0)) ;
        GrB_Vector_free_(&v) ;

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_Matrix_extract_(B, Amask, NULL, A, GrB_ALL, n, GrB_ALL, n, NULL)) ;
        OK (GrB_Matrix_extract_(A, Amask, NULL, A, GrB_ALL, n, GrB_ALL, n, NULL)) ;
        CHECK (GB_mx_isequal (A, B, 0)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_Col_extract_(v, umask, NULL, A, GrB_ALL, n, 0, NULL)) ;
        OK (GrB_Col_extract_(u, umask, NULL, A, GrB_ALL, n, 0, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 0)) ;
        GrB_Vector_free_(&v) ;

        //----------------------------------------------------------------------
        // GxB_subassign
        //----------------------------------------------------------------------

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GxB_Matrix_subassign_(B, Amask, NULL, A, GrB_ALL, n, GrB_ALL, n, NULL)) ;
        OK (GxB_Matrix_subassign_(A, Amask, NULL, A, GrB_ALL, n, GrB_ALL, n, NULL)) ;

        GB_wait (A, "A", Werk) ;
        GB_wait (B, "B", Werk) ;
        CHECK (GB_mx_isequal (A, B, 0)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GxB_Matrix_subassign_(B, Amask, NULL, A, ilist, n, jlist, n, NULL));
        OK (GxB_Matrix_subassign_(A, Amask, NULL, A, ilist, n, jlist, n, NULL));
        GB_wait (A, "A", Werk) ;
        GB_wait (B, "B", Werk) ;
        CHECK (GB_mx_isequal (A, B, 0)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GxB_Vector_subassign_(v, umask, NULL, u, GrB_ALL, n, NULL)) ;
        OK (GxB_Vector_subassign_(u, umask, NULL, u, GrB_ALL, n, NULL)) ;
        GB_wait ((GrB_Matrix) u, "u", Werk) ;
        GB_wait ((GrB_Matrix) v, "v", Werk) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 0)) ;
        GrB_Vector_free_(&v) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GxB_Vector_subassign_(v, umask, NULL, u, ilist, n, NULL)) ;
        OK (GxB_Vector_subassign_(u, umask, NULL, u, ilist, n, NULL)) ;
        GB_wait ((GrB_Matrix) v, "v", Werk) ;
        GB_wait ((GrB_Matrix) u, "u", Werk) ;

        OK (GxB_Vector_fprint_(v, G3, NULL)) ;
        OK (GxB_Vector_fprint_(u, G3, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 0)) ;
        GrB_Vector_free_(&v) ;

        //----------------------------------------------------------------------
        // GrB_assign
        //----------------------------------------------------------------------

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_Matrix_assign_(B, Amask, NULL, A, GrB_ALL, n, GrB_ALL, n, NULL)) ;
        OK (GrB_Matrix_assign_(A, Amask, NULL, A, GrB_ALL, n, GrB_ALL, n, NULL)) ;
        CHECK (GB_mx_isequal (A, B, 0)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_Matrix_assign_(B, Amask, NULL, A, ilist, n, jlist, n, NULL)) ;
        OK (GrB_Matrix_assign_(A, Amask, NULL, A, ilist, n, jlist, n, NULL)) ;
        if (!GB_IS_FULL (B)) GB_wait (B, "B", Werk) ;
        if (!GB_IS_FULL (A)) GB_wait (A, "A", Werk) ;
        CHECK (GB_mx_isequal (A, B, 0)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_Vector_assign_(v, umask, NULL, u, GrB_ALL, n, NULL)) ;
        OK (GrB_Vector_assign_(u, umask, NULL, u, GrB_ALL, n, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 0)) ;
        GrB_Vector_free_(&v) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_Vector_assign_(v, umask, NULL, u, ilist, n, NULL)) ;
        OK (GrB_Vector_assign_(u, umask, NULL, u, ilist, n, NULL)) ;
        if (!GB_IS_FULL (v)) GB_wait ((GrB_Matrix) v, "v", Werk) ;
        if (!GB_IS_FULL (u)) GB_wait ((GrB_Matrix) u, "u", Werk) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 0)) ;
        GrB_Vector_free_(&v) ;

        //----------------------------------------------------------------------
        // GrB_apply
        //----------------------------------------------------------------------

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GrB_Matrix_apply_(B, Amask, NULL, GrB_AINV_FP64, A, NULL)) ;
        OK (GrB_Matrix_apply_(A, Amask, NULL, GrB_AINV_FP64, A, NULL)) ;
        CHECK (GB_mx_isequal (A, B, 1e-14)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GrB_Vector_apply_(v, umask, NULL, GrB_AINV_FP64, u, NULL)) ;
        OK (GrB_Vector_apply_(u, umask, NULL, GrB_AINV_FP64, u, NULL)) ;
        OK (GxB_Vector_fprint_(v, G3, NULL)) ;
        OK (GxB_Vector_fprint_(u, G3, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 1e-14)) ;
        GrB_Vector_free_(&v) ;

        //----------------------------------------------------------------------
        // GxB_select
        //----------------------------------------------------------------------

        OK (GrB_Matrix_dup (&B, A)) ;
        OK (GxB_Matrix_select_(B, Amask, NULL, GxB_NONZERO, A, NULL, NULL)) ;
        OK (GxB_Matrix_select_(A, Amask, NULL, GxB_NONZERO, A, NULL, NULL)) ;
        CHECK (GB_mx_isequal (A, B, 0)) ;
        GrB_Matrix_free_(&B) ;

        OK (GrB_Vector_dup (&v, u)) ;
        OK (GxB_Vector_select_(v, umask, NULL, GxB_NONZERO, u, NULL, NULL)) ;
        OK (GxB_Vector_select_(u, umask, NULL, GxB_NONZERO, u, NULL, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) u, (GrB_Matrix) v, 0)) ;
        GrB_Vector_free_(&v) ;

        //----------------------------------------------------------------------
        // GrB_transepose
        //----------------------------------------------------------------------

        OK (GrB_Matrix_dup (&B, A)) ;

        OK (GrB_transpose (B, Amask, NULL, A, NULL)) ;
        OK (GrB_transpose (A, Amask, NULL, A, NULL)) ;
        uint64_t ignore ;

        OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
        OK (GrB_Matrix_wait (B, GrB_MATERIALIZE)) ;
        CHECK (GB_mx_isequal (A, B, 0)) ;
        GrB_Matrix_free_(&B) ;

        if (what == 2)
        {
            GrB_Matrix_free_(&Amask) ;
            GrB_Vector_free_(&umask) ;
        }
    }

    //--------------------------------------------------------------------------
    // nthreads
    //--------------------------------------------------------------------------

    printf ("\n----------------------------- nthreads\n") ;

    int nthreads ;

    OK (GxB_Global_Option_set_(GxB_NTHREADS, 42)) ;
    OK (GxB_Global_Option_get_(GxB_NTHREADS, &nthreads)) ;
    CHECK (nthreads == 42) ;

    OK (GxB_Global_Option_set_INT32 (GxB_NTHREADS, 44)) ;
    OK (GxB_Global_Option_get_INT32 (GxB_NTHREADS, &nthreads)) ;
    CHECK (nthreads == 44) ;

    int32_t nthreads2 = 11 ;
    OK (GxB_Global_Option_get_(GxB_NTHREADS, &nthreads2)) ;
    CHECK (nthreads == nthreads2) ;

    //--------------------------------------------------------------------------
    // import/export
    //--------------------------------------------------------------------------

    printf ("\n----------------------------- import/export\n") ;
    OK (GxB_Matrix_fprint (A, "A to import/export", GxB_COMPLETE, stdout)) ;
    uint64_t *Aj ;      // OK
    uint64_t nrows, ncols, nvecs ;
    double *Ax ;
    GrB_Type atype ;
    bool jumbled, iso ;
    uint64_t Ap_size, Aj_size, Ai_size, Ax_size, Ah_size, Ab_size ;

{
    // import/export are all-64-bit
    uint64_t *Ap, *Ah, *Ai ;    // OK

    OK (GxB_Matrix_export_CSR (&A, &atype, &nrows, &ncols,
        &Ap, &Aj, (void **) &Ax, &Ap_size, &Aj_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    OK (GxB_Type_fprint (atype, "type of A", GxB_COMPLETE, stdout)) ;
    printf ("Ax_size %llu\n", Ax_size) ;
    for (int64_t i = 0 ; i < ((int64_t) nrows) ; i++)
    {
        printf ("exported row %lld\n", j) ;
        for (int64_t p = Ap [i] ; p < ((int64_t) (Ap [i+1])) ; p++)
        {
            printf ("   col %lld value %g\n", Aj [p], Ax [p]) ;
        }
    }
    OK (GxB_Matrix_import_CSR (&A, atype, nrows, ncols,
        &Ap, &Aj, (void **) &Ax, Ap_size, Aj_size, Ax_size, iso, jumbled, desc)) ;

    OK (GxB_Matrix_fprint (A, "A imported", GxB_COMPLETE, stdout)) ;

    expected = GrB_NULL_POINTER ;

    ERR (GxB_Matrix_export_CSR (NULL, &atype, &nrows, &ncols,
        &Ap, &Aj, (void **) &Ax, &Ap_size, &Aj_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSR (&A, NULL, &nrows, &ncols,
        &Ap, &Aj, (void **) &Ax, &Ap_size, &Aj_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSR (&A, &atype, NULL, &ncols,
        &Ap, &Aj, (void **) &Ax, &Ap_size, &Aj_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSR (&A, &atype, &nrows, NULL,
        &Ap, &Aj, (void **) &Ax, &Ap_size, &Aj_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSR (&A, &atype, &nrows, &ncols,
        NULL, &Aj, (void **) &Ax, &Ap_size, &Aj_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSR (&A, &atype, &nrows, &ncols,
        &Ap, NULL, (void **) &Ax, &Ap_size, &Aj_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSR (&A, &atype, &nrows, &ncols,
        &Ap, &Aj, NULL, &Ap_size, &Aj_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSR (&A, &atype, &nrows, &ncols,
        &Ap, &Aj, (void **) &Ax, NULL, &Aj_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSR (&A, &atype, &nrows, &ncols,
        &Ap, &Aj, (void **) &Ax, &Ap_size, NULL, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSR (&A, &atype, &nrows, &ncols,
        &Ap, &Aj, (void **) &Ax, &Ap_size, &Aj_size, NULL, &iso,
        &jumbled, desc)) ;


    ERR (GxB_Matrix_export_CSC (NULL, &atype, &nrows, &ncols,
        &Ap, &Ai, (void **) &Ax, &Ap_size, &Ai_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSC (&A, NULL, &nrows, &ncols,
        &Ap, &Ai, (void **) &Ax, &Ap_size, &Ai_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSC (&A, &atype, NULL, &ncols,
        &Ap, &Ai, (void **) &Ax, &Ap_size, &Ai_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSC (&A, &atype, &nrows, NULL,
        &Ap, &Ai, (void **) &Ax, &Ap_size, &Ai_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSC (&A, &atype, &nrows, &ncols,
        NULL, &Ai, (void **) &Ax, &Ap_size, &Ai_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSC (&A, &atype, &nrows, &ncols,
        &Ap, NULL, (void **) &Ax, &Ap_size, &Ai_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ai, NULL, &Ap_size, &Ai_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ai, (void **) &Ax, NULL, &Ai_size, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ai, (void **) &Ax, &Ap_size, NULL, &Ax_size, &iso,
        &jumbled, desc)) ;

    ERR (GxB_Matrix_export_CSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ai, (void **) &Ax, &Ap_size, &Ai_size, NULL, &iso,
        &jumbled, desc)) ;



    ERR (GxB_Matrix_export_HyperCSR (NULL, &atype, &nrows, &ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, &Ap_size, &Ah_size, &Aj_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, NULL, &nrows, &ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, &Ap_size, &Ah_size, &Aj_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, NULL, &ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, &Ap_size, &Ah_size, &Aj_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, NULL,
        &Ap, &Ah, &Aj, (void **) &Ax, &Ap_size, &Ah_size, &Aj_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, &ncols,
        NULL, &Ah, &Aj, (void **) &Ax, &Ap_size, &Ah_size, &Aj_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, &ncols,
        &Ap, NULL, &Aj, (void **) &Ax, &Ap_size, &Ah_size, &Aj_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, NULL, (void **) &Ax, &Ap_size, &Ah_size, &Aj_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Aj, NULL, &Ap_size, &Ah_size, &Aj_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, NULL, &Ah_size, &Aj_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, &Ap_size, NULL, &Aj_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, &Ap_size, &Ah_size, NULL, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, &Ap_size, &Ah_size, &Aj_size, NULL, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, &Ap_size, &Ah_size, &Aj_size, &Ax_size, &iso,
        NULL, &jumbled, desc)) ;





    ERR (GxB_Matrix_export_HyperCSC (NULL, &atype, &nrows, &ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, NULL, &nrows, &ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, NULL, &ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, NULL,
        &Ap, &Ah, &Ai, (void **) &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, &ncols,
        NULL, &Ah, &Ai, (void **) &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, &ncols,
        &Ap, NULL, &Ai, (void **) &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, NULL, (void **) &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Ai, NULL, &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, NULL, &Ah_size, &Ai_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, &Ap_size, NULL, &Ai_size, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, &Ap_size, &Ah_size, NULL, &Ax_size, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, &Ap_size, &Ah_size, &Ai_size, NULL, &iso,
        &nvec, &jumbled, desc)) ;

    ERR (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
        NULL, &jumbled, desc)) ;


    OK (GB_Matrix_check (A, "A still OK", G1, NULL)) ;

    OK (GxB_Matrix_export_CSR (&A, &atype, &nrows, &ncols,
        &Ap, &Aj, (void **) &Ax, &Ap_size, &Aj_size, &Ax_size, &iso,
        &jumbled, desc)) ;


    ERR (GxB_Matrix_import_CSR (NULL, atype, nrows, ncols,
        &Ap, &Aj, (void **) &Ax, Ap_size, Aj_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSR (&A, NULL, nrows, ncols,
        &Ap, &Aj, (void **) &Ax, Ap_size, Aj_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSR (&A, atype, nrows, ncols,
        NULL, &Aj, (void **) &Ax, Ap_size, Aj_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSR (&A, atype, nrows, ncols,
        &Ap, NULL, (void **) &Ax, Ap_size, Aj_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSR (&A, atype, nrows, ncols,
        &Ap, &Aj, NULL, Ap_size, Aj_size, Ax_size, iso,
        jumbled, desc)) ;

    expected = GrB_INVALID_VALUE ;

    ERR (GxB_Matrix_import_CSR (&A, atype, INT64_MAX, ncols,
        &Ap, &Aj, (void **) &Ax, Ap_size, Aj_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSR (&A, atype, nrows, INT64_MAX,
        &Ap, &Aj, (void **) &Ax, Ap_size, Aj_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSR (&A, atype, nrows, ncols,
        &Ap, &Aj, (void **) &Ax, Ap_size, Aj_size, INT64_MAX, iso,
        jumbled, desc)) ;


    expected = GrB_NULL_POINTER ;

    OK (GxB_Matrix_import_CSR (&A, atype, nrows, ncols,
        &Ap, &Aj, (void **) &Ax, Ap_size, Aj_size, Ax_size, iso,
        jumbled, desc)) ;

    OK (GB_Matrix_check (A, "A also OK", G1, NULL)) ;

    OK (GxB_Matrix_export_CSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ai, (void **) &Ax, &Ap_size, &Ai_size, &Ax_size, &iso,
        &jumbled, desc)) ;



    ERR (GxB_Matrix_import_CSC (NULL, atype, nrows, ncols,
        &Ap, &Ai, (void **) &Ax, Ap_size, Ai_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSC (&A, atype, nrows, ncols,
        NULL, &Ai, (void **) &Ax, Ap_size, Ai_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSC (&A, atype, nrows, ncols,
        &Ap, NULL, (void **) &Ax, Ap_size, Ai_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSC (&A, atype, nrows, ncols,
        &Ap, &Ai, NULL, Ap_size, Ai_size, Ax_size, iso,
        jumbled, desc)) ;

    expected = GrB_INVALID_VALUE ;

    ERR (GxB_Matrix_import_CSC (&A, atype, INT64_MAX, ncols,
        &Ap, &Ai, (void **) &Ax, Ap_size, Ai_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSC (&A, atype, nrows, INT64_MAX,
        &Ap, &Ai, (void **) &Ax, Ap_size, Ai_size, Ax_size, iso,
        jumbled, desc)) ;

    ERR (GxB_Matrix_import_CSC (&A, atype, nrows, ncols,
        &Ap, &Ai, (void **) &Ax, Ap_size, Ai_size, INT64_MAX, iso,
        jumbled, desc)) ;


    expected = GrB_NULL_POINTER ;

    OK (GxB_Matrix_import_CSC (&A, atype, nrows, ncols,
        &Ap, &Ai, (void **) &Ax, Ap_size, Ai_size, Ax_size, iso,
        jumbled, desc)) ;

    OK (GB_Matrix_check (A, "A here too OK", G1, NULL)) ;

    OK (GxB_Matrix_export_HyperCSR (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, &Ap_size, &Ah_size, &Aj_size, &Ax_size, &iso,
        &nvecs, &jumbled, desc)) ;



    ERR (GxB_Matrix_import_HyperCSR (NULL, atype, nrows, ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, Ap_size, Ah_size, Aj_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSR (&A, NULL, nrows, ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, Ap_size, Ah_size, Aj_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSR (&A, atype, nrows, ncols,
        NULL, &Ah, &Aj, (void **) &Ax, Ap_size, Ah_size, Aj_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSR (&A, atype, nrows, ncols,
        &Ap, NULL, &Aj, (void **) &Ax, Ap_size, Ah_size, Aj_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSR (&A, atype, nrows, ncols,
        &Ap, &Ah, NULL, (void **) &Ax, Ap_size, Ah_size, Aj_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSR (&A, atype, nrows, ncols,
        &Ap, &Ah, &Aj, NULL, Ap_size, Ah_size, Aj_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;


    expected = GrB_INVALID_VALUE ;

    ERR (GxB_Matrix_import_HyperCSR (&A, atype, INT64_MAX, ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, Ap_size, Ah_size, Aj_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSR (&A, atype, nrows, INT64_MAX,
        &Ap, &Ah, &Aj, (void **) &Ax, Ap_size, Ah_size, Aj_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSR (&A, atype, nrows, ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, Ap_size, Ah_size, Aj_size, INT64_MAX, iso,
        nvecs, jumbled, desc)) ;



    expected = GrB_NULL_POINTER ;

    OK (GxB_Matrix_import_HyperCSR (&A, atype, nrows, ncols,
        &Ap, &Ah, &Aj, (void **) &Ax, Ap_size, Ah_size, Aj_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    OK (GB_Matrix_check (A, "A yet still OK", G1, NULL)) ;

    OK (GxB_Matrix_export_HyperCSC (&A, &atype, &nrows, &ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
        &nvecs, &jumbled, desc)) ;



    ERR (GxB_Matrix_import_HyperCSC (NULL, atype, nrows, ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, Ap_size, Ah_size, Ai_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSC (&A, NULL, nrows, ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, Ap_size, Ah_size, Ai_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSC (&A, atype, nrows, ncols,
        NULL, &Ah, &Ai, (void **) &Ax, Ap_size, Ah_size, Ai_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSC (&A, atype, nrows, ncols,
        &Ap, NULL, &Ai, (void **) &Ax, Ap_size, Ah_size, Ai_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSC (&A, atype, nrows, ncols,
        &Ap, &Ah, NULL, (void **) &Ax, Ap_size, Ah_size, Ai_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSC (&A, atype, nrows, ncols,
        &Ap, &Ah, &Ai, NULL, Ap_size, Ah_size, Ai_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    expected = GrB_INVALID_VALUE ;

    ERR (GxB_Matrix_import_HyperCSC (&A, atype, INT64_MAX, ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, Ap_size, Ah_size, Ai_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSC (&A, atype, nrows, INT64_MAX,
        &Ap, &Ah, &Ai, (void **) &Ax, Ap_size, Ah_size, Ai_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    ERR (GxB_Matrix_import_HyperCSC (&A, atype, nrows, ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, Ap_size, Ah_size, Ai_size, INT64_MAX, iso,
        nvecs, jumbled, desc)) ;

    expected = GrB_NULL_POINTER ;

    OK (GxB_Matrix_import_HyperCSC (&A, atype, nrows, ncols,
        &Ap, &Ah, &Ai, (void **) &Ax, Ap_size, Ah_size, Ai_size, Ax_size, iso,
        nvecs, jumbled, desc)) ;

    OK (GB_Matrix_check (A, "A yet again OK", G1, NULL)) ;

    //--------------------------------------------------------------------------
    // vector import/export
    //--------------------------------------------------------------------------

    nvals = 99 ;
    OK (GxB_Vector_fprint (u, "u to import/export", GxB_COMPLETE, stdout)) ;
    GrB_Type utype ;
    OK (GxB_Vector_export_CSC (&u, &utype, &n, &Ai, (void **) &Ax,
        &Ai_size, &Ax_size, &iso,
        &nvals, &jumbled, desc)) ;

    OK (GxB_Type_fprint (utype, "type of u", GxB_COMPLETE, stdout)) ;
    printf ("nvals %llu\n", nvals) ;
    CHECK (nvals != 99) ;
    for (int64_t p = 0 ; p < ((int64_t) nvals) ; p++)
    {
        int64_t i = Ai [p] ;
        double x = Ax [p] ;
        CHECK (i >= 0 && i < n) ;
        printf ("   col %lld value %g\n", Ai [p], Ax [p]) ;
    }
    OK (GxB_Vector_import_CSC (&u, utype, n, &Ai, (void **) &Ax, Ai_size, Ax_size, iso, nvals, jumbled, desc)) ;
    OK (GxB_Vector_fprint (u, "u imported", GxB_COMPLETE, stdout)) ;

    expected = GrB_NULL_POINTER ;

    ERR (GxB_Vector_export_CSC (NULL, &utype, &n, &Ai, (void **) &Ax, &Ai_size, &Ax_size, &iso, &nvals, &jumbled, desc)) ;
    ERR (GxB_Vector_export_CSC (&u, NULL, &n, &Ai, (void **) &Ax, &Ai_size, &Ax_size, &iso, &nvals, &jumbled, desc)) ;
    ERR (GxB_Vector_export_CSC (&u, &utype, NULL, &Ai, (void **) &Ax, &Ai_size, &Ax_size, &iso, &nvals, &jumbled, desc)) ;
    ERR (GxB_Vector_export_CSC (&u, &utype, &n, NULL, (void **) &Ax, &Ai_size, &Ax_size, &iso, &nvals, &jumbled, desc)) ;
    ERR (GxB_Vector_export_CSC (&u, &utype, &n, &Ai, NULL, &Ai_size, &Ax_size, &iso, &nvals, &jumbled, desc)) ;
    ERR (GxB_Vector_export_CSC (&u, &utype, &n, &Ai, (void **) &Ax, NULL, &Ax_size, &iso, &nvals, &jumbled, desc)) ;
    ERR (GxB_Vector_export_CSC (&u, &utype, &n, &Ai, (void **) &Ax, &Ai_size, NULL, &iso, &nvals, &jumbled, desc)) ;
    ERR (GxB_Vector_export_CSC (&u, &utype, &n, &Ai, (void **) &Ax, &Ai_size, &Ax_size, &iso, NULL, &jumbled, desc)) ;
//  ERR (GxB_Vector_export_CSC (&u, &utype, &n, &Ai, (void **) &Ax, &Ai_size, &Ax_size, &iso, &nvals, NULL, desc)) ;

    OK (GB_Vector_check (u, "u still OK", G1, NULL)) ;

    OK (GxB_Vector_export_CSC (&u, &utype, &n, &Ai, (void **) &Ax, &Ai_size, &Ax_size, &iso, &nvals, &jumbled, desc)) ;

    ERR (GxB_Vector_import_CSC (NULL, utype, n, &Ai, (void **) &Ax, Ai_size, Ax_size, iso, nvals, jumbled, desc)) ;
    ERR (GxB_Vector_import_CSC (&u, NULL, n, &Ai, (void **) &Ax, Ai_size, Ax_size, iso, nvals, jumbled, desc)) ;
    ERR (GxB_Vector_import_CSC (&u, utype, n, NULL, (void **) &Ax, Ai_size, Ax_size, iso, nvals, jumbled, desc)) ;
    ERR (GxB_Vector_import_CSC (&u, utype, n, &Ai, NULL, Ai_size, Ax_size, iso, nvals, jumbled, desc)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Vector_import_CSC (&u, utype, INT64_MAX, &Ai, (void **) &Ax, Ai_size, Ax_size, iso, nvals, jumbled, desc)) ;
    ERR (GxB_Vector_import_CSC (&u, utype, n, &Ai, (void **) &Ax, INT64_MAX, Ax_size, iso, nvals, jumbled, desc)) ;

    expected = GrB_NULL_POINTER ;

    OK (GxB_Vector_import_CSC (&u, utype, n, &Ai, (void **) &Ax, Ai_size, Ax_size, iso, nvals, jumbled, desc)) ;

    OK (GB_Vector_check (u, "u still OK", G3, NULL)) ;
}

    //--------------------------------------------------------------------------
    // free all
    //--------------------------------------------------------------------------

    // this is also done by FREE_ALL, but the list is meant to be
    // accurate, so nmalloc should be zero at the check below

    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("\n\nfree all: nmalloc %d\n", nmalloc) ;

    GrB_Matrix_free_(&Empty1) ;       CHECK (Empty1       == NULL) ;
    GrB_Vector_free_(&v) ;            CHECK (v            == NULL) ;
    GrB_Vector_free_(&u) ;            CHECK (u            == NULL) ;
    GrB_Matrix_free_(&A) ;            CHECK (A            == NULL) ;
    GrB_Vector_free_(&u) ;            CHECK (u            == NULL) ;
    GrB_Vector_free_(&z) ;            CHECK (z            == NULL) ;
    GrB_Vector_free_(&h) ;            CHECK (h            == NULL) ;
    GrB_Matrix_free_(&B) ;            CHECK (B            == NULL) ;
    GrB_Matrix_free_(&C) ;            CHECK (C            == NULL) ;
    GrB_Matrix_free_(&E) ;            CHECK (E            == NULL) ;
    GrB_Matrix_free_(&F) ;            CHECK (F            == NULL) ;
    GrB_Matrix_free_(&Z) ;            CHECK (Z            == NULL) ;
    GrB_Matrix_free_(&H) ;            CHECK (H            == NULL) ;
    GrB_Type_free_(&T) ;              CHECK (T            == NULL) ;
    GrB_Matrix_free_(&Acrud) ;        CHECK (Acrud        == NULL) ;
    GrB_Type_free_(&Tcrud) ;          CHECK (Tcrud        == NULL) ;
    GrB_UnaryOp_free_(&op1crud) ;     CHECK (op1crud      == NULL) ;
    GrB_BinaryOp_free_(&op2crud) ;    CHECK (op2crud      == NULL) ;
    GrB_BinaryOp_free_(&op3) ;        CHECK (op3          == NULL) ;
    GrB_UnaryOp_free_(&op1b) ;        CHECK (op1b         == NULL) ;
    GrB_BinaryOp_free_(&op2b) ;       CHECK (op2b         == NULL) ;
    GrB_Semiring_free_(&semiring2) ;  CHECK (semiring2    == NULL) ;
    GrB_Descriptor_free_(&descb) ;    CHECK (descb        == NULL) ;
    GrB_Vector_free_(&vb) ;           CHECK (vb           == NULL) ;
    GrB_Monoid_free_(&monoidb) ;      CHECK (monoidb      == NULL) ;
    GrB_Monoid_free_(&monoid_crud) ;  CHECK (monoid_crud  == NULL) ;
    GrB_Semiring_free_(&semicrud) ;   CHECK (semicrud     == NULL) ;
    GrB_Vector_free_(&vcrud) ;        CHECK (vcrud        == NULL) ;
    GrB_Matrix_free_(&Aempty) ;       CHECK (Aempty       == NULL) ;
    GrB_Vector_free_(&vempty) ;       CHECK (vempty       == NULL) ;
    GrB_Descriptor_free_(&desc) ;     CHECK (desc         == NULL) ;
    GrB_Descriptor_free_(&dtn) ;      CHECK (dtn          == NULL) ;
    GrB_Descriptor_free_(&dnt) ;      CHECK (dnt          == NULL) ;
    GrB_Descriptor_free_(&dtt) ;      CHECK (dtt          == NULL) ;
    GrB_Descriptor_free_(&dcrud) ;    CHECK (dcrud        == NULL) ;

    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("nmalloc %d before complex_finalize\n", nmalloc) ;
    Complex_finalize ( ) ;
    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("nmalloc %d done\n", nmalloc) ;
    GB_mx_at_exit ( ) ;
    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("nmalloc %d all freed\n", nmalloc) ;

    FREE_ALL ;
    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("nmalloc %d all freed\n", nmalloc) ;
    GB_mx_at_exit ( ) ;
    nmalloc = GB_Global_nmalloc_get ( ) ;
    printf ("nmalloc %d after finalize\n", nmalloc) ;
    CHECK (nmalloc == 0) ;

    printf ("\ncheck errlog.txt for errors tested\n") ;
    printf ("All error-handling tests passed"
            " (all errors above were expected)\n") ;
    fprintf (f, "\nAll error-handling tests passed"
            " (all errors above were expected)\n") ;
    fclose (f) ;
    fclose (ff) ;
}

