//------------------------------------------------------------------------------
// GB_mex_test37: index binary op tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

//------------------------------------------------------------------------------
// isequal: ensure two matrices are identical
//------------------------------------------------------------------------------

#undef  FREE_ALL
#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free (&D) ;              \
}

bool isequal (GrB_Matrix C1, GrB_Matrix C2) ;
bool isequal (GrB_Matrix C1, GrB_Matrix C2)
{
    GrB_Info info = GrB_SUCCESS ;
    GrB_Matrix D = NULL ;
    // finish any pending work
    OK (GrB_Matrix_wait (C1, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (C2, GrB_MATERIALIZE)) ;
    // ensure C2 has the same sparsity and row/col storage as C1
    int32_t s ;
    OK (GrB_Matrix_get_INT32 (C1, &s, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Matrix_set_INT32 (C2,  s, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Matrix_get_INT32 (C1, &s, GxB_SPARSITY_STATUS)) ;
    OK (GrB_Matrix_set_INT32 (C2,  s, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_wait (C1, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (C2, GrB_MATERIALIZE)) ;
    // check if C1 and C2 are equal
    bool ok = GB_mx_isequal (C1, C2, 0) ;
    if (!ok)
    {
        printf ("\n=========================================\n") ;
        printf ("matrices differ!\n") ;
        printf ("\n=========================================\n") ;
        uint64_t nvals = 0, nrows = 0, ncols = 0 ;
        OK (GrB_Matrix_nrows (&nrows, C1)) ;
        OK (GrB_Matrix_ncols (&ncols, C1)) ;
        OK (GrB_Matrix_new (&D, GrB_FP64, nrows, ncols)) ;
        OK (GrB_Matrix_eWiseAdd_BinaryOp (D, NULL, NULL, GrB_MINUS_FP64,
            C1, C2, NULL)) ;
        OK (GrB_Matrix_select_FP64 (D, NULL, NULL, GrB_VALUENE_FP64, D,
            (double) 0, NULL)) ;
        OK (GrB_Matrix_nvals (&nvals, D)) ;
        OK (GxB_print (D, 5)) ;
        OK (GrB_Matrix_free (&D)) ;
    }
    return (ok) ;
}

//------------------------------------------------------------------------------
// gb_test37_idxbinop
//------------------------------------------------------------------------------

void gb_test37_idxbinop (double *z,
    const double *x, uint64_t ix, uint64_t jx,
    const double *y, uint64_t iy, uint64_t jy,
    const double *theta) ;

void gb_test37_idxbinop (double *z,
    const double *x, uint64_t ix, uint64_t jx,
    const double *y, uint64_t iy, uint64_t jy,
    const double *theta)
{
    (*z) = (*x) + 2*(*y) - 42*ix + jx + 3*iy + 1000*jy - (*theta) ;
}

#define TEST37_IDXBINOP_DEFN                                                \
"void gb_test37_idxbinop (double *z,                                       \n" \
"    const double *x, uint64_t ix, uint64_t jx,                         \n" \
"    const double *y, uint64_t iy, uint64_t jy,                         \n" \
"    const double *theta)                                               \n" \
"{                                                                      \n" \
"    (*z) = (*x) + 2*(*y) - 42*ix + jx + 3*iy + 1000*jy - (*theta) ;    \n" \
"}                                                                      \n"

//------------------------------------------------------------------------------
// ewise: compute the result without using GraphBLAS
//------------------------------------------------------------------------------

// C0 = add (A,A')
// B0 = union (A,A')
// E0 = emult (A,A')
// G0<M> = emult (A,A')

#define FREE_WORK                                           \
{                                                           \
    if (Ab != NULL) { free_function (Ab) ; } ; Ab = NULL ;  \
    if (Ax != NULL) { free_function (Ax) ; } ; Ax = NULL ;  \
    if (Bb != NULL) { free_function (Bb) ; } ; Bb = NULL ;  \
    if (Bx != NULL) { free_function (Bx) ; } ; Bx = NULL ;  \
    GrB_Matrix_free (&a) ;                                  \
    GrB_Matrix_free (&b) ;                                  \
    GrB_Matrix_free (&T) ;                                  \
}

#undef  FREE_ALL
#define FREE_ALL                        \
{                                       \
    FREE_WORK ;                         \
    GrB_Matrix_free (&C) ;              \
}

GrB_Info ewise
(
    GrB_Matrix *C_handle,
    GrB_Matrix A,
    GrB_Matrix M,
    double *alpha,
    double *beta,
    double *theta,
    int kind
) ;

GrB_Info ewise
(
    GrB_Matrix *C_handle,
    GrB_Matrix A,
    GrB_Matrix M,
    double *alpha,
    double *beta,
    double *theta,
    int kind
)
{
    GrB_Info info = GrB_SUCCESS ;
    int8_t *Ab = NULL, *Bb = NULL, *Tb = NULL ;
    double *Ax = NULL, *Bx = NULL, *Tx = NULL ;
    GrB_Matrix T = NULL, C = NULL, a = NULL, b = NULL ;
    uint64_t Ab_size = 0, Ax_size = 0, A_nvals = 0,
             Bb_size = 0, Bx_size = 0, B_nvals = 0,
             Tb_size = 0, Tx_size = 0, T_nvals = 0 ;
    void (* free_function) (void *) = NULL ;
    uint64_t n = 0 ;
    (*C_handle) = NULL ;

    //--------------------------------------------------------------------------
    // get the current free function
    //--------------------------------------------------------------------------

    free_function = GB_Global_free_function_get ( ) ;

    //--------------------------------------------------------------------------
    // create bitmap format of A, A', and T
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_nrows (&n, A)) ;

    // a = A
    OK (GrB_Matrix_dup (&a, A)) ;

    // b = A'
    OK (GrB_Matrix_dup (&b, A)) ;
    OK (GrB_transpose (b, NULL, NULL, b, NULL)) ;

    // extract a in bitmap CSC format
    OK (GxB_Matrix_unpack_BitmapC (a, &Ab, (void **) &Ax, &Ab_size, &Ax_size,
        NULL, &A_nvals, NULL)) ;
    GrB_Matrix_free (&a) ;

    // extract b in bitmap CSC format
    OK (GxB_Matrix_unpack_BitmapC (b, &Bb, (void **) &Bx, &Bb_size, &Bx_size,
        NULL, &B_nvals, NULL)) ;
    GrB_Matrix_free (&b) ;

    // create T and extract in bitmap CSC format
    OK (GrB_Matrix_new (&T, GrB_FP64, n, n)) ;
    OK (GxB_Matrix_unpack_BitmapC (T, &Tb, (void **) &Tx, &Tb_size, &Tx_size,
        NULL, &T_nvals, NULL)) ;

    //--------------------------------------------------------------------------
    // t = op (a,b,theta)
    //--------------------------------------------------------------------------

    // 0: C0 = add (A,A')
    // 1: B0 = union (A,A')
    // 2: E0 = emult (A,A')
    // 3: G0<M> = emult (A,A')

    T_nvals = 0 ;

    for (uint64_t i = 0 ; i < n ; i++)
    {
        for (uint64_t j = 0 ; j < n ; j++)
        {
            int64_t p = i + j*n ;

            int8_t ab = Ab [p] ;
            int8_t bb = Bb [p] ;
            int8_t tb = 0 ;
            double ax = Ax [p] ;
            double bx = Bx [p] ;
            double tx = 0 ;

            if (ab && bb)
            {
                // both A(i,j) and B(i,j) are present: apply the operator
                gb_test37_idxbinop (&tx, &ax, i, j, &bx, i, j, theta) ;
                tb = 1 ;
            }
            else if (ab && !bb)
            {
                // A(i,j) is present but B(i,j) is not
                switch (kind)
                {
                    case 0 :    // add
                        tx = ax ;
                        tb = 1 ;
                        break ;
                    case 1 :    // union
                        gb_test37_idxbinop (&tx, &ax, i, j, beta, i, j, theta) ;
                        tb = 1 ;
                        break ;
                    default :   // emult
                        break ;
                }
            }
            else if (!ab && bb)
            {
                // B(i,j) is present but A(i,j) is not
                switch (kind)
                {
                    case 0 :    // add
                        tx = bx ;
                        tb = 1 ;
                        break ;
                    case 1 :    // union
                        gb_test37_idxbinop (&tx, alpha, i, j, &bx, i, j, theta) ;
                        tb = 1 ;
                        break ;
                    default:
                        break ;
                }
            }

            // save the result in T(i,j)
            Tx [p] = tx ;
            Tb [p] = tb ;
            T_nvals += tb ;
        }
    }

    // pack T in bitmap CSC format
    OK (GxB_Matrix_pack_BitmapC (T, &Tb, (void **) &Tx, Tb_size, Tx_size,
        false, T_nvals, NULL)) ;

    //--------------------------------------------------------------------------
    // create C
    //--------------------------------------------------------------------------

    if (kind == 3)
    {
        // C<M> = T
        OK (GrB_Matrix_new (&C, GrB_FP64, n, n)) ;
        OK (GrB_Matrix_assign_(C, M, NULL, T, GrB_ALL, n, GrB_ALL, n,
            GrB_DESC_R)) ;
    }
    else
    {
        C = T ;
        T = NULL ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*C_handle) = C ;
    FREE_WORK ;
    return (GrB_SUCCESS) ;
}

#undef FREE_WORK

//------------------------------------------------------------------------------

#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

#undef  FREE_ALL
#define FREE_ALL                        \
{                                       \
    GrB_Scalar_free (&Theta) ;          \
    GrB_Scalar_free (&Alpha) ;          \
    GrB_Scalar_free (&Beta) ;           \
    GrB_Scalar_free (&Crud_Scalar) ;    \
    GrB_Type_free (&Crud_Type) ;        \
    GrB_Matrix_free (&A) ;              \
    GrB_Matrix_free (&M) ;              \
    GrB_Matrix_free (&A2) ;             \
    GrB_Matrix_free (&C1) ;             \
    GrB_Matrix_free (&C2) ;             \
    GrB_Matrix_free (&B1) ;             \
    GrB_Matrix_free (&B2) ;             \
    GrB_Matrix_free (&E1) ;             \
    GrB_Matrix_free (&E2) ;             \
    GrB_Matrix_free (&F1) ;             \
    GrB_Matrix_free (&F2) ;             \
    GrB_Matrix_free (&G1) ;             \
    GrB_Matrix_free (&G2) ;             \
    GrB_Matrix_free (&C0) ;             \
    GrB_Matrix_free (&B0) ;             \
    GrB_Matrix_free (&E0) ;             \
    GrB_Matrix_free (&G0) ;             \
    GrB_BinaryOp_free (&Bop) ;          \
    GxB_IndexBinaryOp_free (&Iop) ;     \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // create index binary ops and test matrices
    //--------------------------------------------------------------------------

    GrB_Type Crud_Type = NULL ;
    GrB_Scalar Theta = NULL, Alpha = NULL, Beta = NULL, Crud_Scalar ;
    GxB_IndexBinaryOp Iop = NULL, Crud_Iop = NULL ;
    GrB_BinaryOp Bop = NULL, Crud_Bop = NULL ;
    GrB_Matrix A = NULL, C1 = NULL, C2 = NULL, B1 = NULL, B2 = NULL,
        E1 = NULL, E2 = NULL, A2 = NULL, F1 = NULL, F2 = NULL, M = NULL,
        G1 = NULL, G2 = NULL, C0 = NULL, B0 = NULL, E0 = NULL, G0 = NULL ;

    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 10)) ;

    OK (GrB_Matrix_new (&C1, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&C2, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&B1, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&B2, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&E1, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&E2, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&F1, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&F2, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&G1, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&G2, GrB_FP64, 10, 10)) ;

    // C1 and B1 always stay by column
    OK (GrB_Matrix_set_INT32 (C1, GrB_COLMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Matrix_set_INT32 (B1, GrB_COLMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;

    double x = 1 ;
    for (int64_t i = 0 ; i < 9 ; i++)
    {
        OK (GrB_Matrix_setElement_FP64 (A, x, i, i)) ;
        x = x*1.2 ;
        OK (GrB_Matrix_setElement_FP64 (A, x, i, i+1)) ;
        x = x*1.2 ;
        OK (GrB_Matrix_setElement_FP64 (A, x, i+1, i)) ;
        x = x*1.2 ;
    }
    OK (GrB_Matrix_setElement_FP64 (A, x, 9, 9)) ;
    x = x - 1000 ;
    OK (GrB_Matrix_setElement_FP64 (A, x, 5, 2)) ;

    double theta = x ;
    OK (GrB_Scalar_new (&Theta, GrB_FP64)) ;
    OK (GrB_Scalar_setElement_FP64 (Theta, theta)) ;

    OK (GxB_IndexBinaryOp_new (&Iop,
        (GxB_index_binary_function) gb_test37_idxbinop,
        GrB_FP64, GrB_FP64, GrB_FP64, GrB_FP64,
        "gb_test37_idxbinop", TEST37_IDXBINOP_DEFN)) ;

    OK (GxB_IndexBinaryOp_set_String (Iop, "test37 idx binop", GrB_NAME)) ;
    OK (GxB_print (Iop, 5)) ;

    size_t theta_type_namelen = 0 ;
    OK (GxB_IndexBinaryOp_get_SIZE (Iop, &theta_type_namelen,
        GxB_THETA_TYPE_STRING)) ;
    printf ("theta name length: %d\n", (int) theta_type_namelen) ;
    CHECK (theta_type_namelen == strlen ("GrB_FP64") + 1) ;

    char theta_type_name [256] ;
    theta_type_name [0] = '\0' ;
    OK (GxB_IndexBinaryOp_get_String (Iop, theta_type_name,
        GxB_THETA_TYPE_STRING)) ;
    CHECK (strcmp (theta_type_name, "GrB_FP64") == 0)  ;

    int32_t theta_type_code = -1 ;
    OK (GxB_IndexBinaryOp_get_INT32 (Iop, &theta_type_code,
        GxB_THETA_TYPE_CODE)) ;
    CHECK (theta_type_code == GrB_FP64_CODE) ;

    OK (GrB_BinaryOp_get_INT32 (GxB_FIRSTI1_INT32, &theta_type_code,
        GxB_THETA_TYPE_CODE)) ;
    CHECK (theta_type_code == GrB_INT32_CODE) ;

    OK (GrB_BinaryOp_get_INT32 (GxB_FIRSTI1_INT64, &theta_type_code,
        GxB_THETA_TYPE_CODE)) ;
    CHECK (theta_type_code == GrB_INT64_CODE) ;

    OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
    OK (GxB_print (Bop, 5)) ;

    OK (GrB_Scalar_new (&Alpha, GrB_FP64)) ;

    double y = 0 ;
    int expected = GrB_INVALID_VALUE ;
    ERR (GxB_IndexBinaryOp_get_Scalar (Iop, Alpha, GxB_THETA)) ;

    y = 0 ;
    OK (GrB_Scalar_clear (Alpha)) ;
    OK (GrB_BinaryOp_get_Scalar (Bop, Alpha, GxB_THETA)) ;
    OK (GrB_Scalar_extractElement_FP64 (&y, Alpha)) ;
    CHECK (y == theta) ;

    theta_type_code = -1 ;
    OK (GrB_BinaryOp_get_INT32 (Bop, &theta_type_code,
        GxB_THETA_TYPE_CODE)) ;
    CHECK (theta_type_code == GrB_FP64_CODE) ;

    theta_type_namelen = 0 ;
    OK (GrB_BinaryOp_get_SIZE (Bop, &theta_type_namelen,
        GxB_THETA_TYPE_STRING)) ;
    CHECK (theta_type_namelen == strlen ("GrB_FP64") + 1) ;

    theta_type_name [0] = '\0' ;
    OK (GrB_BinaryOp_get_String (Bop, theta_type_name,
        GxB_THETA_TYPE_STRING)) ;
    CHECK (strcmp (theta_type_name, "GrB_FP64") == 0)  ;

    double alpha = 3.14159 ;
    double beta = 42 ;
    OK (GrB_Scalar_new (&Beta, GrB_FP64)) ;
    OK (GrB_Scalar_setElement_FP64 (Alpha, alpha)) ;
    OK (GrB_Scalar_setElement_FP64 (Beta,  beta)) ;

    OK (GrB_Matrix_dup (&A2, A)) ;
    OK (GrB_Matrix_dup (&M, A)) ;

    OK (GrB_Matrix_set_INT32 (M, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;

    //--------------------------------------------------------------------------
    // create the expected results
    //--------------------------------------------------------------------------

    OK (ewise (&C0, A, NULL, NULL,   NULL,  &theta, 0)) ; // C0 = add(A,A')
    OK (ewise (&B0, A, NULL, &alpha, &beta, &theta, 1)) ; // B0 = union(A,A')
    OK (ewise (&E0, A, NULL, NULL,   NULL,  &theta, 2)) ; // E0 = emult(A,A')
    OK (ewise (&G0, A, M,    NULL,   NULL,  &theta, 3)) ; // G0<M> = emult(A,A')

    //--------------------------------------------------------------------------
    // test index binary ops
    //--------------------------------------------------------------------------

    for (int a1_sparsity = 0 ; a1_sparsity <= 1 ; a1_sparsity++)
    {
        for (int a2_sparsity = 0 ; a2_sparsity <= 1 ; a2_sparsity++)
        {
            for (int a1_store = 0 ; a1_store <= 1 ; a1_store++)
            {
                for (int a2_store = 0 ; a2_store <= 1 ; a2_store++)
                {
                    for (int c2_store = 0 ; c2_store <= 1 ; c2_store++)
                    {
                        for (int b2_store = 0 ; b2_store <= 1 ; b2_store++)
                        {
                            for (int jit = 0 ; jit <= 1 ; jit++)
                            {

                                printf (".") ;

                                // turn on/off the JIT
                                OK (GrB_Global_set_INT32 (GrB_GLOBAL,
                                    jit ? GxB_JIT_ON : GxB_JIT_OFF,
                                    GxB_JIT_C_CONTROL)) ;

                                // change A sparsity
                                OK (GrB_Matrix_set_INT32 (A,
                                    a1_sparsity ? GxB_SPARSE : GxB_BITMAP,
                                    GxB_SPARSITY_CONTROL)) ;

                                // change A storage orientation
                                OK (GrB_Matrix_set_INT32 (A,
                                    a1_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;

                                // C1 = add (A, A')
                                OK (GrB_Matrix_eWiseAdd_BinaryOp (C1,
                                    NULL, NULL, Bop, A, A, GrB_DESC_T1)) ;
                                // B1 = union (A, A')
                                OK (GxB_Matrix_eWiseUnion (B1, NULL, NULL, Bop,
                                    A, Alpha, A, Beta, GrB_DESC_T1)) ;
                                // E1 = emult (A, A')
                                OK (GrB_Matrix_eWiseMult_BinaryOp (E1,
                                    NULL, NULL, Bop, A, A, GrB_DESC_T1)) ;
                                // F1 = emult (A, A')
                                OK (GrB_Matrix_eWiseMult_BinaryOp (F1,
                                    NULL, NULL, Bop, A, A2, GrB_DESC_T1)) ;
                                // G1<M> = emult (A, A')
                                OK (GrB_Matrix_eWiseMult_BinaryOp (G1,
                                    M, NULL, Bop, A, A2, GrB_DESC_RT1)) ;

                                // change A sparsity again
                                OK (GrB_Matrix_set_INT32 (A2,
                                    a2_sparsity ? GxB_SPARSE : GxB_BITMAP,
                                    GxB_SPARSITY_CONTROL)) ;

                                // change A storage again
                                OK (GrB_Matrix_set_INT32 (A,
                                    a2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;

                                // change C2, etc storage
                                OK (GrB_Matrix_set_INT32 (C2,
                                    c2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;
                                OK (GrB_Matrix_set_INT32 (B2,
                                    b2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;
                                OK (GrB_Matrix_set_INT32 (E2,
                                    b2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;
                                OK (GrB_Matrix_set_INT32 (F2,
                                    b2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;
                                OK (GrB_Matrix_set_INT32 (G2,
                                    b2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;

                                // C2 = add (A, A')
                                OK (GrB_Matrix_eWiseAdd_BinaryOp (C2,
                                    NULL, NULL, Bop, A, A, GrB_DESC_T1)) ;
                                // B2 = union (A, A')
                                OK (GxB_Matrix_eWiseUnion (B2, NULL, NULL,
                                    Bop, A, Alpha, A, Beta, GrB_DESC_T1)) ;
                                // E2 = emult (A, A')
                                OK (GrB_Matrix_eWiseMult_BinaryOp (E2,
                                    NULL, NULL, Bop, A, A, GrB_DESC_T1)) ;
                                // F2 = emult (A, A2')
                                OK (GrB_Matrix_eWiseMult_BinaryOp (F2,
                                    NULL, NULL, Bop, A, A2, GrB_DESC_T1)) ;
                                // G2<M> = emult (A, A2')
                                OK (GrB_Matrix_eWiseMult_BinaryOp (G2,
                                    M, NULL, Bop, A, A2, GrB_DESC_RT1)) ;

                                CHECK (isequal (C1, C2)) ;
                                CHECK (isequal (B1, B2)) ;
                                CHECK (isequal (E1, E2)) ;
                                CHECK (isequal (F1, F2)) ;
                                CHECK (isequal (F1, E2)) ;
                                CHECK (isequal (G1, G2)) ;

                                CHECK (isequal (C1, C0)) ;
                                CHECK (isequal (B1, B0)) ;
                                CHECK (isequal (E1, E0)) ;
                                CHECK (isequal (G1, G0)) ;
                            }
                        }
                    }
                }
            }
        }
    }

    //------------------------------------------------------------------------
    // error tests
    //------------------------------------------------------------------------

    // turn on the JIT
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL)) ;

    int save_jit = 0, save_burble = 0 ;
    OK (GxB_get (GxB_JIT_C_CONTROL, &save_jit)) ;
    CHECK (save_jit == GxB_JIT_ON) ;

    printf ("\nerror handling tests: JIT is %d\n", save_jit) ;

    expected = GrB_INVALID_OBJECT ;
    void *p = Bop->theta_type ;
    Bop->theta_type = NULL ;
    ERR (GB_BinaryOp_check (Bop, "Bop: bad theta_type", 5, stdout)) ;
    Bop->theta_type = p ;

    p = Iop->idxbinop_function ;
    Iop->idxbinop_function = NULL ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: null function", 5, stdout)) ;
    Iop->idxbinop_function = p ;

    p = Iop->ztype ;
    Iop->ztype = NULL ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: null ztype", 5, stdout)) ;
    Iop->ztype = p ;

    p = Iop->xtype ;
    Iop->xtype = NULL ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: null xtype", 5, stdout)) ;
    Iop->xtype = p ;

    p = Iop->ytype ;
    Iop->ytype = NULL ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: null ytype", 5, stdout)) ;
    Iop->ytype = p ;

    p = Iop->theta_type ;
    Iop->theta_type = NULL ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: null theta_type", 5, stdout)) ;
    Iop->theta_type = p ;

    GB_Opcode code = Iop->opcode ;
    Iop->opcode = GB_PLUS_binop_code ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: invalid opcode", 5, stdout)) ;
    Iop->opcode = code ;

    int len = Iop->name_len ;
    Iop->name_len = 3 ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: invalid name_len", 5, stdout)) ;
    Iop->name_len = len ;

    expected = GrB_NULL_POINTER ;
    ERR (GB_IndexBinaryOp_check (NULL, "Iop: null", 5, stdout)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_IndexBinaryOp_set_Scalar (Iop, Theta, GrB_NAME)) ;
    ERR (GxB_IndexBinaryOp_set_INT32 (Iop, 2, GrB_SIZE)) ;
    ERR (GxB_IndexBinaryOp_set_VOID (Iop, NULL, GrB_SIZE, 0)) ;

    expected = GrB_DOMAIN_MISMATCH ;
    OK (GrB_Type_new (&Crud_Type, 4)) ;
    OK (GrB_Scalar_new (&Crud_Scalar, Crud_Type)) ;
    ERR (GxB_BinaryOp_new_IndexOp (&Crud_Bop, Iop, Crud_Scalar)) ;
    ERR (GrB_Matrix_apply (A, NULL, NULL, (GrB_UnaryOp) Bop, A, NULL)) ;

    //------------------------------------------------------------------------
    // JIT testing
    //------------------------------------------------------------------------

    printf ("\n\n-------------- lots of compiler errors expected here:\n") ;

    #define CRUD_IDXBINOP_DEFN                          \
    "void gb_crud_idxbinop (double *z, "                \
    " const double *x, uint64_t ix, uint64_t jx, "      \
    " const double *y, uint64_t iy, uint64_t jy, "      \
    " const double *theta) "                            \
    "{ "                                                \
    "    compiler error occurs here "                   \
    "}"

    printf ("-------- test JIT compiler error:\n") ;

    // turn on the JIT and the burble
    OK (GxB_get (GxB_JIT_C_CONTROL, &save_jit)) ;
    OK (GxB_get (GxB_BURBLE, &save_burble)) ;
    OK (GxB_set (GxB_BURBLE, true)) ;
    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_OFF)) ;
    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;

    expected = GxB_JIT_ERROR ;
    ERR (GxB_IndexBinaryOp_new (&Crud_Iop, NULL,
        GrB_FP64, GrB_FP64, GrB_FP64, GrB_FP64,
        "gb_crud_idxbinop", CRUD_IDXBINOP_DEFN)) ;

    // restore the JIT control and the burble
    OK (GxB_set (GxB_JIT_C_CONTROL, save_jit)) ;
    OK (GxB_set (GxB_BURBLE, save_burble)) ;
    printf ("\n-------- lots of JIT compiler errors expected above\n") ;

    //------------------------------------------------------------------------
    // finalize GraphBLAS
    //------------------------------------------------------------------------

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test37:  all tests passed\n\n") ;
}

