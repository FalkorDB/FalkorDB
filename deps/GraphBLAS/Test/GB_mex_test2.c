//------------------------------------------------------------------------------
// GB_mex_test2: more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Test lots of random stuff.  The function otherwise serves no purpose.

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_ij.h"

#define USAGE "GB_mex_test2"

typedef struct
{
    int stuff [16] ;
}
wild ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix A = NULL, B = NULL, C = NULL ;
    GrB_Scalar scalar = NULL ;
    GrB_Vector victor = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_Type Wild = NULL ;
    const char *err ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    FILE *f = fopen ("errlog3.txt", "w") ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // test removeElement/setElement when jumbled
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_INT32, 10, 10)) ;
    OK (GrB_Vector_new (&victor, GrB_INT32, 10)) ;
    OK (GxB_Vector_Option_set_(victor, GxB_BITMAP_SWITCH, 2.0)) ;
    OK (GrB_Scalar_new (&scalar, GrB_INT32)) ;

    OK (GxB_Matrix_fprint (A, "A before set", 3, NULL)) ;
    OK (GrB_Matrix_setElement_INT32 (A, 314159, 0, 0)) ;
    OK (GxB_Matrix_fprint (A, "A after set", 3, NULL)) ;
    A->jumbled = true ;
    OK (GrB_Matrix_removeElement (A, 0, 0)) ;
    OK (GxB_Matrix_fprint (A, "A after remove", 3, NULL)) ;
    A->jumbled = true ;
    OK (GrB_Matrix_setElement_INT32 (A, 99099, 0, 0)) ;
    OK (GxB_Matrix_fprint (A, "A after set again", 3, NULL)) ;

    OK (GxB_Vector_fprint (victor, "victor before set", 3, NULL)) ;
    OK (GrB_Vector_setElement_INT32 (victor, 44, 0)) ;
    OK (GxB_Vector_fprint (victor, "victor after set", 3, NULL)) ;
    victor->jumbled = true ;
    OK (GrB_Vector_removeElement (victor, 0)) ;
    OK (GxB_Vector_fprint (victor, "victor remove set", 3, NULL)) ;
    victor->jumbled = true ;
    OK (GrB_Vector_setElement_INT32 (victor, 88, 0)) ;
    OK (GxB_Vector_fprint (victor, "victor after set again", 3, NULL)) ;

    OK (GxB_Scalar_fprint (scalar, "scalar before set", 3, NULL)) ;
    OK (GrB_Scalar_setElement_INT32 (scalar, 404)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar after set", 3, NULL)) ;
    int i = 0 ;
    OK (GrB_Scalar_extractElement_INT32 (&i, scalar)) ;
    CHECK (i == 404) ;
    OK (GxB_Scalar_fprint (scalar, "scalar after extract", 3, NULL)) ;
    OK (GrB_Matrix_removeElement ((GrB_Matrix) scalar, 0, 0)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar after remove", 3, NULL)) ;
    i = 777 ;
    expected = GrB_NO_VALUE ;
    ERR (GrB_Scalar_extractElement_INT32 (&i, scalar)) ;
    CHECK (i == 777) ;

    // force a zombie into the scalar
    OK (GrB_Scalar_setElement_INT32 (scalar, 707)) ;
    OK (GrB_Scalar_wait (scalar, GrB_MATERIALIZE)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar after wait", 3, NULL)) ;
    OK (GxB_Matrix_Option_set ((GrB_Matrix) scalar, GxB_SPARSITY_CONTROL,
        GxB_SPARSE)) ;
    CHECK (scalar->i != NULL) ;
    scalar->i [0] = GB_FLIP (0) ;
    scalar->nzombies = 1 ;
    OK (GxB_Scalar_fprint (scalar, "scalar with zombie", 3, NULL)) ;
    expected = GrB_NO_VALUE ;
    ERR (GrB_Scalar_extractElement_INT32 (&i, scalar)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar after extract", 3, NULL)) ;
    CHECK (i == 777) ;

    GrB_Vector_free_(&victor) ;
    GrB_Matrix_free_(&A) ;
    GrB_Scalar_free_(&scalar) ;

    //--------------------------------------------------------------------------
    // builtin comparators not defined for complex types
    //--------------------------------------------------------------------------

    int n = 10 ;
    OK (GrB_Matrix_new (&A, GxB_FC32, n, n)) ;
    OK (GxB_Matrix_Option_set_INT32 (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;

    OK (GrB_Matrix_new (&C, GxB_FC32, n, n)) ;
    OK (GrB_Scalar_new (&scalar, GxB_FC32)) ;
//  expected = GrB_DOMAIN_MISMATCH ;
    expected = GrB_NOT_IMPLEMENTED ;
    ERR (GxB_Matrix_select (C, NULL, NULL, GxB_LT_THUNK, A, scalar, NULL)) ;
    const char *message = NULL ;
    OK (GrB_Matrix_error (&message, C)) ;
    printf ("expected error: %s\n", message) ;
    GrB_Matrix_free_(&C) ;
    GrB_Scalar_free_(&scalar) ;

    //--------------------------------------------------------------------------
    // GB_pslice
    //--------------------------------------------------------------------------

    int64_t Slice [30] ;
    GB_pslice (Slice, A->p, n, 2, true) ;
    CHECK (Slice [0] == 0) ;

    int64_t Ap [11] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } ;
    GB_pslice (Slice, Ap, 10, 10, false) ;
    printf ("Slice: ") ;
    for (int k = 0 ; k <= 10 ; k++) printf (" %ld", Slice [k]) ;
    printf ("\n") ;

    GrB_Matrix_free_(&A) ;

    //--------------------------------------------------------------------------
    // GrB_Matrix_check
    //--------------------------------------------------------------------------

    double bswitch = 1 ;
    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    OK (GxB_Matrix_Option_set_FP64 (A, GxB_BITMAP_SWITCH, 0.125)) ;
    OK (GxB_Matrix_Option_get_(A, GxB_BITMAP_SWITCH, &bswitch)) ;
    CHECK (fabs (bswitch - 0.125) < 1e-5) ;
    bswitch = 1 ;
    OK (GxB_Matrix_Option_get_FP64 (A, GxB_BITMAP_SWITCH, &bswitch)) ;
    CHECK (fabs (bswitch - 0.125) < 1e-5) ;

    OK (GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GrB_Matrix_assign_INT32 (A, NULL, NULL, 3, GrB_ALL, n, GrB_ALL, n,
        NULL)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    OK (GxB_Matrix_fprint (A, "valid matrix", GxB_SHORT, NULL)) ;
    // mangle the matrix
    GB_FREE (&(A->p), A->p_size) ;
    GB_FREE (&(A->x), A->x_size) ;  // OK
    expected = GrB_INVALID_OBJECT ;
    ERR (GxB_Matrix_fprint (A, "invalid sparse matrix", GxB_SHORT, NULL)) ;
    GrB_Matrix_free_(&A) ;

    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    A->sparsity_control = 999 ;
    ERR (GxB_Matrix_fprint (A, "invalid sparsity control", GxB_SHORT, NULL)) ;
    GrB_Matrix_free_(&A) ;

    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    OK (GrB_Matrix_assign_INT32 (A, NULL, NULL, 3, GrB_ALL, n, GrB_ALL, n,
        NULL)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;

    A->jumbled = true ;
    ERR (GxB_Matrix_fprint (A, "full matrix cannot be jumbled", GxB_SHORT,
        NULL)) ;

    A->jumbled = false ;
    A->plen = 999 ;
    ERR (GxB_Matrix_fprint (A, "invalid full matrix", GxB_SHORT, NULL)) ;

    A->plen = -1 ;
    A->nzombies = 1 ;
    ERR (GxB_Matrix_fprint (A, "full matrix cannot have zombies",
        GxB_SHORT, NULL)) ;
    A->nzombies = 0 ;
    CHECK (GB_Pending_alloc (&(A->Pending), false, GrB_INT32, NULL, true, 4)) ;
    ERR (GxB_Matrix_fprint (A, "full matrix cannot have pending tuples",
        GxB_SHORT, NULL)) ;
    GrB_Matrix_free_(&A) ;

    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    OK (GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    A->plen = 999 ;
    ERR (GxB_Matrix_fprint (A, "invalid bitmap", GxB_SHORT, NULL)) ;

    A->plen = -1 ;
    A->b [0] = 1 ;
    ERR (GxB_Matrix_fprint (A, "invalid bitmap", GxB_SUMMARY, NULL)) ;
    GrB_Matrix_free_(&A) ;

    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    OK (GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    OK (GrB_Matrix_setElement_INT32 (A, 12345, 0, 0)) ;
    OK (GxB_Matrix_fprint (A, "valid matrix", GxB_SHORT, NULL)) ;
    A->b [0] = 3 ;
    ERR (GxB_Matrix_fprint (A, "invalid bitmap", GxB_SHORT, NULL)) ;
    GrB_Matrix_free_(&A) ;

    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    A->nvec_nonempty = 2 ;
    ERR (GxB_Matrix_fprint (A, "invalid nvec_nonempty", GxB_SHORT, NULL)) ;
    GrB_Matrix_free_(&A) ;

    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    OK (GrB_Matrix_setElement_INT32 (A, 12345, 0, 0)) ;
    OK (GxB_Matrix_fprint (A, "valid matrix with 1 pending", GxB_SHORT, NULL)) ;
    A->Pending->size = 900 ;
    ERR (GxB_Matrix_fprint (A, "invalid pending type", GxB_SHORT, NULL)) ;
    GrB_Matrix_free_(&A) ;

    //--------------------------------------------------------------------------
    // lo:stride:hi with stride of zero
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_set_(GxB_BURBLE, true)) ;
    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    GrB_Index I [3] = { 1, 1, 0 } ;
    int32_t I32 [3] = { 1, 1, 0 } ;
    OK (GrB_Matrix_new (&C, GrB_INT32, n, 0)) ;
    OK (GrB_Matrix_extract_(C, NULL, NULL, A, GrB_ALL, n, I, GxB_STRIDE,
        NULL)) ;
    OK (GxB_Matrix_fprint (C, "C = A (:,1:0:1)", GxB_COMPLETE, NULL)) ;
    GrB_Matrix_free_(&C) ;
    OK (GrB_Matrix_new (&C, GrB_INT32, 0, n)) ;
    OK (GrB_Matrix_extract_(C, NULL, NULL, A, I, GxB_STRIDE, GrB_ALL, n,
        NULL)) ;
    OK (GxB_Matrix_fprint (C, "C = A (1:0:1,:)", GxB_COMPLETE, NULL)) ;
    GrB_Matrix_free_(&C) ;
    GrB_Matrix_free_(&A) ;
    OK (GxB_Global_Option_set_(GxB_BURBLE, false)) ;

    int64_t Icolon [3] = { 1, 1, 0 } ;
    CHECK (!GB_ij_is_in_list (NULL, 0, 0, GB_STRIDE, Icolon)) ;

    //--------------------------------------------------------------------------
    // GB_any_aliased
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    OK (GrB_Matrix_setElement_INT32 (A, 12345, 0, 0)) ;
    OK (GrB_Matrix_dup (&C, A)) ;
    CHECK (!GB_any_aliased (A, C)) ;
    GB_FREE (&(C->p), C->p_size) ;
    C->p = A->p ;
    C->p_shallow = true ;
    CHECK (GB_any_aliased (A, C)) ;
    C->p = NULL ;
    C->p_shallow = false ;
    CHECK (!GB_any_aliased (A, C)) ;
    GB_FREE (&(C->i), C->i_size) ;
    C->i = A->i ;
    C->i_shallow = true ;
    CHECK (GB_any_aliased (A, C)) ;
    C->i = NULL ;
    C->i_shallow = false ;
    GrB_Matrix_free_(&A) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // GrB_apply with empty scalar
    //--------------------------------------------------------------------------

    OK (GrB_Scalar_new (&scalar, GrB_INT32)) ;
    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    OK (GrB_Matrix_new (&C, GrB_INT32, n, n)) ;
    expected = GrB_EMPTY_OBJECT ;
    ERR (GxB_Matrix_apply_BinaryOp2nd (C, NULL, NULL, GrB_PLUS_INT32, A,
        scalar, NULL)) ;
    OK (GrB_Matrix_error (&message, C)) ;
    printf ("expected error: %s\n", message) ;
    GrB_Matrix_free_(&A) ;
    GrB_Matrix_free_(&C) ;
    GrB_Scalar_free_(&scalar) ;

    //--------------------------------------------------------------------------
    // invalid descriptor
    //--------------------------------------------------------------------------

    int method ;
    OK (GrB_Descriptor_new (&desc)) ;
    OK (GxB_Descriptor_fprint (desc, "descriptor", GxB_COMPLETE, NULL)) ;

    OK (GxB_Desc_get (NULL, GxB_AxB_METHOD, &method)) ;
    CHECK (method == GxB_DEFAULT) ;
    method = -1 ;
    OK (GxB_Desc_get_INT32 (NULL, GxB_AxB_METHOD, &method)) ;
    CHECK (method == GxB_DEFAULT) ;
    OK (GxB_Desc_set (desc, GxB_AxB_METHOD, GxB_AxB_GUSTAVSON)) ;
    OK (GxB_Descriptor_fprint (desc, "descriptor", GxB_COMPLETE, NULL)) ;
    OK (GxB_Desc_get (desc, GxB_AxB_METHOD, &method)) ;
    CHECK (method == GxB_AxB_GUSTAVSON) ;

    OK (GxB_Desc_set_INT32 (desc, GxB_AxB_METHOD, GxB_AxB_SAXPY)) ;
    OK (GxB_Descriptor_fprint (desc, "descriptor", GxB_COMPLETE, NULL)) ;
    OK (GxB_Desc_get_INT32 (desc, GxB_AxB_METHOD, &method)) ;
    CHECK (method == GxB_AxB_SAXPY) ;

    desc->mask = GrB_REPLACE ;
    expected = GrB_INVALID_OBJECT ;
    ERR (GxB_Descriptor_fprint (desc, "invalid", GxB_COMPLETE, NULL)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // GrB_build an empty matrix
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    OK (GrB_Matrix_build_INT32 (A, I, I, I32, 0, GrB_PLUS_INT32)) ;
    OK (GxB_Matrix_fprint (A, "empty", GxB_COMPLETE, NULL)) ;
    CHECK (!GB_is_shallow (A)) ;
    GrB_Matrix_free_(&A) ;

    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    expected = GrB_DOMAIN_MISMATCH ;
    ERR (GrB_Matrix_build_INT32 (A, I, I, I32, 0, GxB_FIRSTI_INT32)) ;
    OK (GrB_Matrix_error (&message, A)) ;
    printf ("expected error: %s\n", message) ;
    GrB_Matrix_free_(&A) ;

    //--------------------------------------------------------------------------
    // reduce with positional op
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_INT32, n, n)) ;
    OK (GrB_Vector_new (&victor, GrB_INT32, n)) ;
    OK (GxB_Vector_Option_get_(victor, GxB_BITMAP_SWITCH, &bswitch)) ;
    printf ("vector bitmap switch: %g\n\n", bswitch) ;

    double b2 = 0 ;
    OK (GxB_Vector_Option_get_FP64 (victor, GxB_BITMAP_SWITCH, &b2)) ;
    CHECK (bswitch == b2) ;

    expected = GrB_NOT_IMPLEMENTED ;
    ERR (GrB_Matrix_reduce_BinaryOp (victor, NULL, NULL, GxB_FIRSTI_INT32,
        A, NULL)) ;
    OK (GrB_Vector_error (&message, victor)) ;
    printf ("error expected: %s\n", message) ;
    GrB_Matrix_free_(&A) ;
    GrB_Vector_free_(&victor) ;

    //--------------------------------------------------------------------------
    // GrB_init
    //--------------------------------------------------------------------------

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_init (GrB_BLOCKING)) ;

    //--------------------------------------------------------------------------
    // jumbled user-defined matrix
    //--------------------------------------------------------------------------

    wild ww, w2 ;
    n = 3 ;
    memset (ww.stuff, 13, 16 * sizeof (int)) ;
    OK (GrB_Type_new (&Wild, sizeof (wild))) ;
    OK (GrB_Matrix_new (&C, Wild, n, n)) ;
    OK (GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GrB_Matrix_assign_UDT (C, NULL, NULL, &ww, GrB_ALL, n, GrB_ALL, n,
        NULL)) ;
    OK (GxB_Matrix_fprint (C, "wild matrix", GxB_SHORT, NULL)) ;

    // jumble the matrix
    C->jumbled = true ;
    C->i [0] = 1 ;
    C->i [1] = 0 ;
    OK (GxB_Matrix_fprint (C, "wild matrix jumbled", GxB_SHORT, NULL)) ;

    // unjumble the matrix
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    OK (GxB_Matrix_fprint (C, "wild matrix unjumbled", GxB_SHORT, NULL)) ;

    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // malloc/realloc wrappers
    //--------------------------------------------------------------------------

    size_t nbytes ;
    bool ok = false ;
    int *p = GB_malloc_memory (4, sizeof (int), &nbytes) ;
    CHECK (p != NULL) ;
    p = GB_realloc_memory (1024*1024, sizeof (int), p, &nbytes, &ok) ;
    CHECK (p != NULL) ;
    CHECK (ok) ;
    p = GB_realloc_memory (4, GB_NMAX + 1, p, &nbytes, &ok) ;
    CHECK (!ok) ;
    GB_free_memory ((void **) &p, nbytes) ;

    //--------------------------------------------------------------------------
    // try to import a huge full matrix (this will fail):
    //--------------------------------------------------------------------------

    GrB_Matrix X = NULL ;
    info = GxB_Matrix_import_FullC (&X, GrB_FP32, GB_NMAX, GB_NMAX,
        NULL, UINT64_MAX, false, NULL) ;
    if (info != GrB_INVALID_VALUE || X != NULL) mexErrMsgTxt ("huge fail1") ;

    GrB_Index nhuge = (((GrB_Index) 2) << 50) ;
    info = GxB_Matrix_import_BitmapC (&X, GrB_FP32, nhuge, nhuge,
        NULL, NULL, 0, 0, false, 0, NULL) ;
    if (info != GrB_INVALID_VALUE || X != NULL) mexErrMsgTxt ("huge fail5") ;

    // try to convert a huge sparse matrix to bitmap (this will fail too):
    info = GrB_Matrix_new (&X, GrB_FP32, nhuge, nhuge) ;
    if (info != GrB_SUCCESS) mexErrMsgTxt ("huge fail2") ;
    info = GxB_Matrix_Option_set_(X, GxB_SPARSITY_CONTROL, GxB_BITMAP) ;
    if (info != GrB_OUT_OF_MEMORY) mexErrMsgTxt ("huge fail3") ;
    GrB_Matrix_free (&X) ;

    //--------------------------------------------------------------------------
    // hypermatrix prune
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, GB_NMAX, GB_NMAX)) ;
    OK (GrB_Matrix_setElement_FP32 (C, (double) 3, 0, 0)) ;
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    OK (GxB_Matrix_fprint (C, "huge matrix", GxB_SHORT, NULL)) ;
    C->nvec_nonempty = -1 ;
    OK (GB_hypermatrix_prune (C, NULL)) ;
    CHECK (C->nvec_nonempty == 1) ;
    GrB_Matrix_free (&C) ;

    //--------------------------------------------------------------------------
    // vector option set/get
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&victor, GrB_FP32, 10)) ;
    OK (GxB_Vector_Option_set (victor, GxB_BITMAP_SWITCH, (double) 4.5)) ;
    double bitmap_switch = 8 ;
    OK (GxB_Vector_Option_get (victor, GxB_BITMAP_SWITCH, &bitmap_switch)) ;
    CHECK (bitmap_switch == 4.5) ;

    OK (GxB_Vector_Option_set_FP64 (victor, GxB_BITMAP_SWITCH, (double) 5.25)) ;
    bitmap_switch = 8 ;
    OK (GxB_Vector_Option_get_FP64 (victor, GxB_BITMAP_SWITCH, &bitmap_switch)) ;
    CHECK (bitmap_switch == 5.25) ;

    GrB_Vector_free (&victor) ;

    //--------------------------------------------------------------------------
    // split/concat for user-defined types
    //--------------------------------------------------------------------------

    printf ("\n ======================== split/concat tests: ") ;
    OK (GxB_Global_Option_set_(GxB_BURBLE, true)) ;
    int sparsity [4] ;
    sparsity [0] = GxB_HYPERSPARSE ;
    sparsity [1] = GxB_SPARSE ;
    sparsity [2] = GxB_BITMAP ;
    sparsity [3] = GxB_FULL ;
    GrB_Matrix Tiles [4] ;
    memset (Tiles, 0, 4 * sizeof (GrB_Matrix)) ;
    n = 20 ;
    GrB_Index Tile_nrows [2] = { 5, 15 } ;
    GrB_Index Tile_ncols [2] = { 12, 8 } ;

    for (int k = 0 ; k <= 3 ; k++)
    {
        for (int k2 = 0 ; k2 <= 1 ; k2++)
        {
            OK (GrB_Matrix_new (&C, Wild, n, n)) ;
            OK (GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, sparsity [k])) ;
            if (k2 == 0)
            {
                // C(:,:) = ww
                OK (GrB_Matrix_assign_UDT (C, NULL, NULL, &ww, GrB_ALL,
                    n, GrB_ALL, n, NULL)) ;
            }
            else
            {
                // C = diagonal matrix
                for (int64_t kk = 0 ; kk < 20 ; kk++)
                {
                    OK (GrB_Matrix_setElement_UDT (C, &ww, kk, kk)) ;
                }
            }
            // split C into 4 matrices
//          printf ("Split C:\n") ;
//          GxB_print (C, 3) ;
            OK (GxB_Matrix_split (Tiles, 2, 2, Tile_nrows, Tile_ncols, C,
                NULL)) ;
            GxB_print (Tiles [0], 3) ;
            GxB_print (Tiles [1], 3) ;
            GxB_print (Tiles [2], 3) ;
            GxB_print (Tiles [3], 3) ;
            // concatenate the 4 matrices back in X
            OK (GrB_Matrix_new (&X, Wild, n, n)) ;
            OK (GxB_Matrix_Option_set (X, GxB_SPARSITY_CONTROL, sparsity [k])) ;
            OK (GxB_Matrix_concat (X, Tiles, 2, 2, NULL)) ;
//          printf ("Got X:\n") ;
//          GxB_print (X, 3) ;
            // ensure C and X are the same (just use a brute force method)
            for (int64_t i = 0 ; i < n ; i++)
            {
                for (int64_t j = 0 ; j < n ; j++)
                {
                    wild wc, wx ;
                    int infoc = GrB_Matrix_extractElement_UDT (&wc, C, i, j) ;
                    int infox = GrB_Matrix_extractElement_UDT (&wx, X, i, j) ;
                    CHECK (infoc == GrB_SUCCESS || infoc == GrB_NO_VALUE) ;
                    CHECK (infoc == infox) ;
//                  printf ("entry (%ld,%ld):\n", i, j) ;
                    if (infoc == GrB_SUCCESS)
                    {
                        for (int kk = 0 ; kk < 16 ; kk++)
                        {
//                          printf ("    %d: %ld %ld\n", kk,
//                          wc.stuff [kk], wx.stuff [kk]) ;
                            CHECK (wc.stuff [kk] == wx.stuff [kk]) ;
                        }
                    }
                }
            }
            GrB_Matrix_free (&X) ;
            GrB_Matrix_free (&C) ;

            expected = GrB_DOMAIN_MISMATCH ;
            OK (GrB_Matrix_new (&X, GrB_FP32, n, n)) ;
            ERR (GxB_Matrix_concat (X, Tiles, 2, 2, NULL)) ;
            OK (GrB_Matrix_error (&err, X)) ;
            printf ("expected error: %s\n", err) ;
            GrB_Matrix_free (&X) ;

            expected = GrB_DIMENSION_MISMATCH ;
            OK (GrB_Matrix_new (&X, Wild, 100, 100)) ;
            ERR (GxB_Matrix_concat (X, Tiles, 2, 2, NULL)) ;
            OK (GrB_Matrix_error (&err, X)) ;
            printf ("expected error: %s\n", err) ;
            GrB_Matrix_free (&X) ;

            OK (GrB_Matrix_new (&X, Wild, n, n)) ;
            GrB_Matrix_free (&(Tiles [3])) ;
            OK (GrB_Matrix_new (&(Tiles [3]), Wild, 15, 100)) ;
            ERR (GxB_Matrix_concat (X, Tiles, 2, 2, NULL)) ;
            // GxB_print (X, 3) ;
            OK (GrB_Matrix_error (&err, X)) ;
            printf ("expected error: %s\n", err) ;
            GrB_Matrix_free (&(Tiles [3])) ;

            OK (GrB_Matrix_new (&(Tiles [3]), Wild, 100, 8)) ;
            ERR (GxB_Matrix_concat (X, Tiles, 2, 2, NULL)) ;
            OK (GrB_Matrix_error (&err, X)) ;
            printf ("expected error: %s\n", err) ;

            for (int kk = 0 ; kk < 4 ; kk++)
            {
                GrB_Matrix_free (&(Tiles [kk])) ;
            }

            expected = GrB_NULL_POINTER ;
            ERR (GxB_Matrix_concat (X, Tiles, 2, 2, NULL)) ;
            GrB_Matrix_free (&X) ;
        }
    }
    OK (GxB_Global_Option_set_(GxB_BURBLE, false)) ;

    //--------------------------------------------------------------------------
    // split/concat error handling
    //--------------------------------------------------------------------------

    expected = GrB_INVALID_VALUE ;
    OK (GrB_Matrix_new (&C, GrB_FP32, n, n)) ;
    ERR (GxB_Matrix_split (Tiles, 0, 0, Tile_nrows, Tile_ncols, C, NULL)) ;
    ERR (GxB_Matrix_concat (C, Tiles, 0, 0, NULL)) ;
    GrB_Matrix_free (&C) ;

    expected = GrB_DIMENSION_MISMATCH ;
    OK (GrB_Matrix_new (&C, GrB_FP32, n, n)) ;
    Tile_nrows [0] = -1 ;
    ERR (GxB_Matrix_split (Tiles, 2, 2, Tile_nrows, Tile_ncols, C, NULL)) ;
    Tile_nrows [0] = 1 ;
    ERR (GxB_Matrix_split (Tiles, 2, 2, Tile_nrows, Tile_ncols, C, NULL)) ;
    Tile_nrows [0] = 5 ;
    Tile_ncols [0] = -1 ;
    ERR (GxB_Matrix_split (Tiles, 2, 2, Tile_nrows, Tile_ncols, C, NULL)) ;
    Tile_ncols [0] = 1 ;
    ERR (GxB_Matrix_split (Tiles, 2, 2, Tile_nrows, Tile_ncols, C, NULL)) ;
    GrB_Matrix_free (&C) ;

    //--------------------------------------------------------------------------
    // C<C,struct> = scalar
    //--------------------------------------------------------------------------

    printf ("\n\ntesting C<C,struct> = scalar for user-defined type:\n") ;
    OK (GxB_Global_Option_set (GxB_BURBLE, true)) ;
    OK (GrB_Matrix_new (&C, Wild, n, n)) ;
    memset (ww.stuff,  0, 16 * sizeof (int)) ;
    memset (w2.stuff,  1, 16 * sizeof (int)) ;
    for (int64_t kk = 0 ; kk < 16 ; kk++)
    {
        w2.stuff [kk] = kk ;
    }
    for (int64_t kk = 0 ; kk < 20 ; kk++)
    {
        OK (GrB_Matrix_setElement_UDT (C, &ww, kk, kk)) ;
    }
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    info = GrB_Matrix_assign_UDT (C, C, NULL, &w2, GrB_ALL, 20, GrB_ALL, 20,
        GrB_DESC_S) ;
    wild w3 ;

    for (int64_t kk = 0 ; kk < 20 ; kk++)
    {
        memset (w3.stuff,  9, 16 * sizeof (int)) ;
        info = (GrB_Matrix_extractElement_UDT (&w3, C, kk, kk)) ;
        CHECK (info == GrB_SUCCESS) ;
        for (int64_t t = 0 ; t < 16 ; t++)
        {
            CHECK (w3.stuff [t] == t) ;
        }
    }
    GrB_Matrix_free (&C) ;
    OK (GxB_Global_Option_set (GxB_BURBLE, false)) ;

    //--------------------------------------------------------------------------
    // GxB_Matrix_diag and GxB_Vector_diag error handling
    //--------------------------------------------------------------------------

    expected = GrB_DIMENSION_MISMATCH ;

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 20)) ;
    OK (GrB_Vector_new (&victor, GrB_FP32, 10)) ;
    ERR (GxB_Matrix_diag (C, victor, 0, NULL)) ;
    OK (GrB_Matrix_error (&err, C)) ;
    printf ("expected error: %s\n", err) ;
    GrB_Matrix_free (&C) ;

    OK (GrB_Matrix_new (&C, GrB_FP32, 5, 5)) ;
    ERR (GxB_Matrix_diag (C, victor, 0, NULL)) ;
    OK (GrB_Matrix_error (&err, C)) ;
    printf ("expected error: %s\n", err) ;
    ERR (GxB_Vector_diag (victor, C, 0, NULL)) ;
    OK (GrB_Vector_error (&err, victor)) ;
    printf ("expected error: %s\n", err) ;
    GrB_Matrix_free (&C) ;

    expected = GrB_DOMAIN_MISMATCH ;

    OK (GrB_Matrix_new (&C, Wild, 10, 10)) ;
    ERR (GxB_Matrix_diag (C, victor, 0, NULL)) ;
    OK (GrB_Matrix_error (&err, C)) ;
    printf ("expected error: %s\n", err) ;
    ERR (GxB_Vector_diag (victor, C, 0, NULL)) ;
    OK (GrB_Vector_error (&err, victor)) ;
    printf ("expected error: %s\n", err) ;
    GrB_Matrix_free (&C) ;
    GrB_Vector_free (&victor) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GrB_Type_free_(&Wild) ;
    GB_mx_put_global (true) ;   
    fclose (f) ;
    printf ("\nAll errors printed above were expected.\n") ;
    printf ("GB_mex_test2: all tests passed\n\n") ;
}

