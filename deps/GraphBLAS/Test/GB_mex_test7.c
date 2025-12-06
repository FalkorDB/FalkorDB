//------------------------------------------------------------------------------
// GB_mex_test7: still more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Test lots of random stuff.  The function otherwise serves no purpose.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test7"
#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix A = NULL, B = NULL, Y_mangled = NULL, Y = NULL, C1 = NULL,
        C2 = NULL ;
    GrB_Scalar s = NULL ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // matrix check
    //--------------------------------------------------------------------------

    OK (GrB_Scalar_new (&s, GrB_INT64)) ;
    OK (GrB_Scalar_setElement_INT64 (s, 0)) ;
    OK (GrB_Global_set_Scalar (GrB_GLOBAL, s, GxB_HYPER_HASH)) ;
    OK (GrB_Scalar_clear (s)) ;
    OK (GrB_Global_get_Scalar (GrB_GLOBAL, s, GxB_HYPER_HASH)) ;
    int64_t i = 1 ;
    OK (GrB_Scalar_extractElement_INT64 (&i, s)) ;
    CHECK (i == 0) ;

    OK (GrB_Matrix_new (&A, GrB_FP64, 100, 100)) ;
    OK (GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
    OK (GrB_Matrix_setElement_FP64 (A, (double) 1.2, 0, 0)) ;
    OK (GrB_Matrix_wait (A, 1)) ;
    OK (GxB_Matrix_fprint (A, "A valid", 3, NULL)) ;

    int will_wait = true ;
    OK (GrB_Matrix_get_INT32 (A, &will_wait, GxB_WILL_WAIT)) ;
    CHECK (!will_wait) ;

    printf ("\ninvalid A->p:\n") ;
    size_t save = A->p_size ;
    A->p_size = 3 ;
    expected = GrB_INVALID_OBJECT ;
    ERR (GxB_Matrix_fprint (A, "A with invalid A->p", 3, NULL)) ;
    A->p_size = save ;

    printf ("\ninvalid A->h:\n") ;
    save = A->h_size ;
    A->h_size = 3 ;
    ERR (GxB_Matrix_fprint (A, "A with invalid A->h", 3, NULL)) ;
    A->h_size = save ;

    printf ("\ninvalid A->Y:\n") ;
    CHECK (A->Y != NULL) ;
    A->Y->magic = GB_MAGIC2 ;
    ERR (GxB_Matrix_fprint (A, "A with invalid A->Y", 3, NULL)) ;
    A->Y->magic = GB_MAGIC ;

    OK (GrB_Matrix_new (&Y_mangled, GrB_FP64, 100, 100)) ;
    OK (GrB_Matrix_free (&(A->Y))) ;
    A->Y = Y_mangled ;
    ERR (GxB_Matrix_fprint (A, "A with invalid A->Y (wrong type)", 3, NULL)) ;
    A->Y = NULL ;

    OK (GrB_Matrix_free (&A)) ;

    OK (GrB_Matrix_new (&A, GrB_FP64, 100, 100)) ;
    OK (GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GrB_Matrix_setElement_FP64 (A, (double) 1.2, 0, 0)) ;
    OK (GrB_Matrix_wait (A, 1)) ;
    OK (GxB_Matrix_fprint (A, "A valid (sparse)", 3, NULL)) ;

    A->Y = Y_mangled ;
    ERR (GxB_Matrix_fprint (A, "A with invalid A->Y (not hyper)", 3, NULL)) ;
    A->Y = NULL ;

    OK (GrB_Matrix_free (&Y_mangled)) ;
    OK (GrB_Matrix_free (&A)) ;

    OK (GrB_Matrix_new (&A, GrB_FP64, 100, 100)) ;
    OK (GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
    OK (GrB_Matrix_setElement_FP64 (A, (double) 1.2, 0, 0)) ;
    OK (GrB_Matrix_wait (A, 1)) ;
    OK (GxB_Matrix_fprint (A, "A valid (hypersparse)", 3, NULL)) ;

    OK (GrB_Matrix_new (&B, GrB_FP64, 100, 100)) ;
    OK (GrB_Matrix_free (&(A->Y))) ;
    ERR (GxB_pack_HyperHash (A, &B, NULL)) ;
    OK (GrB_Matrix_free (&B)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;

    OK (GrB_Matrix_new (&B, GrB_FP64, 100, 100)) ;
    OK (GxB_Matrix_Option_set (B, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
    OK (GrB_Matrix_setElement_FP64 (B, (double) 1.2, 0, 0)) ;
    OK (GrB_Matrix_wait (B, 1)) ;
    OK (GrB_Matrix_free (&(B->Y))) ;
    B->Y = A->Y ;
    B->Y_shallow = true ;

    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true,
        GxB_INCLUDE_READONLY_STATISTICS)) ;
    OK (GxB_Matrix_fprint (B,
        "B valid (shallow hypersparse: stats_mem_shallow true)", 3, NULL)) ;

    OK (GrB_Global_set_INT32 (GrB_GLOBAL, false,
        GxB_INCLUDE_READONLY_STATISTICS)) ;
    OK (GxB_Matrix_fprint (B,
        "B valid (shallow hypersparse: stats_mem_shallow false)", 3, NULL)) ;

    expected = GrB_INVALID_OBJECT ;
//  B->jumbled = true ;
//  ERR (GxB_Matrix_fprint (B, "B invalid; jumbled and shallow", 3, NULL)) ;
    B->jumbled = false ;

    OK (GxB_print (A,3)) ;
    OK (GxB_print (B,3)) ;

    A->jumbled = true ;
    will_wait = false ;
    OK (GrB_Matrix_get_INT32 (A, &will_wait, GxB_WILL_WAIT)) ;
    CHECK (will_wait) ;

    A->jumbled = false ;
    will_wait = true ;
    OK (GrB_Matrix_get_INT32 (A, &will_wait, GxB_WILL_WAIT)) ;
    CHECK (!will_wait) ;

    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true,
        GxB_INCLUDE_READONLY_STATISTICS)) ;

    OK (GxB_print (B,3)) ;

    CHECK (GB_any_aliased (A, B)) ;
    OK (GrB_Matrix_free (&B)) ;

    OK (GxB_Matrix_fprint (A, "A still valid (hypersparse)", 3, NULL)) ;

    Y = NULL ;
    OK (GxB_pack_HyperHash (A, &Y, NULL)) ;
    OK (GxB_Matrix_fprint (A, "A hypersparse (pack did nothing)", 3, NULL)) ;

    GB_MDECL (A_Yi, , ) ;
    A_Yi = A->Y->i ;
    GB_IPTR (A_Yi, A->Y->i_is_32) ;
    GB_ISET (A_Yi, 0, 99) ; // A_Yi [0] = 99 ;
    ERR (GxB_Matrix_fprint (A, "A->Y invalid (not found) ", 3, NULL)) ;
    GB_ISET (A_Yi, 0,  0) ; // A_Yi [0] = 0 ;

    int64_t *Yx = A->Y->x ;
    Yx [0] = 99 ;
    ERR (GxB_Matrix_fprint (A, "A->Y invalid (wrong k) ", 3, NULL)) ;

    OK (GrB_Matrix_free (&A)) ;

    //--------------------------------------------------------------------------
    // hyper_hash with many collisions
    //--------------------------------------------------------------------------

    // The hyper_hash test assumes the following hash function and # of
    // buckets:

    // int64_t yvdim = ((uint64_t) 1) << (GB_FLOOR_LOG2 (anvec) + 1) ;
    // // divide by 4 to get a load factor of 2 to 4:
    // yvdim = yvdim / 4 ;
    // yvdim = GB_IMAX (yvdim, 4) ;
    // #define GB_HASHF2(i,b) ((((i) >> 2) + 17L*((i) >> 8)) & (b))

    // The hash bucket zero will have 257 entries:  change this if yvdim in
    // GB_hyper_hash_build.c, or GB_HASHF2 in GB_hash.h, change.

    int64_t n = 1024*1024 ;
    OK (GrB_Matrix_new (&A, GrB_FP64, n, n)) ;

    OK (GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
    OK (GxB_Matrix_Option_set (A, GxB_FORMAT, GxB_BY_ROW)) ;

    int64_t List [257] = {
        0, 1, 2, 3,
        1640, 1641, 1642, 1643,
        3280, 3281, 3282, 3283,
        6492, 6493, 6494, 6495,
        8132, 8133, 8134, 8135,
        11344, 11345, 11346, 11347,
        12984, 12985, 12986, 12987,
        16196, 16197, 16198, 16199,
        17836, 17837, 17838, 17839,
        21048, 21049, 21050, 21051,
        22688, 22689, 22690, 22691,
        25900, 25901, 25902, 25903,
        27540, 27541, 27542, 27543,
        29180, 29181, 29182, 29183,
        30752, 30753, 30754, 30755,
        32392, 32393, 32394, 32395,
        34032, 34033, 34034, 34035,
        35604, 35605, 35606, 35607,
        37244, 37245, 37246, 37247,
        38884, 38885, 38886, 38887,
        40456, 40457, 40458, 40459,
        42096, 42097, 42098, 42099,
        43736, 43737, 43738, 43739,
        46948, 46949, 46950, 46951,
        48588, 48589, 48590, 48591,
        51800, 51801, 51802, 51803,
        53440, 53441, 53442, 53443,
        56652, 56653, 56654, 56655,
        58292, 58293, 58294, 58295,
        61504, 61505, 61506, 61507,
        63144, 63145, 63146, 63147,
        66356, 66357, 66358, 66359,
        67996, 67997, 67998, 67999,
        71208, 71209, 71210, 71211,
        72848, 72849, 72850, 72851,
        74488, 74489, 74490, 74491,
        76060, 76061, 76062, 76063,
        77700, 77701, 77702, 77703,
        79340, 79341, 79342, 79343,
        80912, 80913, 80914, 80915,
        82552, 82553, 82554, 82555,
        84192, 84193, 84194, 84195,
        85764, 85765, 85766, 85767,
        87404, 87405, 87406, 87407,
        89044, 89045, 89046, 89047,
        92256, 92257, 92258, 92259,
        93896, 93897, 93898, 93899,
        97108, 97109, 97110, 97111,
        98748, 98749, 98750, 98751,
        101960, 101961, 101962, 101963,
        103600, 103601, 103602, 103603,
        106812, 106813, 106814, 106815,
        108452, 108453, 108454, 108455,
        111664, 111665, 111666, 111667,
        113304, 113305, 113306, 113307,
        116516, 116517, 116518, 116519,
        118156, 118157, 118158, 118159,
        119796, 119797, 119798, 119799,
        121368, 121369, 121370, 121371,
        123008, 123009, 123010, 123011,
        124648, 124649, 124650, 124651,
        126220, 126221, 126222, 126223,
        127860, 127861, 127862, 127863,
        129500, 129501, 129502, 129503,
        131072 } ;

    for (int64_t k = 0 ; k < 257 ; k++)
    {
        int64_t i = List [k] ;
        OK (GrB_Matrix_setElement (A, 2, i, i)) ;
    }

    for (int64_t k = 257 ; k < 1024 ; k++)
    {
        int64_t i = k + 200000 ;
        OK (GrB_Matrix_setElement (A, 4, i, i)) ;
    }

    OK (GrB_Matrix_wait (A, 1)) ;
    OK (GxB_Matrix_fprint (A, "A->Y with many collisions", 2, NULL)) ;

    CHECK (A->Y != NULL) ;
    GB_MDECL (A_Yp, , ) ;
    A_Yp = A->Y->p ;
    GB_IPTR (A_Yp, A->Y->p_is_32) ;
    CHECK (GB_IGET (A_Yp, 1) == 257) ;

    for (int shallow = 0 ; shallow <= 1 ; shallow++)
    {
        for (int j_is_32 = 0 ; j_is_32 <= 1 ; j_is_32++)
        {
            OK (GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL,
                GxB_HYPERSPARSE)) ;
            GrB_Matrix Y = NULL ;
            if (shallow)
            {
                A->Y_shallow = true ;
                Y = A->Y ;
            }
            OK (GB_convert_int (A, false, j_is_32, false, true)) ;
            if (shallow)
            {
                CHECK (A->Y == NULL) ;
                CHECK (A->Y_shallow == false) ;
                OK (GrB_Matrix_free (&Y)) ;
            }
            OK (GxB_Matrix_fprint (A, "hyper A for hyper_hash test", 2, NULL)) ;
            OK (GrB_Matrix_new (&C1, GrB_FP64, n, n)) ;
            OK (GrB_Matrix_new (&C2, GrB_FP64, n, n)) ;
            OK (GxB_Matrix_Option_set (C1, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
            OK (GxB_Matrix_Option_set (C2, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
            OK (GrB_mxm (C1, A, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, A,
                GrB_DESC_T0)) ;
            OK (GxB_Matrix_fprint (C1, "C1<A>=A'*A", 2, NULL)) ;
            OK (GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
            OK (GxB_Matrix_fprint (A, "sparse A for hyper_hash test", 2,
                NULL)) ;
            OK (GrB_mxm (C2, A, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, A,
                GrB_DESC_T0)) ;
            OK (GxB_Matrix_fprint (C2, "C2<A>=A'*A", 2, NULL)) ;
            CHECK (GB_mx_isequal (C1, C2, 0)) ;
            OK (GrB_Matrix_free (&C1)) ;
            OK (GrB_Matrix_free (&C2)) ;
        }
    }
    OK (GrB_Matrix_free (&A)) ;

    //--------------------------------------------------------------------------
    // axv2 and avx512f
    //--------------------------------------------------------------------------

    bool have_avx2 = GB_Global_cpu_features_avx2 ( ) ;
    bool have_avx512f = GB_Global_cpu_features_avx512f ( ) ;
    printf ("\navx2: %d avx512f: %d\n", have_avx2, have_avx512f) ;

    //--------------------------------------------------------------------------
    // compiler
    //--------------------------------------------------------------------------

    char *compiler ;
    int compiler_version [3] ;
    OK (GxB_Global_Option_get (GxB_COMPILER_NAME, &compiler)) ;
    OK (GxB_Global_Option_get (GxB_COMPILER_VERSION, compiler_version)) ;
    printf ("GraphBLAS compiled with:\n[%s] [v%d.%d.%d]\n", compiler,
        compiler_version [0], compiler_version [1], compiler_version [2]) ;

    //--------------------------------------------------------------------------
    // iterator errors
    //--------------------------------------------------------------------------

    GxB_Iterator iterator = NULL ;
    METHOD (GxB_Iterator_new (&iterator)) ;
    GxB_Iterator_free (&iterator) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    OK (GrB_Scalar_setElement_INT64 (s, 1024)) ;
    OK (GrB_Global_set_Scalar (GrB_GLOBAL, s, GxB_HYPER_HASH)) ;
    OK (GrB_Scalar_clear (s)) ;
    OK (GrB_Global_get_Scalar (GrB_GLOBAL, s, GxB_HYPER_HASH)) ;
    OK (GrB_Scalar_extractElement_INT64 (&i, s)) ;
    CHECK (i == 1024) ;
    OK (GrB_Scalar_free (&s)) ;
    GB_mx_put_global (true) ;   
    printf ("\nGB_mex_test7: all tests passed\n\n") ;
}

