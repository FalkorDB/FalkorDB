//------------------------------------------------------------------------------
// GB_mex_test44: test gpu controls
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                    \
{                                   \
    GxB_Context_free (&Context) ;   \
    GrB_Scalar_free (&scalar) ;     \
    GB_Global_hack_set (6, 0) ;     \
    GB_Global_hack_set (7, 0) ;     \
}

//------------------------------------------------------------------------------
// GB_mex_test44 mexFunction
//------------------------------------------------------------------------------

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
    GxB_Context Context = NULL ;
    GrB_Scalar scalar = NULL ;

    //--------------------------------------------------------------------------
    // enable the gpu hack and pretend there are 4 GPUs
    //--------------------------------------------------------------------------

    int32_t ngpus_pretend = 4 ;
    GB_Global_hack_set (6, 1) ;
    GB_Global_hack_set (7, ngpus_pretend) ;

    //--------------------------------------------------------------------------
    // check the global gpu controls
    //--------------------------------------------------------------------------

    int32_t ngpus_max = -1 ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &ngpus_max, GxB_NGPUS_MAX)) ;
    CHECK (ngpus_max == ngpus_pretend) ;

    int32_t gpu_ids [3] = {3, 0, 2} ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, 3, GxB_NGPUS)) ;
    OK (GrB_Global_set_VOID  (GrB_GLOBAL, (void *) gpu_ids, GxB_GPU_IDS,
        3 * sizeof (int32_t))) ;

    int32_t ngpus_to_use = -1 ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &ngpus_to_use, GxB_NGPUS)) ;
    CHECK (ngpus_to_use == 3) ;

    int32_t gpu_ids2 [4] = {-1, -1, -1, -1} ;
    OK (GrB_Global_get_VOID (GrB_GLOBAL, (void *) gpu_ids2, GxB_GPU_IDS)) ;
    for (int k = 0 ; k < 3 ; k++)
    {
        CHECK (gpu_ids [k] == gpu_ids2 [k]) ;
    }
    CHECK (gpu_ids2 [3] == -1) ;

    size_t size = 0 ;
    OK (GrB_Global_get_SIZE (GrB_GLOBAL, &size, GxB_GPU_IDS)) ;
    CHECK (size == sizeof (int32_t) * GB_MAX_NGPUS) ;

    int expected = GrB_INVALID_VALUE ;
    ERR (GrB_Global_set_VOID  (GrB_GLOBAL, (void *) gpu_ids, GxB_GPU_IDS,
        2 * sizeof (int32_t))) ;

    OK (GrB_Global_set_VOID  (GrB_GLOBAL, (void *) gpu_ids2, GxB_GPU_IDS,
        3 * sizeof (int32_t))) ;

    gpu_ids2 [1] = 99 ;
    ERR (GrB_Global_set_VOID  (GrB_GLOBAL, (void *) gpu_ids2, GxB_GPU_IDS,
        3 * sizeof (int32_t))) ;

    //--------------------------------------------------------------------------
    // check the context gpu controls
    //--------------------------------------------------------------------------

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY GxB_Context_free (&Context) ;
    METHOD (GxB_Context_new (&Context)) ;

    ngpus_to_use = -1 ;
    OK (GxB_Context_get_INT (Context, &ngpus_to_use, GxB_NGPUS)) ;
    CHECK (ngpus_to_use == 3) ;

    int32_t gpu_ids3 [4] = {-1, -1, -1, -1} ;
    OK (GrB_Global_get_VOID (GrB_GLOBAL, (void *) gpu_ids3, GxB_GPU_IDS)) ;
    for (int k = 0 ; k < 3 ; k++)
    {
        CHECK (gpu_ids [k] == gpu_ids3 [k]) ;
    }
    CHECK (gpu_ids3 [3] == -1) ;

    ngpus_to_use = -1 ;
    OK (GrB_Scalar_new (&scalar, GrB_INT32)) ;
    OK (GxB_Context_get_Scalar (Context, scalar, GxB_NGPUS)) ;
    OK (GrB_Scalar_extractElement_INT32 (&ngpus_to_use, scalar)) ;
    CHECK (ngpus_to_use == 3) ;

    int32_t gpu_ids4 [2] = {1, 3} ;
    OK (GxB_Context_set_INT (Context, 2, GxB_NGPUS)) ;
    OK (GxB_Context_set_VOID  (Context, (void *) gpu_ids4, GxB_GPU_IDS,
        2 * sizeof (int32_t))) ;

    ngpus_to_use = -1 ;
    OK (GxB_Context_get_INT (Context, &ngpus_to_use, GxB_NGPUS)) ;
    CHECK (ngpus_to_use == 2) ;

    ngpus_to_use = GB_Context_gpu_ids (NULL) ;
    CHECK (ngpus_to_use == 3) ;

    OK (GxB_Context_engage (Context)) ;
    ngpus_to_use = GB_Context_gpu_ids (NULL) ;
    CHECK (ngpus_to_use == 2) ;

    OK (GxB_Context_fprint (Context, "Context", 5, NULL)) ;
    OK (GxB_Context_get_VOID (Context, (void *) gpu_ids3, GxB_GPU_IDS)) ;
    for (int k = 0 ; k < 2 ; k++)
    {
        CHECK (gpu_ids4 [k] == gpu_ids3 [k]) ;
    }

    ERR (GxB_Context_set_VOID  (Context, (void *) gpu_ids4, GxB_GPU_IDS,
        1 * sizeof (int32_t))) ;

    OK (GrB_Scalar_setElement_INT32 (scalar, 1)) ;
    OK (GxB_Context_set_Scalar (Context, scalar, GxB_NGPUS)) ;
    ngpus_to_use = -1 ;
    OK (GxB_Context_get_INT (Context, &ngpus_to_use, GxB_NGPUS)) ;
    CHECK (ngpus_to_use == 1) ;

    OK (GxB_Context_disengage (Context)) ;

    //--------------------------------------------------------------------------
    // internal methods
    //--------------------------------------------------------------------------

    ERR (GB_Context_gpu_ids_set (NULL, NULL, 42)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test44:  all tests passed\n\n") ;
}

