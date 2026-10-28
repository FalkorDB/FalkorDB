//------------------------------------------------------------------------------
// GB_no_malloc_free: stubs that do nothing when CUDA is not in use
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These malloc/free functions are used to reserve arenas 8 to 71 when CUDA is
// not in use.  These GB_rmm_malloc_# methods return NULL, and GB_rmm_free_# do
// nothing.

#include "GB.h"

#if !defined ( GRAPHBLAS_HAS_CUDA )

#define GB_RMM_MALLOC_FREE(id)                      \
    void *GB_rmm_malloc_ ## id (size_t size)        \
    {                                               \
        /* do not allocate anything */              \
        return (NULL) ;                             \
    }                                               \
    void GB_rmm_free_ ## id (void *p)               \
    {                                               \
        /* do nothing */ ;                          \
    }

GB_RMM_MALLOC_FREE (0) ;
GB_RMM_MALLOC_FREE (1) ;
GB_RMM_MALLOC_FREE (2) ;
GB_RMM_MALLOC_FREE (3) ;
GB_RMM_MALLOC_FREE (4) ;
GB_RMM_MALLOC_FREE (6) ;
GB_RMM_MALLOC_FREE (7) ;
GB_RMM_MALLOC_FREE (8) ;
GB_RMM_MALLOC_FREE (9) ;

GB_RMM_MALLOC_FREE (10) ;
GB_RMM_MALLOC_FREE (11) ;
GB_RMM_MALLOC_FREE (12) ;
GB_RMM_MALLOC_FREE (13) ;
GB_RMM_MALLOC_FREE (14) ;
GB_RMM_MALLOC_FREE (16) ;
GB_RMM_MALLOC_FREE (17) ;
GB_RMM_MALLOC_FREE (18) ;
GB_RMM_MALLOC_FREE (19) ;

GB_RMM_MALLOC_FREE (20) ;
GB_RMM_MALLOC_FREE (21) ;
GB_RMM_MALLOC_FREE (22) ;
GB_RMM_MALLOC_FREE (23) ;
GB_RMM_MALLOC_FREE (24) ;
GB_RMM_MALLOC_FREE (26) ;
GB_RMM_MALLOC_FREE (27) ;
GB_RMM_MALLOC_FREE (28) ;
GB_RMM_MALLOC_FREE (29) ;

GB_RMM_MALLOC_FREE (30) ;
GB_RMM_MALLOC_FREE (31) ;
GB_RMM_MALLOC_FREE (32) ;
GB_RMM_MALLOC_FREE (33) ;
GB_RMM_MALLOC_FREE (34) ;
GB_RMM_MALLOC_FREE (36) ;
GB_RMM_MALLOC_FREE (37) ;
GB_RMM_MALLOC_FREE (38) ;
GB_RMM_MALLOC_FREE (39) ;

GB_RMM_MALLOC_FREE (40) ;
GB_RMM_MALLOC_FREE (41) ;
GB_RMM_MALLOC_FREE (42) ;
GB_RMM_MALLOC_FREE (43) ;
GB_RMM_MALLOC_FREE (44) ;
GB_RMM_MALLOC_FREE (46) ;
GB_RMM_MALLOC_FREE (47) ;
GB_RMM_MALLOC_FREE (48) ;
GB_RMM_MALLOC_FREE (49) ;

GB_RMM_MALLOC_FREE (50) ;
GB_RMM_MALLOC_FREE (51) ;
GB_RMM_MALLOC_FREE (52) ;
GB_RMM_MALLOC_FREE (53) ;
GB_RMM_MALLOC_FREE (54) ;
GB_RMM_MALLOC_FREE (56) ;
GB_RMM_MALLOC_FREE (57) ;
GB_RMM_MALLOC_FREE (58) ;
GB_RMM_MALLOC_FREE (59) ;

GB_RMM_MALLOC_FREE (60) ;
GB_RMM_MALLOC_FREE (61) ;
GB_RMM_MALLOC_FREE (62) ;
GB_RMM_MALLOC_FREE (63) ;

#endif

