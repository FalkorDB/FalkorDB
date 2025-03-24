//------------------------------------------------------------------------------
// GB_mex_bsort: sort IJK using a struct qsort
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[I,J,K] = GB_mex_bsort (I,J,K)"

typedef struct
{
    uint32_t i ;
    uint32_t j ;
    uint32_t k ;
}
GB_bsort_32_32_32_t ;

typedef struct
{
    uint64_t i ;
    uint64_t j ;
    uint64_t k ;
}
GB_bsort_64_64_64_t ;

#undef  GB_lt_1
#define GB_lt_1(A, a, B, b) (A [a].k < B [b].k)

#undef  GB_lt_2
#define GB_lt_2(A, a, B, b)         \
(                                   \
    (A [a].j < B [b].j) ?           \
    (                               \
        true                        \
    )                               \
    :                               \
    (                               \
        (A [a].j == B [b].j) ?      \
        (                           \
            GB_lt_1 (A, a, B, b)    \
        )                           \
        :                           \
        (                           \
            false                   \
        )                           \
    )                               \
)

#undef  GB_lt_3
#define GB_lt_3(A, a, B, b)         \
(                                   \
    (A [a].i < B [b].i) ?           \
    (                               \
        true                        \
    )                               \
    :                               \
    (                               \
        (A [a].i == B [b].i) ?      \
        (                           \
            GB_lt_2 (A, a, B, b)    \
        )                           \
        :                           \
        (                           \
            false                   \
        )                           \
    )                               \
)

#undef  GB_lt
#define GB_lt(A,a,B,b) GB_lt_3(A,a,B,b)

// swap A [a] and A [b]
#undef  GB_swap
#define GB_swap(A,a,b)              \
{                                   \
    GB_BSORT_T t = A [a] ;          \
    A [a] = A [b] ;                 \
    A [b] = t ;                     \
}

#define GB_BSORT_T GB_bsort_32_32_32_t
#define GB_partition GB_partition_32_32_32
#define GB_quicksort GB_quicksort_32_32_32
#include "factory/GB_bsort_template.c"

#undef  GB_BSORT_T
#undef  GB_partition
#undef  GB_quicksort

#define GB_BSORT_T GB_bsort_64_64_64_t
#define GB_partition GB_partition_64_64_64
#define GB_quicksort GB_quicksort_64_64_64
#include "factory/GB_bsort_template.c"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    // check inputs
    if (nargin != 3 || nargout != 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    bool I_is_32 ;
    if (mxIsClass (pargin [0], "uint32"))
    { 
        I_is_32 = true ;
    }
    else if (mxIsClass (pargin [0], "uint64"))
    {
        I_is_32 = false ;
    }
    else
    {
        mexErrMsgTxt ("I must be a uint32 or uint64 array") ;
    }

    bool J_is_32 ;
    if (mxIsClass (pargin [1], "uint32"))
    { 
        J_is_32 = true ;
    }
    else if (mxIsClass (pargin [1], "uint64"))
    {
        J_is_32 = false ;
    }
    else
    {
        mexErrMsgTxt ("I must be a uint32 or uint64 array") ;
    }

    bool K_is_32 ;
    if (mxIsClass (pargin [2], "uint32"))
    { 
        K_is_32 = true ;
    }
    else if (mxIsClass (pargin [2], "uint64"))
    {
        K_is_32 = false ;
    }
    else
    {
        mexErrMsgTxt ("I must be a uint32 or uint64 array") ;
    }

    void *I = mxGetData (pargin [0]) ;
    int64_t n = (uint64_t) mxGetNumberOfElements (pargin [0]) ;
    uint32_t *I32 = (I_is_32) ? I : NULL ;
    uint64_t *I64 = (I_is_32) ? NULL : I ;

    void *J = mxGetData (pargin [1]) ;
    uint32_t *J32 = (J_is_32) ? J : NULL ;
    uint64_t *J64 = (J_is_32) ? NULL : J ;
    if (n != (uint64_t) mxGetNumberOfElements (pargin [1])) 
    {
        mexErrMsgTxt ("I and J must be the same length") ;
    }

    void *K = mxGetData (pargin [2]) ;
    uint32_t *K32 = (K_is_32) ? K : NULL ;
    uint64_t *K64 = (K_is_32) ? NULL : K ;
    if (n != (uint64_t) mxGetNumberOfElements (pargin [2])) 
    {
        mexErrMsgTxt ("I and K must be the same length") ;
    }

    uint64_t seed = n ;

    pargout [0] = GB_mx_create_full (n, 1, I_is_32 ? GrB_UINT32 : GrB_UINT64) ;
    void *Io = mxGetData (pargout [0]) ;
    uint32_t *Io32 = (I_is_32) ? Io : NULL ;
    uint64_t *Io64 = (I_is_32) ? NULL : Io ;

    pargout [1] = GB_mx_create_full (n, 1, J_is_32 ? GrB_UINT32 : GrB_UINT64) ;
    void *Jo = mxGetData (pargout [1]) ;
    uint32_t *Jo32 = (J_is_32) ? Jo : NULL ;
    uint64_t *Jo64 = (J_is_32) ? NULL : Jo ;

    pargout [2] = GB_mx_create_full (n, 1, K_is_32 ? GrB_UINT32 : GrB_UINT64) ;
    void *Ko = mxGetData (pargout [2]) ;
    uint32_t *Ko32 = (K_is_32) ? Ko : NULL ;
    uint64_t *Ko64 = (K_is_32) ? NULL : Ko ;

    double t ;

    if (I_is_32 && J_is_32 && K_is_32)
    {
        GB_bsort_32_32_32_t *A = mxMalloc (n * sizeof (GB_bsort_32_32_32_t)) ;
        for (int64_t k = 0 ; k < n ; k++)
        {
            A [k].i = I32 [k] ;
            A [k].j = J32 [k] ;
            A [k].k = K32 [k] ;
        }
//      t = GB_omp_get_wtime ( ) ;
        GB_quicksort_32_32_32 (A, n, &seed) ;
//      t = GB_omp_get_wtime ( ) - t ;
        for (int64_t k = 0 ; k < n ; k++)
        {
            Io32 [k] = A [k].i ;
            Jo32 [k] = A [k].j ;
            Ko32 [k] = A [k].k ;
        }
    }
    else
    { 
        GB_bsort_64_64_64_t *A = mxMalloc (n * sizeof (GB_bsort_64_64_64_t)) ;
        for (int64_t k = 0 ; k < n ; k++)
        {
            A [k].i = I32 ? ((uint64_t) I32 [k]) : I64 [k] ;
            A [k].j = J32 ? ((uint64_t) J32 [k]) : J64 [k] ;
            A [k].k = K32 ? ((uint64_t) K32 [k]) : K64 [k] ;
        }
//      t = GB_omp_get_wtime ( ) ;
        GB_quicksort_64_64_64 (A, n, &seed) ;
//      t = GB_omp_get_wtime ( ) - t ;
        for (int64_t k = 0 ; k < n ; k++)
        {
            if (Io32) { Io32 [k] = A [k].i ; } else { Io64 [k] = A [k].i ; }
            if (Jo32) { Jo32 [k] = A [k].j ; } else { Jo64 [k] = A [k].j ; }
            if (Ko32) { Ko32 [k] = A [k].k ; } else { Ko64 [k] = A [k].k ; }
        }
    }
//  printf ("bsort time: %g\n:", t) ;
}


