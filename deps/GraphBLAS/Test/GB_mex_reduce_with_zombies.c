//------------------------------------------------------------------------------
// GB_mex_reduce_with_zombies: sum up the entries in a matrix with zombies
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "s = GB_mex_reduce_with_zombies (A)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GB_mx_put_global (true) ;           \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    uint64_t nvals = 0 ;

    // check inputs
    if (nargout > 1 || nargin > 1)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get a deep copy of the input matrix
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", true, true) ;
    OK (GxB_Matrix_fprint (A, "A without zombies", GxB_SILENT, NULL)) ;

    // ensure the matrix is sparse, by column
    OK (GrB_Matrix_set_INT32 (A, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (A, GrB_COLMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;

    // make every other entry a zombie
    uint64_t nzombies = 0 ;
    if (A->i_is_32)
    {
        uint32_t *Ai = A->i ;
        for (int64_t k = 0 ; k < nvals ; k += 2)
        {
            Ai [k] = ~(Ai [k]) ;
            nzombies++ ;
        }
    }
    else
    {
        uint64_t *Ai = A->i ;
        for (int64_t k = 0 ; k < nvals ; k += 2)
        {
            Ai [k] = ~(Ai [k]) ;
            nzombies++ ;
        }
    }
    A->nzombies = nzombies ;
    OK (GxB_Matrix_fprint (A, "A with zombies", GxB_SILENT, NULL)) ;

    // sum up the entries, excluding the zombies
    double result = 0 ;
    OK (GrB_Matrix_reduce_FP64 (&result, NULL, GrB_PLUS_MONOID_FP64, A, NULL)) ;

    // free workspace return the result
    pargout [0] = mxCreateDoubleScalar (result) ;
    FREE_ALL ;
}

