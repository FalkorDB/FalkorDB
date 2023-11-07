//------------------------------------------------------------------------------
// GB_mx_isequal: check if two matrices are equal
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

bool GB_mx_isequal     // true if A and B are exactly the same
(
    GrB_Matrix A,
    GrB_Matrix B,
    double eps      // if A and B are both FP32 or FP64, and if eps > 0,
                    // then the values are considered equal if their relative
                    // difference is less than or equal to eps.
)
{

    if (A == B) return (true) ;
    if (A == NULL) return (false) ;
    if (B == NULL) return (false) ;

    int A_sparsity = GB_sparsity (A) ;
    if (A_sparsity != GB_sparsity (B))
    {
        return (false) ;
    }

    GB_Pending AP = A->Pending ;
    GB_Pending BP = B->Pending ;

    if (A->magic != B->magic) return (false) ;
    if (A->type  != B->type ) return (false) ;
    if (A->vlen  != B->vlen ) return (false) ;
    if (A->vdim  != B->vdim ) return (false) ;
    if (A->nvec  != B->nvec ) return (false) ;

    if (GB_nnz (A)  != GB_nnz (B) ) return (false) ;

    if ((A->h != NULL) != (B->h != NULL)) return (false) ;
    if (A->is_csc   != B->is_csc  ) return (false) ;

    if (A->nzombies         != B->nzombies         ) return (false) ;

    if ((AP != NULL) != (BP != NULL)) return (false) ;

    if (AP != NULL)
    {
        if (AP->n      != BP->n     ) return (false) ;
        if (AP->sorted != BP->sorted) return (false) ;
        if (AP->op     != BP->op    ) return (false) ;
        if (AP->type   != BP->type  ) return (false) ;
        if (AP->size   != BP->size  ) return (false) ;
    }

    int64_t n = A->nvec ;
    int64_t nnz = GB_nnz (A) ;
    size_t s = sizeof (int64_t) ;
    size_t asize = A->type->size ;

    ASSERT (n >= 0 && n <= A->vdim) ;

    bool A_is_dense = GB_is_dense (A) || GB_IS_FULL (A) ;
    bool B_is_dense = GB_is_dense (B) || GB_IS_FULL (B) ;

    if (A_is_dense != B_is_dense) return (false) ;

    if (!A_is_dense)
    {
        if (!GB_mx_same  ((char *) A->p, (char *) B->p, (n+1) * s))
        {
            return (false) ;
        }
        if (A->h != NULL)
        {
            if (!GB_mx_same ((char *) A->h, (char *) B->h, n * s))
                return (false) ;
        }
    }

    if (A_sparsity == GxB_BITMAP)
    {
        if (!GB_mx_same ((char *) A->b, (char *) B->b, nnz))
        {
            return (false) ;
        }
    }

    if (GB_nnz_max (A) > 0 && GB_nnz_max (B) > 0)
    {
        if (!A_is_dense)
        {
            if (!GB_mx_same  ((char *) A->i, (char *) B->i, nnz * s))
            {
                return (false) ;
            }
        }

        if (A->type == GrB_FP32 && eps > 0)
        {
            if (!GB_mx_xsame32 (
                A->x, A->iso,   // OK
                B->x, B->iso,   // OK
                A->b, nnz, A->i, eps))
            {
                return (false) ;
            }
        }
        else if (A->type == GrB_FP64 && eps > 0)
        {
            if (!GB_mx_xsame64 (
                A->x, A->iso,   // OK
                B->x, B->iso,   // OK
                A->b, nnz, A->i, eps))
            {
                return (false) ;
            }
        }
        else
        {
            if (!GB_mx_xsame (
                A->x, A->iso,   // OK
                B->x, B->iso,   // OK
                A->b, nnz, asize, A->i))
            {
                return (false) ;
            }
        }
    }

    if (AP != NULL)
    {
        size_t psize = AP->size ;
        int64_t np = AP->n ;
        if (!GB_mx_same ((char *) AP->i, (char *) BP->i, np*s)) return (false) ;
        if (!GB_mx_same ((char *) AP->j, (char *) BP->j, np*s)) return (false) ;
        if ((AP->x == NULL) != (BP->x == NULL)) // OK
        {
            return (false) ;
        }
        if (AP->x != NULL && BP->x != NULL)     // OK
        {
            if (!GB_mx_same ((char *) AP->x, (char *) BP->x, np*psize)) // OK
            {
                return (false) ;
            }
        }
    }

    return (true) ;
}

