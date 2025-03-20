//------------------------------------------------------------------------------
// GB_unload_into_container: unload a GrB_Matrix into a Container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method takes O(1) time and performs no mallocs at all, unless A has
// pending work that must be finished.  It typically will perform no frees,
// unless A has an error string in A->logger, or if the Container has prior
// content (which is freed).

#include "GB_container.h"
#define GB_FREE_ALL ;

GrB_Info GB_unload_into_container   // GrB_Matrix -> GxB_Container
(
    GrB_Matrix A,               // matrix to unload into the Container
    GxB_Container Container,    // Container to hold the contents of A
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A to unload into Container", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (Container->Y, "Container->Y before unload", GB0) ;
    GB_CHECK_CONTAINER (Container) ;

    //--------------------------------------------------------------------------
    // finish any pending work, but permit A to still be jumbled
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;

    //--------------------------------------------------------------------------
    // unload the matrix into the container
    //--------------------------------------------------------------------------

    int64_t nvals = GB_nnz (A) ;
    int64_t nx = GB_nnz_held (A) ;
    bool iso = A->iso ;
    bool is_csc = A->is_csc ;

    int64_t nvec_nonempty = GB_nvec_nonempty_get (A) ;

    Container->nrows = (is_csc) ? A->vlen : A->vdim ;
    Container->ncols = (is_csc) ? A->vdim : A->vlen ;
    Container->nrows_nonempty = (is_csc) ? -1 : nvec_nonempty ;
    Container->ncols_nonempty = (is_csc) ? nvec_nonempty : -1 ;
    Container->nvals = nvals ;
    Container->format = GB_sparsity (A) ;
    Container->orientation = (is_csc) ? GrB_COLMAJOR : GrB_ROWMAJOR ;
    Container->iso = iso ;
    Container->jumbled = A->jumbled ;

    switch (Container->format)
    {

        case GxB_HYPERSPARSE : 

            // unload A->Y into the Container unless it is entirely shallow
            GB_Matrix_free (&(Container->Y)) ;
            if (!A->Y_shallow)
            { 
                // A->Y may still have shallow components, which is OK
                Container->Y = A->Y ;
                A->Y = NULL ;
            }
            // unload A->p, A->h, and A->i into the Container
            GB_vector_load (Container->p, &(A->p),
                A->p_is_32 ? GrB_UINT32 : GrB_UINT64,
                A->nvec+1, A->p_size, A->p_shallow) ;
            GB_vector_load (Container->h, &(A->h),
                A->j_is_32 ? GrB_INT32 : GrB_INT64,
                A->nvec, A->h_size, A->h_shallow) ;
            GB_vector_load (Container->i, &(A->i),
                A->i_is_32 ? GrB_INT32 : GrB_INT64,
                nvals, A->i_size, A->i_shallow) ;
            break ;

        case GxB_SPARSE : 

            // unload A->p and A->i into the Container
            GB_vector_load (Container->p, &(A->p),
                A->p_is_32 ? GrB_UINT32 : GrB_UINT64,
                A->plen+1, A->p_size, A->p_shallow) ;
            GB_vector_load (Container->i, &(A->i),
                A->i_is_32 ? GrB_INT32 : GrB_INT64,
                nvals, A->i_size, A->i_shallow) ;
            break ;

        case GxB_BITMAP : 

            // unload A->b into the Container
            GB_vector_load (Container->b, (void **) &(A->b), GrB_INT8,
                nx, A->b_size, A->b_shallow) ;
            break ;

        case GxB_FULL : 
        default :;
            break ;
    }

    // unload A->x into the Container
    GB_vector_load (Container->x, &(A->x), A->type, iso ? 1 : nx,
        A->x_size, A->x_shallow) ;

    //--------------------------------------------------------------------------
    // change A to a dense 0-by-0 matrix with no content
    //--------------------------------------------------------------------------

    // A->user_name, A->type, and all controls are preserved.  Everything else
    // is revised.

    GB_phybix_free (A) ;
    A->plen = -1 ;
    A->vlen = 0 ;
    A->vdim = 0 ;
//  A->nvec_nonempty = 0 ;
    GB_nvec_nonempty_set (A, 0) ;
    A->p_is_32 = false ;
    A->j_is_32 = false ;
    A->i_is_32 = false ;
    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A unloaded into Container", GB0) ;
    ASSERT_VECTOR_OK (Container->p, "Container->p after unload", GB0) ;
    ASSERT_VECTOR_OK (Container->h, "Container->h after unload", GB0) ;
    ASSERT_VECTOR_OK (Container->b, "Container->b after unload", GB0) ;
    ASSERT_VECTOR_OK (Container->i, "Container->i after unload", GB0) ;
    ASSERT_VECTOR_OK (Container->x, "Container->x after unload", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (Container->Y, "Container->Y after unload", GB0) ;
    return (GrB_SUCCESS) ;
}

