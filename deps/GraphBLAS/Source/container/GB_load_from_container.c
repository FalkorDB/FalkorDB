//------------------------------------------------------------------------------
// GB_load_from_container: load a GrB_Matrix from a Container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A->user_name and all controls are preserved.  Everything else in the matrix
// A is revised: the dimensions, type, content, 32/64 integer status, iso
// status, jumbled status, orientation (by row/col), etc.

#include "GB_container.h"
#define GB_FREE_ALL GB_phybix_free (A) ;

GrB_Info GB_load_from_container // GxB_Container -> GrB_Matrix
(
    GrB_Matrix A,               // matrix to load from the Container
    GxB_Container Container,    // Container with contents to load into A
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A to load from Container", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (Container->Y, "Container->Y before load", GB0) ;
    GB_CHECK_CONTAINER (Container) ;

    //--------------------------------------------------------------------------
    // free any prior content of A
    //--------------------------------------------------------------------------

    GB_phybix_free (A) ;

    //--------------------------------------------------------------------------
    // load the matrix from the container
    //--------------------------------------------------------------------------

    int format = Container->format ;
    uint64_t nvals = (format == GxB_FULL) ? 0 : (Container->nvals) ;
    uint64_t nrows = Container->nrows ;
    uint64_t ncols = Container->ncols ;

    A->nvals = nvals ;
    A->is_csc = (Container->orientation == GrB_COLMAJOR) ;
    A->vlen = (A->is_csc) ? nrows : ncols ;
    A->vdim = (A->is_csc) ? ncols : nrows ;

//  A->nvec_nonempty = (A->is_csc) ?
//      Container->ncols_nonempty : Container->nrows_nonempty ;
    GB_nvec_nonempty_set (A, (A->is_csc) ?
        Container->ncols_nonempty : Container->nrows_nonempty) ;

    A->iso = Container->iso ;
    A->jumbled = false ;
    uint64_t plen1 = 0, plen, Ab_len = 0, Ax_len = 0, Ai_len = 0 ;
    GrB_Type Ap_type = NULL, Ah_type = NULL, Ab_type = NULL, Ai_type = NULL ;
    uint64_t Ah_size = 0, Ap_size = 0, Ai_size = 0, Ab_size = 0, Ax_size = 0 ;
    uint64_t nrows_times_ncols = UINT64_MAX ;
    bool ok = GB_uint64_multiply (&nrows_times_ncols, nrows, ncols) ;
    bool jumbled = Container->jumbled ;

    // determine the A->j_is_32 condition when Container->h is empty
    bool h_empty = (Container->h == NULL ||
           (Container->h->vlen == 0 && Container->h->x == NULL)) ;
    if ((format == GxB_SPARSE) || (format == GxB_HYPERSPARSE && h_empty))
    { 
        bool Ap_is_32, Aj_is_32, Ai_is_32 ;
        GB_determine_pji_is_32 (&Ap_is_32, &Aj_is_32, &Ai_is_32,
            GxB_SPARSE, A->nvals, A->vlen, A->vdim, Werk) ;
        Ah_type = Aj_is_32 ? GrB_UINT32 : GrB_UINT64 ;
    }

    // clear the Container scalars
    Container->nrows = 0 ;
    Container->ncols = 0 ;
    Container->nrows_nonempty = -1 ;
    Container->ncols_nonempty = -1 ;
    Container->nvals = 0 ;
    Container->format = GxB_FULL ;
    Container->orientation = GrB_ROWMAJOR ;
    Container->iso = false ;
    Container->jumbled = false ;

    // Get or clear the phybix content: Ap, Ah, A->Y, A->b, A->i, and A->x,
    // depending on the format of the data held in the container.

    switch (format)
    {

        case GxB_HYPERSPARSE : 

            //------------------------------------------------------------------
            // hypersparse: load A->p, A->h, A->Y, and A->i from the container
            //------------------------------------------------------------------

            // load A->p
            GB_OK (GB_vector_unload (Container->p, &(A->p), &Ap_type,
                &plen1, &Ap_size, &(A->p_shallow), Werk)) ;
            A->p_size = (size_t) Ap_size ;

            // load or create A->h
            if (h_empty)
            { 
                // A is an empty hypersparse matrix but A->h must not be NULL;
                // allocate space for A->h of type Ah_type for a single entry
                // Ah_type is uint32 or uint64, so sizeof (uint64_t) is fine.
                plen = 0 ;
                size_t s = 0 ;
                A->h = GB_CALLOC_MEMORY (1, sizeof (uint64_t), &s) ;
                if (A->h == NULL)
                { 
                    GB_FREE_ALL ;
                    return (GrB_OUT_OF_MEMORY) ;
                }
                Ah_size = (uint64_t) s ;
                A->h_shallow = false ;
            }
            else
            { 
                GB_OK (GB_vector_unload (Container->h, &(A->h), &Ah_type,
                    &plen, &Ah_size, &(A->h_shallow), Werk)) ;
            }
            A->h_size = (size_t) Ah_size ;

            // load A->Y
            A->Y = Container->Y ;
            Container->Y = NULL ;

            // clear Container->b
            GB_vector_reset (Container->b) ;

            // load A->i
            GB_OK (GB_vector_unload (Container->i, &(A->i), &Ai_type,
                &Ai_len, &Ai_size, &(A->i_shallow), Werk)) ;
            A->i_size = (size_t) Ai_size ;

            // define plen, nvec, and jumbled
            A->plen = plen ;
            A->nvec = plen ;
            A->jumbled = jumbled ;

            // basic sanity checks
            if (plen1 != plen + 1 ||
                !(A->nvec >= 0 && A->nvec <= A->plen && A->plen <= A->vdim))
            { 
                GB_FREE_ALL ;
                return (GrB_INVALID_VALUE) ;
            }
            break ;

        case GxB_SPARSE : 

            //------------------------------------------------------------------
            // sparse: load A->p and A->i from the container
            //------------------------------------------------------------------

            // load A->p
            GB_OK (GB_vector_unload (Container->p, &(A->p), &Ap_type,
                &plen1, &Ap_size, &(A->p_shallow), Werk)) ;
            A->p_size = (size_t) Ap_size ;

            // clear Container->h, Y, and b
            GB_vector_reset (Container->h) ;
            GB_Matrix_free (&(Container->Y)) ;
            GB_vector_reset (Container->b) ;

            // load A->i
            GB_OK (GB_vector_unload (Container->i, &(A->i), &Ai_type,
                &Ai_len, &Ai_size, &(A->i_shallow), Werk)) ;
            A->i_size = (size_t) Ai_size ;

            // define plen, nvec, and jumbled
            A->plen = plen1 - 1 ;
            A->nvec = A->plen ;
            A->jumbled = jumbled ;

            // basic sanity checks
            if (!(A->nvec == A->plen && A->plen == A->vdim))
            { 
                GB_FREE_ALL ;
                return (GrB_INVALID_VALUE) ;
            }
            break ;

        case GxB_BITMAP : 

            //------------------------------------------------------------------
            // bitmap: load A->b from the container
            //------------------------------------------------------------------

            // clear Container->p, h, and Y
            GB_vector_reset (Container->p) ;
            GB_vector_reset (Container->h) ;
            GB_Matrix_free (&(Container->Y)) ;

            // load A->b
            GB_OK (GB_vector_unload (Container->b, (void **) &(A->b), &Ab_type,
                &Ab_len, &Ab_size, &(A->b_shallow), Werk)) ;
            A->b_size = (size_t) Ab_size ;

            // clear Container->i
            GB_vector_reset (Container->i) ;

            // define plen and nvec
            A->plen = -1 ;
            A->nvec = A->vdim ;

            // basic sanity checks
            if (Ab_type != GrB_INT8 || !ok || Ab_len < nrows_times_ncols)
            { 
                GB_FREE_ALL ;
                return (GrB_INVALID_VALUE) ;
            }
            break ;

        case GxB_FULL : 

            //------------------------------------------------------------------
            // full: clear phybi components
            //------------------------------------------------------------------

            GB_vector_reset (Container->p) ;
            GB_vector_reset (Container->h) ;
            GB_Matrix_free (&(Container->Y)) ;
            GB_vector_reset (Container->b) ;
            GB_vector_reset (Container->i) ;

            // define plen and nvec
            A->plen = -1 ;
            A->nvec = A->vdim ;
            break ;

        default :;
            break ;
    }

    // load A->x
    GB_OK (GB_vector_unload (Container->x, &(A->x), &(A->type),
        &Ax_len, &Ax_size, &(A->x_shallow), Werk)) ;
    A->x_size = (size_t) Ax_size ;

    // define the integer types
    A->p_is_32 = (Ap_type == GrB_UINT32) ;
    A->j_is_32 = (Ah_type == GrB_UINT32) ;
    A->i_is_32 = (Ai_type == GrB_UINT32) ;

    //--------------------------------------------------------------------------
    // more basic sanity checks
    //--------------------------------------------------------------------------

    // Loading a GrB_Matrix from a Container takes O(1) time, so there is not
    // enough time to test the entire content of the matrix to see if it's
    // valid.  The user application can do that with GxB_Matrix_fprint with a
    // print level of zero, after the matrix is loaded.

    // ensure Ax_len is the right size
    if (A->iso)
    { 
        // A->x must have size >= 1 for all iso matrices
        ok = (Ax_len >= 1) ;
    }
    else if (format == GxB_HYPERSPARSE || format == GxB_SPARSE)
    { 
        // A->x must have size >= A->nvals for non-iso sparse/hypersparse
        ok = (Ax_len >= A->nvals) ;
    }
    else
    { 
        // A->x must have size >= nrows*ncols for non-iso full/bitmap
        ok = ok && (Ax_len >= nrows_times_ncols) ;
    }

    // ensure Ai_len is the right size
    if (format == GxB_HYPERSPARSE || format == GxB_SPARSE)
    { 
        // A->i must have size >= A->nvals for sparse/hypersparse
        ok = ok && (Ai_len >= A->nvals) ;

        // A->p [A->plen] must match A->nvals
        GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
        ok = ok && (A->nvals == GB_IGET (Ap, A->plen)) ;
    }

    // if A->jumbled is true, ensure A has no readonly components
    if (A->jumbled)
    { 
        ok = ok && !GB_is_shallow (A) ;
    }

    if (!ok)
    { 
        GB_FREE_ALL ;
        return (GrB_INVALID_VALUE) ;
    }

    // the matrix has passed the basic checks
    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A loaded from Container", GB0) ;
    ASSERT_VECTOR_OK (Container->p, "Container->p after load", GB0) ;
    ASSERT_VECTOR_OK (Container->h, "Container->h after load", GB0) ;
    ASSERT_VECTOR_OK (Container->b, "Container->b after load", GB0) ;
    ASSERT_VECTOR_OK (Container->i, "Container->i after load", GB0) ;
    ASSERT_VECTOR_OK (Container->x, "Container->x after load", GB0) ;
    return (GrB_SUCCESS) ;
}

