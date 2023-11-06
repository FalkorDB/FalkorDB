//------------------------------------------------------------------------------
// GB_extractTuples: extract all the tuples from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// Extracts all tuples from a matrix, like [I,J,X] = find (A).  If any
// parameter I, J and/or X is NULL, then that component is not extracted.  The
// size of the I, J, and X arrays (those that are not NULL) is given by nvals,
// which must be at least as large as GrB_nvals (&nvals, A).  The values in the
// matrix are typecasted to the type of X, as needed.

// This function does the work for the user-callable GrB_*_extractTuples
// functions, and helps build the tuples for GB_concat_hyper.

// If A is iso and X is not NULL, the iso scalar Ax [0] is expanded into X.

#include "GB.h"

#define GB_FREE_ALL                             \
{                                               \
    GB_Matrix_free (&T) ;                       \
    GB_FREE_WORK (&Ap, Ap_size) ;               \
    GB_FREE_WORK (&X_bitmap, X_bitmap_size) ;   \
}

GrB_Info GB_extractTuples       // extract all tuples from a matrix
(
    GrB_Index *I_out,           // array for returning row indices of tuples
    GrB_Index *J_out,           // array for returning col indices of tuples
    void *X,                    // array for returning values of tuples
    GrB_Index *p_nvals,         // I,J,X size on input; # tuples on output
    const GB_Type_code xcode,   // type of array X
    const GrB_Matrix A,         // matrix to extract tuples from
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;
    GB_void *restrict X_bitmap = NULL ; size_t X_bitmap_size = 0 ;
    int64_t *restrict Ap       = NULL ; size_t Ap_size = 0 ;

    ASSERT_MATRIX_OK (A, "A to extract", GB0) ;
    ASSERT (p_nvals != NULL) ;

    // delete any lingering zombies and assemble any pending tuples;
    // allow A to remain jumbled
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;

    GB_BURBLE_DENSE (A, "(A %s) ") ;
    ASSERT (xcode <= GB_UDT_code) ;
    const GB_Type_code acode = A->type->code ;
    const size_t asize = A->type->size ;

    // xcode and A must be compatible
    if (!GB_code_compatible (xcode, acode))
    { 
        return (GrB_DOMAIN_MISMATCH) ;
    }

    const int64_t anz = GB_nnz (A) ;
    if (anz == 0)
    { 
        // no work to do
        (*p_nvals) = 0 ;
        return (GrB_SUCCESS) ;
    }

    int64_t nvals = *p_nvals ;          // size of I,J,X on input
    if (nvals < anz && (I_out != NULL || J_out != NULL || X != NULL))
    { 
        // output arrays are not big enough
        return (GrB_INSUFFICIENT_SPACE) ;
    }

    //-------------------------------------------------------------------------
    // determine the number of threads to use
    //-------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz + A->nvec, chunk, nthreads_max) ;

    //-------------------------------------------------------------------------
    // handle the CSR/CSC format
    //--------------------------------------------------------------------------

    GrB_Index *I, *J ;
    if (A->is_csc)
    { 
        I = I_out ;
        J = J_out ;
    }
    else
    { 
        I = J_out ;
        J = I_out ;
    }

    //--------------------------------------------------------------------------
    // bitmap case
    //--------------------------------------------------------------------------

    if (GB_IS_BITMAP (A))
    {

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        bool need_typecast = (X != NULL) && (xcode != acode) ;
        if (need_typecast)
        { 
            // X must be typecasted
            int64_t anzmax = GB_IMAX (anz, 1) ;
            X_bitmap = GB_MALLOC_WORK (anzmax*asize, GB_void, &X_bitmap_size) ;
        }
        Ap = GB_MALLOC_WORK (A->vdim+1, int64_t, &Ap_size) ;
        if (Ap == NULL || (need_typecast && X_bitmap == NULL))
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // extract the tuples
        //----------------------------------------------------------------------

        // TODO: pass xcode to GB_convert_bitmap_worker and let it do the
        // typecasting.  This works for now, however.

        // if A is iso, GB_convert_bitmap_worker expands the iso scalar
        // into its result, X or X_bitmap

        GB_OK (GB_convert_bitmap_worker (Ap, (int64_t *) I, (int64_t *) J,
            (GB_void *) (need_typecast ? X_bitmap : X), NULL, A, Werk)) ;

        //----------------------------------------------------------------------
        // typecast X if needed
        //----------------------------------------------------------------------

        if (need_typecast)
        { 
            // typecast the values from X_bitmap into X, using a temporary
            // full anz-by-1 matrix T
            ASSERT (X != NULL) ;
            ASSERT (xcode != acode) ;
            GB_CLEAR_STATIC_HEADER (T, &T_header) ;
            GB_OK (GB_new (&T, // full, existing header
                A->type, anz, 1, GB_Ap_null, true, GxB_FULL, 0, 0)) ;
            T->x = X_bitmap ;
            T->x_shallow = true ;
            T->magic = GB_MAGIC ;
            T->plen = -1 ;
            T->nvec = 1 ;
            ASSERT_MATRIX_OK (T, "T to cast_array", GB0) ;
            GB_OK (GB_cast_array ((GB_void *) X, xcode, T, nthreads)) ;
            GB_Matrix_free (&T) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // sparse, hypersparse, or full case
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // extract the row indices
        //----------------------------------------------------------------------

        if (I != NULL)
        {
            if (A->i == NULL)
            {
                // A is full; construct the row indices
                int64_t avlen = A->vlen ;
                int64_t p ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (p = 0 ; p < anz ; p++)
                { 
                    I [p] = (p % avlen) ;
                }
            }
            else
            { 
                GB_memcpy (I, A->i, anz * sizeof (int64_t), nthreads) ;
            }
        }

        //----------------------------------------------------------------------
        // extract the column indices
        //----------------------------------------------------------------------

        if (J != NULL)
        {
            GB_OK (GB_extract_vector_list ((int64_t *) J, A, Werk)) ;
        }

        //----------------------------------------------------------------------
        // extract the values
        //----------------------------------------------------------------------

        if (X != NULL)
        {
            if (A->iso)
            { 
                // typecast the scalar and expand it into X
                size_t xsize = GB_code_size (xcode, asize) ;
                GB_void scalar [GB_VLA(xsize)] ;
                GB_cast_scalar (scalar, xcode, A->x, acode, asize) ;
                GB_expand_iso (X, anz, scalar, xsize) ;
            }
            else if (xcode == acode)
            { 
                // copy the values from A into X, no typecast
                GB_memcpy (X, A->x, anz * asize, nthreads) ;
            }
            else
            { 
                // typecast the values from A into X
                ASSERT (X != NULL) ;
                ASSERT_MATRIX_OK (A, "A to cast_array", GB0) ;
                GB_OK (GB_cast_array ((GB_void *) X, xcode, A, nthreads)) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result 
    //--------------------------------------------------------------------------

    *p_nvals = anz ;            // number of tuples extracted
    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

