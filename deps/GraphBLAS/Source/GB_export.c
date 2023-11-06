//------------------------------------------------------------------------------
// GB_export: export a matrix or vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// No conversion is done, except to convert to non-iso if requested.  The
// matrix is exported in its current sparsity structure and by-row/by-col
// format.

#include "GB_export.h"

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE (&Ap_new, Ap_new_size) ;    \
    GB_FREE (&Ah_new, Ah_new_size) ;    \
}

GrB_Info GB_export      // export/unpack a matrix in any format
(
    bool unpacking,     // unpack if true, export and free if false

    GrB_Matrix *A,      // handle of matrix to export and free, or unpack
    GrB_Type *type,     // type of matrix to export
    GrB_Index *vlen,    // vector length
    GrB_Index *vdim,    // vector dimension
    bool is_sparse_vector,      // true if A is a sparse GrB_Vector

    // the 5 arrays:
    GrB_Index **Ap,     // pointers
    GrB_Index *Ap_size, // size of Ap in bytes

    GrB_Index **Ah,     // vector indices
    GrB_Index *Ah_size, // size of Ah in bytes

    int8_t **Ab,        // bitmap
    GrB_Index *Ab_size, // size of Ab in bytes

    GrB_Index **Ai,     // indices
    GrB_Index *Ai_size, // size of Ai in bytes

    void **Ax,          // values
    GrB_Index *Ax_size, // size of Ax in bytes

    // additional information for specific formats:
    GrB_Index *nvals,   // # of entries for bitmap format.
    bool *jumbled,      // if true, sparse/hypersparse may be jumbled.
    GrB_Index *nvec,    // size of Ah for hypersparse format.

    // information for all formats:
    int *sparsity,      // hypersparse, sparse, bitmap, or full
    bool *is_csc,       // if true then matrix is by-column, else by-row
    bool *iso,          // if true then A is iso and only one entry is returned
                        // in Ax, regardless of nvals(A).
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    int64_t *Ap_new = NULL ; size_t Ap_new_size = 0 ;
    int64_t *Ah_new = NULL ; size_t Ah_new_size = 0 ;
    ASSERT (A != NULL) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*A) ;
    ASSERT_MATRIX_OK (*A, "A to export", GB0) ;
    ASSERT (!GB_ZOMBIES (*A)) ;
    ASSERT (GB_JUMBLED_OK (*A)) ;
    ASSERT (!GB_PENDING (*A)) ;

    GB_RETURN_IF_NULL (type) ;
    GB_RETURN_IF_NULL (vlen) ;
    GB_RETURN_IF_NULL (vdim) ;
    GB_RETURN_IF_NULL (Ax) ;
    GB_RETURN_IF_NULL (Ax_size) ;

    int s = GB_sparsity (*A) ;

    switch (s)
    {
        case GxB_HYPERSPARSE : 
            GB_RETURN_IF_NULL (nvec) ;
            GB_RETURN_IF_NULL (Ah) ; GB_RETURN_IF_NULL (Ah_size) ;
            // fall through to the sparse case

        case GxB_SPARSE : 
            if (is_sparse_vector)
            { 
                GB_RETURN_IF_NULL (nvals) ;
            }
            else
            { 
                GB_RETURN_IF_NULL (Ap) ; GB_RETURN_IF_NULL (Ap_size) ;
            }
            GB_RETURN_IF_NULL (Ai) ; GB_RETURN_IF_NULL (Ai_size) ;
            break ;

        case GxB_BITMAP : 
            GB_RETURN_IF_NULL (nvals) ;
            GB_RETURN_IF_NULL (Ab) ; GB_RETURN_IF_NULL (Ab_size) ;
            // fall through to the full case

        case GxB_FULL : 
            break ;

        default: ;
    }

    //--------------------------------------------------------------------------
    // allocate new space for Ap and Ah if unpacking
    //--------------------------------------------------------------------------

    int64_t avdim = (*A)->vdim ;
    int64_t plen_new, nvec_new ;
    if (unpacking)
    {
        plen_new = (avdim == 0) ? 0 : 1 ;
        nvec_new = (avdim == 1) ? 1 : 0 ;
        Ap_new = GB_CALLOC (plen_new+1, int64_t, &(Ap_new_size)) ;
        if (avdim > 1)
        { 
            // A is sparse if avdim <= 1, hypersparse if avdim > 1
            Ah_new = GB_CALLOC (1, int64_t, &(Ah_new_size)) ;
        }
        if (Ap_new == NULL || (avdim > 1 && Ah_new == NULL))
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // ensure A is non-iso if requested, or export A as-is
    //--------------------------------------------------------------------------

    if (iso == NULL)
    {
        // ensure A is non-iso
        // set A->iso = false   OK
        if ((*A)->iso)
        { 
            GBURBLE ("(iso to non-iso export) ") ;
        }
        GB_OK (GB_convert_any_to_non_iso (*A, true)) ;
        ASSERT (!((*A)->iso)) ;
    }
    else
    {
        // do not convert the matrix; export A as-is, either iso or non-iso
        (*iso) = (*A)->iso ;
        if (*iso)
        { 
            GBURBLE ("(iso export) ") ;
        }
    }

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    (*type) = (*A)->type ;
    (*vlen) = (*A)->vlen ;
    (*vdim) = avdim ;

    // export A->x
    #ifdef GB_MEMDUMP
    printf ("export A->x from memtable: %p\n", (*A)->x) ;   // MEMDUMP
    #endif
    GB_Global_memtable_remove ((*A)->x) ;
    (*Ax) = (*A)->x ; (*A)->x = NULL ;
    (*Ax_size) = (*A)->x_size ;

    switch (s)
    {
        case GxB_HYPERSPARSE : 
            (*nvec) = (*A)->nvec ;

            // export A->h
            #ifdef GB_MEMDUMP
            printf ("export A->h from memtable: %p\n", (*A)->h) ;   // MEMDUMP
            #endif
            GB_Global_memtable_remove ((*A)->h) ;
            (*Ah) = (GrB_Index *) ((*A)->h) ; (*A)->h = NULL ;
            (*Ah_size) = (*A)->h_size ;
            // fall through to the sparse case

        case GxB_SPARSE : 
            if (jumbled != NULL)
            { 
                (*jumbled) = (*A)->jumbled ;
            }

            // export A->p, unless A is a sparse vector in CSC format
            if (is_sparse_vector)
            { 
                (*nvals) = (*A)->p [1] ;
            }
            else
            { 
                #ifdef GB_MEMDUMP
                printf ("export A->p from memtable: %p\n", (*A)->p) ; // MEMDUMP
                #endif
                GB_Global_memtable_remove ((*A)->p) ;
                (*Ap) = (GrB_Index *) ((*A)->p) ; (*A)->p = NULL ;
                (*Ap_size) = (*A)->p_size ;
            }

            // export A->i
            #ifdef GB_MEMDUMP
            printf ("export A->i from memtable: %p\n", (*A)->i) ;   // MEMDUMP
            #endif
            GB_Global_memtable_remove ((*A)->i) ;
            (*Ai) = (GrB_Index *) ((*A)->i) ; (*A)->i = NULL ;
            (*Ai_size) = (*A)->i_size ;
            break ;

        case GxB_BITMAP : 
            (*nvals) = (*A)->nvals ;

            // export A->b
            #ifdef GB_MEMDUMP
            printf ("export A->b from memtable: %p\n", (*A)->b) ;   // MEMDUMP
            #endif
            GB_Global_memtable_remove ((*A)->b) ;
            (*Ab) = (*A)->b ; (*A)->b = NULL ;
            (*Ab_size) = (*A)->b_size ;

        case GxB_FULL : 

        default: ;
    }

    if (sparsity != NULL)
    { 
        (*sparsity) = s ;
    }
    if (is_csc != NULL)
    { 
        (*is_csc) = (*A)->is_csc ;
    }

    //--------------------------------------------------------------------------
    // free or clear the GrB_Matrix
    //--------------------------------------------------------------------------

    // both export and unpack free the hyper_hash, A->Y

    if (unpacking)
    { 
        // unpack: clear the matrix, leaving it hypersparse (or sparse if
        // it is a vector (vdim of 1) or has vdim of zero)
        GB_phybix_free (*A) ;
        (*A)->plen = plen_new ;
        (*A)->nvec = nvec_new ;
        (*A)->p = Ap_new ; (*A)->p_size = Ap_new_size ;
        (*A)->h = Ah_new ; (*A)->h_size = Ah_new_size ;
        (*A)->magic = GB_MAGIC ;
        ASSERT_MATRIX_OK (*A, "A unpacked", GB0) ;
    }
    else
    { 
        // export: free the header of A, and A->p if A is a sparse GrB_Vector
        GB_Matrix_free (A) ;
        ASSERT ((*A) == NULL) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

