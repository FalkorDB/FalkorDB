//------------------------------------------------------------------------------
// GB_new: create a new GraphBLAS matrix, but do not allocate A->{b,i,x}
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Creates a new matrix but does not allocate space for A->b, A->i, and A->x.
// See GB_new_bix instead.

// If the Ap_option is GB_Ap_calloc, the A->p and A->h are allocated and
// initialized, and A->magic is set to GB_MAGIC to denote a valid matrix.
// Otherwise, the matrix has not yet been completely initialized, and A->magic
// is set to GB_MAGIC2 to denote this.  This case only occurs internally in
// GraphBLAS.  The internal function that calls GB_new must then allocate or
// initialize A->p itself, and then set A->magic = GB_MAGIC when it does so.

// To allocate a full or bitmap matrix, the sparsity parameter
// is GxB_FULL or GxB_BITMAP.  The Ap_option is ignored.  For a full or
// bitmap matrix, only the header is allocated, if NULL on input.

// The GrB_Matrix object holds both a sparse vector and a sparse matrix.  A
// vector is represented as an vlen-by-1 matrix, but it is sometimes treated
// differently in various methods.  Vectors are never transposed via a
// descriptor, for example.

// The matrix may be created in an existing header, which case *Ahandle is
// non-NULL on input.  If an out-of-memory condition occurs, (*Ahandle) is
// returned as NULL, and the existing header is freed as well, if non-NULL on
// input.

#include "GB.h"

GrB_Info GB_new                 // create matrix, except for indices & values
(
    GrB_Matrix *Ahandle,        // handle of matrix to create
    const GrB_Type type,        // matrix type
    const int64_t vlen,         // length of each vector
    const int64_t vdim,         // number of vectors
    const GB_Ap_code Ap_option, // allocate A->p and A->h, or leave NULL
    const bool is_csc,          // true if CSC, false if CSR
    const int sparsity,         // hyper, sparse, bitmap, full, or auto
    const float hyper_switch,   // A->hyper_switch
    const int64_t plen          // size of A->p and A->h, if A hypersparse.
                                // Ignored if A is not hypersparse.
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Ahandle != NULL) ;
    ASSERT_TYPE_OK (type, "type for GB_new", GB0) ;
    ASSERT (vlen >= 0 && vlen <= GB_NMAX)
    ASSERT (vdim >= 0 && vdim <= GB_NMAX) ;

    //--------------------------------------------------------------------------
    // allocate the matrix header, if not already allocated on input
    //--------------------------------------------------------------------------

    bool allocated_header = false ;
    if ((*Ahandle) == NULL)
    {
        size_t header_size ;
        (*Ahandle) = GB_MALLOC (1, struct GB_Matrix_opaque, &header_size) ;
        if (*Ahandle == NULL)
        { 
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }
        allocated_header = true ;
        (*Ahandle)->static_header = false ;  // header of A has been malloc'd
        (*Ahandle)->header_size = header_size ;
    }
//  else
//  { 
//      // the header of A has been provided on input.  It may already be
//      // malloc'd, or it might be statically allocated in the caller. 
//      // (*Ahandle)->static_header is not modified.
//  }

    GrB_Matrix A = *Ahandle ;

    //--------------------------------------------------------------------------
    // initialize the matrix header
    //--------------------------------------------------------------------------

    // basic information
    A->magic = GB_MAGIC2 ;                 // object is not yet valid
    A->type = type ;
    A->user_name = NULL ;
    A->user_name_size = 0 ;     // no user_name yet
    A->logger = NULL ;          // no error logged yet
    A->logger_size = 0 ;

    // CSR/CSC format
    A->is_csc = is_csc ;

    // initial sparsity format
    bool A_is_hyper ;
    bool A_is_full_or_bitmap = false ;
    A->hyper_switch = hyper_switch ;
    A->bitmap_switch = GB_Global_bitmap_switch_matrix_get (vlen, vdim) ;
    A->sparsity_control = GxB_AUTO_SPARSITY ;

    if (sparsity == GxB_HYPERSPARSE)
    { 
        A_is_hyper = true ;             // force A to be hypersparse
    }
    else if (sparsity == GxB_SPARSE)
    { 
        A_is_hyper = false ;            // force A to be sparse
    }
    else if (sparsity == GxB_FULL || sparsity == GxB_BITMAP)
    { 
        A_is_full_or_bitmap = true ;    // force A to be full or bitmap
        A_is_hyper = false ;
    }
    else // auto: sparse or hypersparse
    { 
        // auto selection:  sparse if one vector or less or
        // if the global hyper_switch is negative; hypersparse otherwise.
        // Never select A to be full or bitmap for this case.
        A_is_hyper = !(vdim <= 1 || 0 > hyper_switch) ;
    }

    // matrix dimensions
    A->vlen = vlen ;
    A->vdim = vdim ;

    // content that is freed or reset in GB_phy_free
    if (A_is_full_or_bitmap)
    { 
        // A is full or bitmap
        A->plen = -1 ;
        A->nvec = vdim ;
        // all vectors present, unless matrix has a zero dimension 
        A->nvec_nonempty = (vlen > 0) ? vdim : 0 ;
    }
    else if (A_is_hyper)
    { 
        // A is hypersparse
        A->plen = (vdim == 1) ? 1 : GB_IMIN (plen, vdim) ;
        A->nvec = 0 ;                   // no vectors present
        A->nvec_nonempty = 0 ;
    }
    else
    { 
        // A is sparse
        A->plen = vdim ;
        A->nvec = vdim ;                // all vectors present
        A->nvec_nonempty = 0 ;
    }

    // no content yet
    A->p = NULL ; A->p_shallow = false ; A->p_size = 0 ;
    A->h = NULL ; A->h_shallow = false ; A->h_size = 0 ;
    A->Y = NULL ; A->Y_shallow = false ;
    A->b = NULL ; A->b_shallow = false ; A->b_size = 0 ;
    A->i = NULL ; A->i_shallow = false ; A->i_size = 0 ;
    A->x = NULL ; A->x_shallow = false ; A->x_size = 0 ;

    A->nvals = 0 ;
    A->nzombies = 0 ;
    A->jumbled = false ;
    A->Pending = NULL ;
    A->iso = false ;            // OK: if iso, burble in the caller

    //--------------------------------------------------------------------------
    // Allocate A->p and A->h if requested
    //--------------------------------------------------------------------------

    bool ok ;
    if (A_is_full_or_bitmap || Ap_option == GB_Ap_null)
    { 
        // A is not initialized yet; A->p and A->h are both NULL.
        A->magic = GB_MAGIC2 ;
        A->p = NULL ;
        A->h = NULL ;
        ok = true ;
    }
    else if (Ap_option == GB_Ap_calloc)
    {
        // Sets the vector pointers to zero, which defines all vectors as empty
        A->magic = GB_MAGIC ;
        A->p = GB_CALLOC (A->plen+1, int64_t, &(A->p_size)) ;
        ASSERT (A->p_size == GB_Global_memtable_size (A->p)) ;
        ok = (A->p != NULL) ;
        if (A_is_hyper)
        { 
            // since nvec is zero, there is never any need to initialize A->h
            A->h = GB_MALLOC (A->plen, int64_t, &(A->h_size)) ;
            ok = ok && (A->h != NULL) ;
        }
    }
    else // Ap_option == GB_Ap_malloc
    {
        // This is faster but can only be used internally by GraphBLAS since
        // the matrix is allocated but not yet completely initialized.  The
        // caller must set A->p [0..plen] and then set A->magic to GB_MAGIC,
        // before returning the matrix to the user application.
        A->magic = GB_MAGIC2 ;
        A->p = GB_MALLOC (A->plen+1, int64_t, &(A->p_size)) ;
        ASSERT (A->p_size == GB_Global_memtable_size (A->p)) ;
        ok = (A->p != NULL) ;
        if (A_is_hyper)
        { 
            A->h = GB_MALLOC (A->plen, int64_t, &(A->h_size)) ;
            ok = ok && (A->h != NULL) ;
        }
    }

    if (!ok)
    {
        // out of memory
        if (allocated_header)
        { 
            // free all of A, including the header
            GB_Matrix_free (Ahandle) ;
        }
        else
        { 
            // the header was not allocated here; only free the content of A
            GB_phybix_free (A) ;
        }
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // The vector pointers A->p are initialized only if Ap_calloc is true
    if (A->magic == GB_MAGIC)
    { 
        ASSERT_MATRIX_OK (A, "new matrix from GB_new", GB0) ;
    }
    return (GrB_SUCCESS) ;
}

