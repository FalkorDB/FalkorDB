//------------------------------------------------------------------------------
// GB_transplant: replace contents of one matrix with another
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Transplant A into C, and then free A.  If any part of A is shallow, or if A
// must be typecasted, a deep copy is made into C.  Prior content of C is
// ignored.  Then A is freed, except for any shallow components of A which are
// left untouched (after unlinking them from A).  The resulting matrix C is not
// shallow.  This function is not user-callable.  The new type of C (ctype)
// must be compatible with A->type.

// C->hyper_switch, C->bitmap_switch, C->sparsity_control, C->static_header,
// C->user_name, and C->user_name_size are not modified by the transplant.

#define GB_FREE_ALL                 \
{                                   \
    GB_phybix_free (C) ;            \
    GB_Matrix_free (Ahandle) ;      \
}

#include "GB.h"

GrB_Info GB_transplant          // transplant one matrix into another
(
    GrB_Matrix C,               // output matrix to overwrite with A
    const GrB_Type ctype,       // new type of C
    GrB_Matrix *Ahandle,        // input matrix to copy from and free
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (Ahandle != NULL) ;
    GrB_Matrix A = *Ahandle ;
    ASSERT (!GB_any_aliased (C, A)) ;

    ASSERT_MATRIX_OK (A, "A before transplant", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;    // zombies in A transplanted into C
    ASSERT (GB_JUMBLED_OK (A)) ;    // if A is jumbled, then C is jumbled
    ASSERT (GB_PENDING_OK (A)) ;    // pending tuples n A transplanted into C

    // C is about to be cleared, any pending work is OK
    ASSERT (C != NULL) ;
    ASSERT_TYPE_OK (ctype, "new type for C", GB0) ;
    ASSERT (GB_PENDING_OK (C)) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;

    // the ctype and A->type must be compatible.  C->type is ignored
    ASSERT (GB_Type_compatible (ctype, A->type)) ;

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;
    const bool A_iso = A->iso ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t anvals = A->nvals ;
    int64_t anz = GB_nnz_held (A) ;
    int64_t anvec = A->nvec ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // clear C and transplant the type, size, format, and pending tuples
    //--------------------------------------------------------------------------

    // free all content of C
    GB_phybix_free (C) ;

    ASSERT (!GB_PENDING (C)) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;

    // It is now safe to change the type and dimension of C
    C->type = ctype ;
    C->is_csc = A->is_csc ;
    C->vlen = avlen ;
    C->vdim = avdim ;
    C->nvec_nonempty = A->nvec_nonempty ;
    C->iso = A_iso ;        // OK:transplant

    // C is not shallow, and has no content yet
    ASSERT (!GB_is_shallow (C)) ;
    ASSERT (C->p == NULL) ;
    ASSERT (C->h == NULL) ;
    ASSERT (C->b == NULL) ;
    ASSERT (C->i == NULL) ;
    ASSERT (C->x == NULL) ;
    ASSERT (C->Y == NULL) ;
    ASSERT (C->Pending == NULL) ;

    // determine if C should be constructed as a bitmap or full matrix
    bool C_is_hyper = GB_IS_HYPERSPARSE (A) ;
    bool C_is_bitmap = GB_IS_BITMAP (A) ;
    bool C_is_full = GB_as_if_full (A) && !C_is_bitmap && !C_is_hyper ;

    //--------------------------------------------------------------------------
    // transplant A->Y into C->Y
    //--------------------------------------------------------------------------

    if (C_is_hyper && A->Y != NULL)
    {
        if (A->Y_shallow || GB_is_shallow (A->Y))
        { 
            // A->Y is shallow, so create a deep copy for C
            GB_OK (GB_dup (&(C->Y), A->Y, Werk)) ;
        }
        else
        { 
            // A->Y is not shallow, so transplant it into C
            C->Y = A->Y ;
            A->Y = NULL ;
            A->Y_shallow = false ;
        }
        C->Y_shallow = false ;
    }

    //--------------------------------------------------------------------------
    // transplant pending tuples from A to C
    //--------------------------------------------------------------------------

    C->Pending = A->Pending ;
    A->Pending = NULL ;

    //--------------------------------------------------------------------------
    // allocate new space for C->b, C->i, and C->x if A is shallow
    //--------------------------------------------------------------------------

    // C->b is allocated only if A->b exists and is shallow.
    // C->i is not allocated if C is full or bitmap.
    // C->x is allocated if A->x is shallow, or if the type is changing

    bool allocate_Cb = (A->b_shallow) && (C_is_bitmap) ;
    bool allocate_Ci = (A->i_shallow) && (!(C_is_full || C_is_bitmap)) ;
    bool allocate_Cx = (A->x_shallow || C->type != A->type) ;

    // allocate new components if needed
    bool ok = true ;

    if (allocate_Cb)
    { 
        // allocate new C->b component
        C->b = GB_MALLOC (anz, int8_t, &(C->b_size)) ;
        ok = ok && (C->b != NULL) ;
    }

    if (allocate_Ci)
    { 
        // allocate new C->i component
        C->i = GB_MALLOC (anz, int64_t, &(C->i_size)) ;
        ok = ok && (C->i != NULL) ;
    }

    if (allocate_Cx)
    { 
        // allocate new C->x component; use calloc if C is bitmap
        C->x = GB_XALLOC (C_is_bitmap, A_iso, anz, // x:OK
            C->type->size, &(C->x_size)) ;
        ok = ok && (C->x != NULL) ;
    }

    if (!ok)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // transplant or copy A->x numerical values
    //--------------------------------------------------------------------------

    ASSERT_TYPE_OK (C->type, "target C->type for values", GB0) ;
    ASSERT_TYPE_OK (A->type, "source A->type for values", GB0) ;

    if (C->type == A->type)
    {
        // types match
        if (A->x_shallow)
        { 
            // A is shallow so make a deep copy; no typecast needed
            GB_OK (GB_cast_matrix (C, A)) ;
            A->x = NULL ;
        }
        else
        { 
            // OK to move pointers instead
            C->x = A->x ; C->x_size = A->x_size ;
            A->x = NULL ;
        }
    }
    else
    {
        // types differ, must typecast from A to C.
        GB_OK (GB_cast_matrix (C, A)) ;
        if (!A->x_shallow)
        { 
            GB_FREE (&(A->x), A->x_size) ;
        }
        A->x = NULL ;
    }

    ASSERT (A->x == NULL) ;     // has been freed or removed
    A->x_shallow = false ;
    C->x_shallow = false ;

    //--------------------------------------------------------------------------
    // transplant A->p vector pointers and A->h hyperlist
    //--------------------------------------------------------------------------

    if (C_is_full || C_is_bitmap)
    { 

        //----------------------------------------------------------------------
        // C is full or bitmap: C->p and C->h do not exist
        //----------------------------------------------------------------------

        C->plen = -1 ;
        C->nvec = avdim ;

        // free any non-shallow A->p and A->h content of A
        GB_phy_free (A) ;

    }
    else if (A->p_shallow || A->h_shallow)
    {

        //----------------------------------------------------------------------
        // A->p or A->h are shallow copies another matrix; make a deep copy
        //----------------------------------------------------------------------

        int nth = GB_nthreads (anvec, chunk, nthreads_max) ;

        if (A->h != NULL)
        {
            // A is hypersparse, create new C->p and C->h
            C->plen = GB_IMAX (1, anvec) ;
            C->nvec = anvec ;
            C->p = GB_MALLOC (C->plen+1, int64_t, &(C->p_size)) ;
            C->h = GB_MALLOC (C->plen  , int64_t, &(C->h_size)) ;
            if (C->p == NULL || C->h == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            // copy A->p and A->h into the newly created C->p and C->h
            GB_memcpy (C->p, A->p, (anvec+1) * sizeof (int64_t), nth) ;
            GB_memcpy (C->h, A->h,  anvec    * sizeof (int64_t), nth) ;
        }
        else
        {
            // A is sparse, create new C->p
            C->plen = avdim ;
            C->nvec = avdim ;
            C->p = GB_MALLOC (C->plen+1, int64_t, &(C->p_size)) ;
            if (C->p == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            // copy A->p into the newly created C->p
            GB_memcpy (C->p, A->p, (avdim+1) * sizeof (int64_t), nth) ;
        }

        // free any non-shallow A->p and A->h content of A
        GB_phy_free (A) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // both A->p and A->h are not shallow: quick transplant into C
        //----------------------------------------------------------------------

        // Quick transplant of A->p and A->h into C.  This works for both
        // sparse and hypersparse cases.
        ASSERT (C->p == NULL) ;
        ASSERT (C->h == NULL) ;
        C->p = A->p ; C->p_size = A->p_size ;
        C->h = A->h ; C->h_size = A->h_size ;
        C->plen = A->plen ;
        C->nvec = anvec ;
    }

    // A->p and A->h have been freed or removed from A
    A->p = NULL ;
    A->h = NULL ;
    A->p_shallow = false ;
    A->h_shallow = false ;
    C->p_shallow = false ;
    C->h_shallow = false ;

    C->nvals = anvals ;
    C->magic = GB_MAGIC ;           // C is now initialized
    A->magic = GB_MAGIC2 ;          // A is now invalid

    //--------------------------------------------------------------------------
    // transplant or copy A->i row indices
    //--------------------------------------------------------------------------

    if (C_is_full || C_is_bitmap)
    { 

        //----------------------------------------------------------------------
        // C is full or bitmap
        //----------------------------------------------------------------------

        // C is full or bitmap; C->i stays NULL
        C->i = NULL ;

    }
    else if (A->i_shallow)
    { 

        //----------------------------------------------------------------------
        // A->i is a shallow copy of another matrix, so we need a deep copy
        //----------------------------------------------------------------------

        // copy A->i into C->i
        GB_memcpy (C->i, A->i, anz * sizeof (int64_t), nthreads) ;
        A->i = NULL ;
        A->i_shallow = false ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // A->i is not shallow, so just transplant the pointer from A to C
        //----------------------------------------------------------------------

        C->i = A->i ; C->i_size = A->i_size ;
        A->i = NULL ;
        A->i_shallow = false ;
    }

    C->i_shallow = false ;
    C->nzombies = A->nzombies ;     // zombies may have been transplanted into C
    C->jumbled = A->jumbled ;       // C is jumbled if A is jumbled

    //--------------------------------------------------------------------------
    // transplant or copy A->b bitmap
    //--------------------------------------------------------------------------

    if (!C_is_bitmap)
    { 

        //----------------------------------------------------------------------
        // A is not bitmap; A->b does not exist
        //----------------------------------------------------------------------

        // C is not bitmap; C->b stays NULL
        C->b = NULL ;

    }
    else if (A->b_shallow)
    { 

        //----------------------------------------------------------------------
        // A->b is a shallow copy of another matrix, so we need a deep copy
        //----------------------------------------------------------------------

        // copy A->b into C->b
        GB_memcpy (C->b, A->b, anz * sizeof (int8_t), nthreads) ;
        A->b = NULL ;
        A->b_shallow = false ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // A->b is not shallow, so just transplant the pointer from A to C
        //----------------------------------------------------------------------

        C->b = A->b ; C->b_size = A->b_size ;
        A->b = NULL ;
        A->b_shallow = false ;
    }

    C->b_shallow = false ;

    //--------------------------------------------------------------------------
    // free A and return result
    //--------------------------------------------------------------------------

    GB_Matrix_free (Ahandle) ;
    ASSERT_MATRIX_OK (C, "C after transplant", GB0) ;
    return (GrB_SUCCESS) ;
}

