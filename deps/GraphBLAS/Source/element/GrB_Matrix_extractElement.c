//------------------------------------------------------------------------------
// GrB_Matrix_extractElement: extract a single entry from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extract the value of single scalar, x = A(i,j), typecasting from the
// type of A to the type of x, as needed.

// Returns GrB_SUCCESS if A(i,j) is present, and sets x to its value.
// If A(i,j) is not present: if x is a bare scalar, x is unmodified and
// GrB_NO_VALUE is returned; if x is a GrB_scalar, x is returned as empty,
// and GrB_SUCCESS is returned.

#include "GB.h"

#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// GrB_Matrix_extractElement_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_extractElement_Scalar   // S = A(i,j)
(
    GrB_Scalar S,                       // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    uint64_t i,                         // row index
    uint64_t j                          // column index
)
{

    //--------------------------------------------------------------------------
    // check inputs (just the GrB_Scalar S)
    //--------------------------------------------------------------------------

    GB_WHERE2 (S, A, "GrB_Matrix_extractElement_Scalar (s, A, row, col)") ;
    GB_RETURN_IF_NULL (S) ;
    GB_RETURN_IF_NULL (A) ;

    //--------------------------------------------------------------------------
    // ensure S is bitmap
    //--------------------------------------------------------------------------

    if (!GB_IS_BITMAP (S))
    { 
        // convert S to bitmap
        GB_OK (GB_convert_any_to_bitmap ((GrB_Matrix) S, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // extract the entry (also checks the inputs A, i, and j)
    //--------------------------------------------------------------------------

    void *x = S->x ;

    switch (S->type->code)
    {
        case GB_BOOL_code    : 
            info = GrB_Matrix_extractElement_BOOL ((bool *) x, A, i, j) ;
            break ;

        case GB_INT8_code    : 
            info = GrB_Matrix_extractElement_INT8 ((int8_t *) x, A, i, j) ;
            break ;

        case GB_INT16_code   : 
            info = GrB_Matrix_extractElement_INT16 ((int16_t *) x, A, i, j) ;
            break ;

        case GB_INT32_code   : 
            info = GrB_Matrix_extractElement_INT32 ((int32_t *) x, A, i, j) ;
            break ;

        case GB_INT64_code   : 
            info = GrB_Matrix_extractElement_INT64 ((int64_t *) x, A, i, j) ;
            break ;

        case GB_UINT8_code   : 
            info = GrB_Matrix_extractElement_UINT8 ((uint8_t *) x, A, i, j) ;
            break ;

        case GB_UINT16_code  : 
            info = GrB_Matrix_extractElement_UINT16 ((uint16_t *) x, A, i, j) ;
            break ;

        case GB_UINT32_code  : 
            info = GrB_Matrix_extractElement_UINT32 ((uint32_t *) x, A, i, j) ;
            break ;

        case GB_UINT64_code  : 
            info = GrB_Matrix_extractElement_UINT64 ((uint64_t *) x, A, i, j) ;
            break ;

        case GB_FP32_code    : 
            info = GrB_Matrix_extractElement_FP32 ((float *) x, A, i, j) ;
            break ;

        case GB_FP64_code    : 
            info = GrB_Matrix_extractElement_FP64 ((double *) x, A, i, j) ;
            break ;

        case GB_FC32_code    : 
            info = GxB_Matrix_extractElement_FC32 ((GxB_FC32_t *) x, A, i, j) ;
            break ;

        case GB_FC64_code    : 
            info = GxB_Matrix_extractElement_FC64 ((GxB_FC64_t *) x, A, i, j) ;
            break ;

        case GB_UDT_code     : 
            info = GrB_Matrix_extractElement_UDT ((void *) x, A, i, j) ;
            break ;

        default: ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    bool entry_present = (info == GrB_SUCCESS) ;
    bool no_entry = (info == GrB_NO_VALUE) ;
    S->b [0] = entry_present ;
    S->nvals = entry_present ? 1 : 0 ;
    return ((entry_present || no_entry) ? GrB_SUCCESS : info) ;
}

//------------------------------------------------------------------------------
// GB_Matrix_find_entry: finds the position of a single entry A(i,j)
//------------------------------------------------------------------------------

// Finds A(i,j) in the matrix A, which must not be jumbled.  The matrix may
// have zombies.  Pending tuples are ignored and not searched; the method
// returns false if A(i,j) is a pending tuple.

GB_STATIC_INLINE void GB_Matrix_find_entry
(
    // output:
    int64_t *pleft,     // position of the entry, if A(i,j) found
    bool *found,        // true if A(i,j) found
    bool *is_zombie,    // true if A(i,j) is found, but is a zombie
    // input
    const GrB_Matrix A,
    int64_t i,
    int64_t j
)
{

    ASSERT (!A->jumbled) ;
    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;

    if (Ap != NULL)
    {

        //----------------------------------------------------------------------
        // A is sparse or hypersparse
        //----------------------------------------------------------------------

        int64_t pA_start, pA_end ;
        if (A->h != NULL)
        { 

            //------------------------------------------------------------------
            // A is hypersparse: look for j in hyperlist A->h [0 ... A->nvec-1]
            //------------------------------------------------------------------

            void *A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
            void *A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
            void *A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
            const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;
            int64_t k = GB_hyper_hash_lookup (A->p_is_32, A->j_is_32,
                A->h, A->nvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits,
                j, &pA_start, &pA_end) ;
            (*found) = (k >= 0) ;

            #ifdef GB_DEBUG
            if (*found)
            {
                GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
                ASSERT (j == GB_IGET (Ah, k)) ;
            }
            #endif

        }
        else
        { 

            //------------------------------------------------------------------
            // A is sparse: look in the jth vector
            //------------------------------------------------------------------

            pA_start = GB_IGET (Ap, j);
            pA_end   = GB_IGET (Ap, j+1) ;
        }

        // vector j has been found, now look for index i
        (*pleft) = pA_start ;
        int64_t pright = pA_end - 1 ;

        // Time taken for this step is at most O(log(nnz(A(:,j))).
        (*found) = GB_binary_search_zombie (i, A->i, A->i_is_32, pleft,
            &pright, A->nzombies > 0, is_zombie) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // A is bitmap or full
        //----------------------------------------------------------------------

        (*pleft) = i + j * (A->vlen) ;
        const int8_t *restrict Ab = A->b ;
        if (Ab != NULL)
        { 
            // A is bitmap
            (*found) = (Ab [(*pleft)] == 1) ;
        }
        else
        { 
            // A is full
            (*found) = true ;
        }
        (*is_zombie) = false ;
    }
}

//------------------------------------------------------------------------------
// GrB_Matrix_extractElement_TYPE and GxB_Matrix_isStoredElement
//------------------------------------------------------------------------------

#define GB_WHERE_STRING "GrB_Matrix_extractElement (&x, A, row, col)"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_BOOL
#define GB_XTYPE bool
#define GB_XCODE GB_BOOL_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_INT8
#define GB_XTYPE int8_t
#define GB_XCODE GB_INT8_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_INT16
#define GB_XTYPE int16_t
#define GB_XCODE GB_INT16_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_INT32
#define GB_XTYPE int32_t
#define GB_XCODE GB_INT32_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_INT64
#define GB_XTYPE int64_t
#define GB_XCODE GB_INT64_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_UINT8
#define GB_XTYPE uint8_t
#define GB_XCODE GB_UINT8_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_UINT16
#define GB_XTYPE uint16_t
#define GB_XCODE GB_UINT16_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_UINT32
#define GB_XTYPE uint32_t
#define GB_XCODE GB_UINT32_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_UINT64
#define GB_XTYPE uint64_t
#define GB_XCODE GB_UINT64_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_FP32
#define GB_XTYPE float
#define GB_XCODE GB_FP32_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_FP64
#define GB_XTYPE double
#define GB_XCODE GB_FP64_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GxB_Matrix_extractElement_FC32
#define GB_XTYPE GxB_FC32_t
#define GB_XCODE GB_FC32_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GxB_Matrix_extractElement_FC64
#define GB_XTYPE GxB_FC64_t
#define GB_XCODE GB_FC64_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_UDT_EXTRACT
#define GB_EXTRACT_ELEMENT GrB_Matrix_extractElement_UDT
#define GB_XTYPE void
#define GB_XCODE GB_UDT_code
#include "element/factory/GB_Matrix_extractElement.c"

#define GB_EXTRACT_ELEMENT GxB_Matrix_isStoredElement
#include "element/factory/GB_Matrix_extractElement.c"

