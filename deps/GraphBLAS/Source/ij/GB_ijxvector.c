//------------------------------------------------------------------------------
// GB_ijxvector: extract a list of indices or values from a GrB_Vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input vector List describes a list of integers or values to be used by
// GrB_assign, GxB_subassign, GrB_extract, or GrB_build, as I, J, or X.
//
// Descriptor settings: for I and J lists passed to GrB_assign,
// GxB_subassign and GrB_extract, and for I,J,X lists passed to GrB_build:
//
//  default:        use List->x as GxB_LIST of indices
//  use indices:    use List->i as GxB_LIST of indices
//  stride:         use List->x for Icolon [3] (signed integers);
//                  List->x must have exactly 3 entries (lo:inc:hi),
//                  and is typecast to Icolon of type int64_t.
//                  becomes GxB_STRIDE.
//
// If the List vector is NULL, it is treated as GrB_ALL; no need for the
// descriptor.  Since the List vector can contain signed integers, there is no
// need for a RANGE or BACKWARDS descriptor.
//
// Descriptor fields that control the interpretation of the List:
//
//      GxB_ROWINDEX_LIST       how to interpret the GrB_Vector I
//      GxB_COLINDEX_LIST       how to interpret the GrB_Vector J
//      GxB_VALUE_LIST          how to interpret the GrB_Vector X for GrB_build
//
// values they can be set to:
//
//      GrB_DEFAULT (0)         use List->x
//      GxB_USE_VALUES (0)      use List->x (same as GrB_DEFAULT)
//      GxB_USE_INDICES (7060)  use List->i
//      GxB_IS_STRIDE (7061)    use List->x, size 3, for Icolon
//
// GrB_build does not allow GxB_IS_STRIDE for I, J, or X.
//
// GrB_extractTuples, with indices returned in GrB_Vectors I, J, and X will use
// none of these settings.  It will always return its results in List->x.  It
// does not use this method and ignores the descriptor settings above.

#include "GB_ij.h"
#include "container/GB_container.h"

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE_MEMORY (&I2, I2_size) ;     \
    if (I != NULL && I_size > 0)        \
    {                                   \
        GB_FREE_MEMORY (&I, I_size) ;   \
    }                                   \
    GB_Matrix_free (&T) ;               \
}

//------------------------------------------------------------------------------
// GB_stride: create a stride, I = begin:inc:end
//------------------------------------------------------------------------------

// GrB_assign, GxB_subassign, and GrB_extract all expect a list of unsigned
// integers for their list I.  The stride can be negative, which is handled by
// setting ni to one of 3 special values:
//
//      GxB_RANGE       I = [begin, end, 1]
//      GxB_BACKWARDS   I = [begin, end, -stride]
//      GxB_STRIDE      I = [begin, end, +stride]
//
// Tyis method is not used for GrB_build.

static inline GrB_Info GB_stride
(
    // input:
    int64_t stride_begin,
    int64_t stride_inc,
    int64_t stride_end,
    // output:
    void **I_handle,        // the list I; may be GrB_ALL
    int64_t *ni_handle,     // the length of I, or special (GxB_RANGE)
    size_t *I_size_handle,  // if > 0, I has been allocated by this
    GrB_Type *I_type_handle // the type of I: always GrB_UINT64
)
{
    ASSERT ((*I_handle) == NULL) ;
    ASSERT ((*I_size_handle) == 0) ;
    (*I_handle) = GB_CALLOC_MEMORY (3, sizeof (uint64_t), I_size_handle) ;
    if ((*I_handle) == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    uint64_t *U64 = (uint64_t *) (*I_handle) ;
    U64 [GxB_BEGIN] = (uint64_t) stride_begin ;
    U64 [GxB_END  ] = (uint64_t) stride_end ;
    if (stride_inc == 1)
    { 
        // in MATLAB notation: begin:1:end
        U64 [GxB_INC] = 1 ;
        (*ni_handle) = GxB_RANGE ;
    }
    else if (stride_inc < 0)
    { 
        // in MATLAB notation: begin:stride:end, where stride < 0
        U64 [GxB_INC] = (uint64_t) (-stride_inc) ;
        (*ni_handle) = GxB_BACKWARDS ;
    }
    else
    { 
        // in MATLAB notation: begin:stride:end, where stride > 1
        U64 [GxB_INC] = (uint64_t) stride_inc ;
        (*ni_handle) = GxB_STRIDE ;
    }
    (*I_type_handle) = GrB_UINT64 ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_ijxvector: intrepret the List
//------------------------------------------------------------------------------

GrB_Info GB_ijxvector
(
    // input:
    GrB_Vector List,        // defines the list, either from List->x or List-i.
                            // If List is NULL, it defines I = GrB_ALL.
    bool need_copy,         // if true, I must be allocated
    int which,              // 0: I list, 1: J list, 2: X list
    const GrB_Descriptor desc,  // row_list, col_list, val_list descriptors
    bool is_build,          // if true, method is GrB_build; otherwise, it is
                            // assign, subassign, or extract
    // output:
    void **I_handle,        // the list I; may be GrB_ALL
    int64_t *ni_handle,     // the length of I, or special (GxB_RANGE)
    size_t *I_size_handle,  // if > 0, I has been allocated by this
                            // method.  Otherwise, it is a shallow pointer into
                            // List->x or List->i, or is equal to GrB_ALL.
    GrB_Type *I_type_handle,    // the type of I: GrB_UINT32 or GrB_UINT64 for
                            // assign, subassign, extract, or for build when
                            // descriptor is GxB_USE_INDICES.  For build,
                            // this is List->type when the descriptor is
                            // GxB_USE_VALUES.
    GB_Werk Werk                            
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (I_handle != NULL) ;
    ASSERT (ni_handle != NULL) ;
    ASSERT (I_size_handle != NULL) ;
    ASSERT (I_type_handle != NULL) ;
    ASSERT_VECTOR_OK_OR_NULL (List, "List", GB0) ;

    (*I_handle) = NULL ;
    (*ni_handle) = 0 ;
    (*I_size_handle) = 0 ;
    (*I_type_handle) = NULL ;

    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;
    size_t I_size = 0, I2_size = 0 ;
    void *I = NULL, *I2 = NULL ;

    //--------------------------------------------------------------------------
    // quick return if List is NULL
    //--------------------------------------------------------------------------

    if (List == NULL)
    { 
        // GrB_build will not call this method with List == NULL
        ASSERT (!is_build) ;
        // List of NULL denotes GrB_ALL, or ":"; descriptor is ignored
        (*I_handle) = (uint64_t *) GrB_ALL ;
        (*I_type_handle) = GrB_UINT64 ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // finish any pending work in the List
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (List) ;

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    // GrB_DEFAULT (0)             use List->x
    // GxB_USE_VALUES (0)          use List->x (same as GrB_DEFAULT)
    // GxB_USE_INDICES (7060)      use List->i
    // GxB_IS_STRIDE (7061)        use List->x, size 3, for Icolon

    int list_descriptor = GrB_DEFAULT ;
    if (desc != NULL)
    { 
        switch (which)
        {
            default:
            case 0 : list_descriptor = desc->row_list ; break ;
            case 1 : list_descriptor = desc->col_list ; break ;
            case 2 : list_descriptor = desc->val_list ; break ;
        }
    }
    bool list_is_stride = (list_descriptor == GxB_IS_STRIDE) ;
    int64_t ni = GB_nnz ((GrB_Matrix) List) ;
    if (list_is_stride && (ni != 3 || is_build))
    { 
        // List must have exactly 3 items (lo,hi,stride) for GxB_IS_STRIDE
        // for assign, subassign, and extract.  GrB_build does not allow
        // GxB_IS_STRIDE.
        return (GrB_INVALID_VALUE) ;
    }

    bool use_values = (list_descriptor != GxB_USE_INDICES) ;

    //--------------------------------------------------------------------------
    // quick return if List is empty
    //--------------------------------------------------------------------------

    if (ni == 0)
    { 
        // List is not NULL, but has no entries (nvals (List) == 0)
        (*I_handle) = GB_CALLOC_MEMORY (1, sizeof (uint64_t), I_size_handle) ;
        if ((*I_handle) == NULL)
        { 
            return (GrB_OUT_OF_MEMORY) ;
        }
        (*I_type_handle) = GrB_UINT64 ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // extract the list of integers from the List vector
    //--------------------------------------------------------------------------

    int List_sparsity = GB_sparsity ((GrB_Matrix) List) ;
    GrB_Type I_type = NULL ;
    bool iso = false ;

    if (List_sparsity == GxB_SPARSE)
    { 

        //----------------------------------------------------------------------
        // List is sparse
        //----------------------------------------------------------------------

        if (use_values)
        { 
            I = List->x ;
            I_type = List->type ;
            iso = List->iso ;
        }
        else
        { 
            I = List->i ;
            I_type = (List->i_is_32) ? GrB_UINT32 : GrB_UINT64 ;
        }

    }
    else if (List_sparsity == GxB_BITMAP)
    { 

        //----------------------------------------------------------------------
        // List is bitmap
        //----------------------------------------------------------------------

        uint64_t Cp [2] ;
        if (use_values)
        { 
            if (List->iso)
            { 
                // get the iso value; it is expanded below
                I = List->x ;
                iso = true ;
            }
            else
            { 
                // extract the values from the bitmap vector
                I = GB_MALLOC_MEMORY (ni, List->type->size, &I_size) ;
                if (I == NULL)
                { 
                    // out of memory
                    return (GrB_OUT_OF_MEMORY) ;
                }
                GB_OK (GB_convert_b2s (Cp, NULL, NULL, /* Cx: */ I, NULL,
                    false, false, false, List->type, (GrB_Matrix) List, Werk)) ;
            }
            I_type = List->type ;
        }
        else
        { 
            // extract the indices from the bitmap vector
            I_type = (ni <= UINT32_MAX) ? GrB_UINT32 : GrB_UINT64 ;
            I = GB_MALLOC_MEMORY (ni, I_type->size, &I_size) ;
            if (I == NULL)
            { 
                // out of memory
                return (GrB_OUT_OF_MEMORY) ;
            }
            GB_OK (GB_convert_b2s (Cp, /* Ci: */ I, NULL, NULL, NULL,
                false, I_type == GrB_UINT32, false, List->type,
                (GrB_Matrix) List, Werk)) ;
        }

    }
    else // List_sparsity == GxB_FULL
    { 

        //----------------------------------------------------------------------
        // List is full
        //----------------------------------------------------------------------

        if (use_values)
        { 
            // if the List is iso, it is expanded below
            I = List->x ;
            I_type = List->type ;
            iso = List->iso ;
        }
        else
        { 
            // create I = 0:1:(length(List)-1) with quick return
            int64_t n = List->vlen ;
            if (is_build)
            { 
                // build an explicit list for GrB_build
                I_type = (n <= UINT32_MAX) ? GrB_UINT32 : GrB_UINT64 ;
                (*I_handle) = GB_MALLOC_MEMORY (n, I_type->size, I_size_handle);
                if ((*I_handle) == NULL)
                { 
                    // out of memory
                    return (GrB_OUT_OF_MEMORY) ;
                }
                int nthreads_max = GB_Context_nthreads_max ( ) ;
                double chunk = GB_Context_chunk ( ) ;
                int nthreads = GB_nthreads (n, chunk, nthreads_max) ;
                int64_t k ;
                if (I_type == GrB_UINT32)
                { 
                    uint32_t *I = (uint32_t *) (*I_handle) ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(static)
                    for (k = 0 ; k < n ; k++)
                    {
                        I [k] = k ;
                    }
                }
                else
                { 
                    uint64_t *I = (uint64_t *) (*I_handle) ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(static)
                    for (k = 0 ; k < n ; k++)
                    {
                        I [k] = k ;
                    }
                }
                (*ni_handle) = n ;
                (*I_type_handle) = I_type ;
                return (GrB_SUCCESS) ;
            }
            else
            { 
                // use I = [0, n-1, 1] and GxB_STRIDE
                return (GB_stride (0, 1, n-1,
                    I_handle, ni_handle, I_size_handle, I_type_handle)) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // expand I if it is iso-valued
    //--------------------------------------------------------------------------

    if (iso)
    { 
        // I has not been allocted; it is a shallow copy of List->x
        ASSERT (I == List->x) ;
        ASSERT (I_size == 0) ;
        I2 = GB_MALLOC_MEMORY (ni, I_type->size, &I2_size) ;
        if (I2 == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        GB_OK (GB_iso_expand (I2, ni, I, I_type)) ;
        // replace I with the newly-allocated and expanded I2
        I = I2 ;
        I_size = I2_size ;
        I2 = NULL ;
        I2_size = 0 ;
        // the list I is no longer iso
        iso = false ;
    }

    //--------------------------------------------------------------------------
    // determine the final output type for I
    //--------------------------------------------------------------------------

    GrB_Type I_target_type = NULL ;
    if (is_build && which == 2)
    { 
        // List remains as-is for the values for build
        I_target_type = I_type ;
    }
    else if (list_is_stride)
    { 
        // ensure the List is typecast to int64_t
        I_target_type = GrB_INT64 ;
    }
    else if (I_type == GrB_INT32 || I_type == GrB_UINT32)
    { 
        // implicit typecast of int32_t to uint32_t (I does not change)
        I_type = GrB_UINT32 ;
        I_target_type = GrB_UINT32 ;
    }
    else if (I_type == GrB_INT64 || I_type == GrB_UINT64)
    { 
        // implicit typecast of int64_t to uint64_t (I does not change)
        I_type = GrB_UINT64 ;
        I_target_type = GrB_UINT64 ;
    }
    else
    { 
        // I_type is not a 32/64 bit integer; typecast it to GrB_UINT64
        I_target_type = GrB_UINT64 ;
    }

    //--------------------------------------------------------------------------
    // copy/typecast the indices if needed
    //--------------------------------------------------------------------------

    if ((need_copy && I_size == 0) || I_type != I_target_type)
    { 
        // Create an ni-by-1 matrix T containing the values of I
        GB_CLEAR_MATRIX_HEADER (T, &T_header) ;
        GB_OK (GB_new (&T, // static header
            I_type, ni, 1, GB_ph_null, true, GxB_FULL, 0, 0,
            false, false, false)) ;
        GB_vector_load ((GrB_Vector) T, &I, I_type, ni, ni * (I_type->size),
            true) ;
        ASSERT_MATRIX_OK (T, "T for typecast to I", GB0) ;

        // I2 = (uint64_t) T->x or (int64_t) T->x
        I2 = GB_MALLOC_MEMORY (ni, sizeof (uint64_t), &I2_size) ;
        if (I2 == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        GB_OK (GB_cast_array (I2, I_target_type->code, T, nthreads_max)) ;
        GB_Matrix_free (&T) ;

        // free the old I and replace it with I2
        if (I_size > 0)
        { 
            GB_FREE_MEMORY (&I, I_size) ;
        }
        I = I2 ;
        I_size = I2_size ;
        I2 = NULL ;
        I2_size = 0 ;
        I_type = I_target_type ;
    }

    ASSERT (I_type == I_target_type) ;
    ASSERT (GB_IMPLIES (need_copy, I_size > 0)) ;

    //--------------------------------------------------------------------------
    // create the stride or return the list I
    //--------------------------------------------------------------------------

    if (list_is_stride)
    { 
        // I currently has type int64_t, so it can handle negative strides,
        // but it must be converted to uint64_t to become Icolon.
        ASSERT (I_type == GrB_INT64) ;
        ASSERT (!is_build) ;
        int64_t *I64 = (int64_t *) I ;
        int64_t stride_begin = I64 [GxB_BEGIN] ;
        int64_t stride_inc   = I64 [GxB_INC  ] ;
        int64_t stride_end   = I64 [GxB_END  ] ;
        // create the stride
        GB_OK (GB_stride (stride_begin, stride_inc, stride_end,
            I_handle, ni_handle, I_size_handle, I_type_handle)) ;
    }
    else
    { 
        // return I as-is
        ASSERT (I_type == GrB_UINT32 || I_type == GrB_UINT64 || 
            (is_build && I_type == List->type)) ;
        (*I_handle) = I ;
        (*ni_handle) = ni ;
        (*I_size_handle) = I_size ;
        (*I_type_handle) = I_type ;
        I = NULL ;
    }

    //--------------------------------------------------------------------------
    // free workspace return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

