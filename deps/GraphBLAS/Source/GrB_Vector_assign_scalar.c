//------------------------------------------------------------------------------
// GrB_Vector_assign_[SCALAR]: assign scalar to vector, via scalar expansion
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Assigns a single scalar to a vector, w<M>(Rows) = accum(w(Rows),x)
// The scalar x is implicitly expanded into a vector u of size nRows-by-1,
// with each entry in u equal to x.

#define GB_FREE_ALL ;
#include "GB_assign.h"
#include "GB_ij.h"
#include "GB_get_mask.h"

#define GB_ASSIGN_SCALAR(prefix,type,T,ampersand)                              \
GrB_Info GB_EVAL3 (prefix, _Vector_assign_, T) /* w<M>(Rows)=accum(w(Rows),x)*/\
(                                                                              \
    GrB_Vector w,                   /* input/output vector for results      */ \
    const GrB_Vector M,             /* optional mask for w                  */ \
    const GrB_BinaryOp accum,       /* optional accum for Z=accum(w(Rows),x)*/ \
    type x,                         /* scalar to assign to w(Rows)          */ \
    const GrB_Index *Rows,          /* row indices                          */ \
    GrB_Index nRows,                /* number of row indices                */ \
    const GrB_Descriptor desc       /* descriptor for w and mask            */ \
)                                                                              \
{                                                                              \
    GB_WHERE (w, "GrB_Vector_assign_" GB_STR(T)                                \
        " (w, M, accum, x, Rows, nRows, desc)") ;                              \
    GB_BURBLE_START ("GrB_assign") ;                                           \
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;                                          \
    GB_RETURN_IF_FAULTY (M) ;                                                  \
    ASSERT (GB_VECTOR_OK (w)) ;                                                \
    ASSERT (GB_IMPLIES (M != NULL, GB_VECTOR_OK (M))) ;                        \
    GrB_Info info = GB_assign_scalar ((GrB_Matrix) w, (GrB_Matrix) M, accum,   \
        ampersand x, GB_## T ## _code, Rows, nRows, GrB_ALL, 1, desc, Werk) ;  \
    GB_BURBLE_END ;                                                            \
    return (info) ;                                                            \
}

GB_ASSIGN_SCALAR (GrB, bool      , BOOL   , &)
GB_ASSIGN_SCALAR (GrB, int8_t    , INT8   , &)
GB_ASSIGN_SCALAR (GrB, uint8_t   , UINT8  , &)
GB_ASSIGN_SCALAR (GrB, int16_t   , INT16  , &)
GB_ASSIGN_SCALAR (GrB, uint16_t  , UINT16 , &)
GB_ASSIGN_SCALAR (GrB, int32_t   , INT32  , &)
GB_ASSIGN_SCALAR (GrB, uint32_t  , UINT32 , &)
GB_ASSIGN_SCALAR (GrB, int64_t   , INT64  , &)
GB_ASSIGN_SCALAR (GrB, uint64_t  , UINT64 , &)
GB_ASSIGN_SCALAR (GrB, float     , FP32   , &)
GB_ASSIGN_SCALAR (GrB, double    , FP64   , &)
GB_ASSIGN_SCALAR (GxB, GxB_FC32_t, FC32   , &)
GB_ASSIGN_SCALAR (GxB, GxB_FC64_t, FC64   , &)
GB_ASSIGN_SCALAR (GrB, void *    , UDT    ,  )

//------------------------------------------------------------------------------
// GrB_Vector_assign_Scalar: assign a GrB_Scalar to a matrix
//------------------------------------------------------------------------------

// If the GrB_Scalar s is non-empty, then this is the same as the non-opapue
// scalar subassignment above.

// If the GrB_Scalar s is empty of type stype, then this is identical to:
//  GrB_Vector_new (&S, stype, nRows) ;
//  GrB_Vector_assign (w, M, accum, S, Rows, nRows, desc) ;
//  GrB_Vector_free (&S) ;

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_Matrix_free (&S) ;
#include "GB_static_header.h"

GrB_Info GrB_Vector_assign_Scalar   // w<Mask>(I) = accum (w(I),s)
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector M_in,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    GrB_Scalar scalar,              // scalar to assign to w(I)
    const GrB_Index *I,             // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and Mask
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix S = NULL ;
    GB_WHERE (w, "GrB_Vector_assign_Scalar"
        " (w, M, accum, s, Rows, nRows, desc)") ;
    GB_BURBLE_START ("GrB_assign") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_NULL_OR_FAULTY (scalar) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL (I) ;
    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (M_in == NULL || GB_VECTOR_OK (M_in)) ;

    // if w has a user-defined type, its type must match the scalar type
    if (w->type->code == GB_UDT_code && w->type != scalar->type)
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH, "Input of type [%s]\n"
            "cannot be typecast to output of type [%s]",
            scalar->type->name, w->type->name) ;
    }

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // w<M>(Rows) = accum (w(Rows), scalar)
    //--------------------------------------------------------------------------

    GrB_Index nvals ;
    GB_OK (GB_nvals (&nvals, (GrB_Matrix) scalar, Werk)) ;

    if (M == NULL && !Mask_comp && ni == 1 && !C_replace)
    {

        //----------------------------------------------------------------------
        // scalar assignment
        //----------------------------------------------------------------------

        const GrB_Index row = I [0] ;
        if (nvals == 1)
        { 
            // set the element: w(row) += scalar or w(row) = scalar
            info = GB_setElement ((GrB_Matrix) w, accum, scalar->x, row, 0,
                scalar->type->code, Werk) ;
        }
        else if (accum == NULL)
        { 
            // delete the w(row) element
            info = GB_Vector_removeElement (w, row, Werk) ;
        }

    }
    else if (nvals == 1)
    { 

        //----------------------------------------------------------------------
        // the opaque GrB_Scalar has a single entry
        //----------------------------------------------------------------------

        // This is identical to non-opaque scalar subassignment

        info = GB_assign (
            (GrB_Matrix) w, C_replace,  // w vector and its descriptor
            M, Mask_comp, Mask_struct,  // mask vector and its descriptor
            false,                      // do not transpose the mask
            accum,                      // for accum (w(Rows),scalar)
            NULL, false,                // no explicit vector u
            I, ni,                      // row indices
            GrB_ALL, 1,                 // column indices
            true,                       // do scalar expansion
            scalar->x,                  // scalar to assign, expands to become u
            scalar->type->code,         // type code of scalar to expand
            GB_ASSIGN,
            Werk) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // the opaque GrB_Scalar has no entry
        //----------------------------------------------------------------------

        // determine the properites of the I index list
        int64_t nRows, RowColon [3] ;
        int RowsKind ;
        GB_ijlength (I, ni, GB_NROWS (w), &nRows, &RowsKind, RowColon);

        // create an empty matrix S of the right size, and use matrix assign
        struct GB_Matrix_opaque S_header ;
        GB_CLEAR_STATIC_HEADER (S, &S_header) ;
        GB_OK (GB_new (&S,  // existing header
            scalar->type, nRows, 1, GB_Ap_calloc, true, GxB_AUTO_SPARSITY,
            GB_HYPER_SWITCH_DEFAULT, 1)) ;
        info = GB_assign (
            (GrB_Matrix) w, C_replace,      // w vector and its descriptor
            M, Mask_comp, Mask_struct,      // mask matrix and its descriptor
            false,                          // do not transpose the mask
            accum,                          // for accum (w(Rows),scalar)
            S, false,                       // S matrix and its descriptor
            I, ni,                          // row indices
            GrB_ALL, 1,                     // column indices
            false, NULL, GB_ignore_code,    // no scalar expansion
            GB_ASSIGN,
            Werk) ;
        GB_FREE_ALL ;
    }

    GB_BURBLE_END ;
    return (info) ;
}

