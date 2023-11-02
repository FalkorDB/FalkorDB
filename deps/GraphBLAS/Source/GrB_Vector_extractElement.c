//------------------------------------------------------------------------------
// GrB_Vector_extractElement: extract a single entry from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extract a single entry, x = v(i), typecasting from the type of v to the type
// of x, as needed.

// Returns GrB_SUCCESS if v(i) is present, and sets x to its value.
// If v(i) is not present: if x is a bare scalar, x is unmodified and
// GrB_NO_VALUE is returned; if x is a GrB_scalar, x is returned as empty,
// and GrB_SUCCESS is returned.

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GrB_Vector_extractElement_Scalar   // S = V(i,j)
(
    GrB_Scalar S,                       // extracted scalar
    const GrB_Vector V,                 // vector to extract a scalar from
    GrB_Index i                         // index
)
{

    //--------------------------------------------------------------------------
    // check inputs (just the GrB_Scalar S)
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_WHERE (S, "GrB_Vector_extractElement_Scalar (s, V, i)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (S) ;

    //--------------------------------------------------------------------------
    // ensure S is bitmap
    //--------------------------------------------------------------------------

    if (!GB_IS_BITMAP (S))
    { 
        // convert S to bitmap
        GB_OK (GB_convert_any_to_bitmap ((GrB_Matrix) S, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // extract the entry (also checks the inputs V and i)
    //--------------------------------------------------------------------------

    void *x = S->x ;

    switch (S->type->code)
    {
        case GB_BOOL_code    : 
            info = GrB_Vector_extractElement_BOOL ((bool *) x, V, i) ;
            break ;

        case GB_INT8_code    : 
            info = GrB_Vector_extractElement_INT8 ((int8_t *) x, V, i) ;
            break ;

        case GB_INT16_code   : 
            info = GrB_Vector_extractElement_INT16 ((int16_t *) x, V, i) ;
            break ;

        case GB_INT32_code   : 
            info = GrB_Vector_extractElement_INT32 ((int32_t *) x, V, i) ;
            break ;

        case GB_INT64_code   : 
            info = GrB_Vector_extractElement_INT64 ((int64_t *) x, V, i) ;
            break ;

        case GB_UINT8_code   : 
            info = GrB_Vector_extractElement_UINT8 ((uint8_t *) x, V, i) ;
            break ;

        case GB_UINT16_code  : 
            info = GrB_Vector_extractElement_UINT16 ((uint16_t *) x, V, i) ;
            break ;

        case GB_UINT32_code  : 
            info = GrB_Vector_extractElement_UINT32 ((uint32_t *) x, V, i) ;
            break ;

        case GB_UINT64_code  : 
            info = GrB_Vector_extractElement_UINT64 ((uint64_t *) x, V, i) ;
            break ;

        case GB_FP32_code    : 
            info = GrB_Vector_extractElement_FP32 ((float *) x, V, i) ;
            break ;

        case GB_FP64_code    : 
            info = GrB_Vector_extractElement_FP64 ((double *) x, V, i) ;
            break ;

        case GB_FC32_code    : 
            info = GxB_Vector_extractElement_FC32 ((GxB_FC32_t *) x, V, i) ;
            break ;

        case GB_FC64_code    : 
            info = GxB_Vector_extractElement_FC64 ((GxB_FC64_t *) x, V, i) ;
            break ;

        case GB_UDT_code     : 
            info = GrB_Vector_extractElement_UDT ((void *) x, V, i) ;
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

#define GB_WHERE_STRING "GrB_Vector_extractElement (&x, v, i)"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_BOOL
#define GB_XTYPE bool
#define GB_XCODE GB_BOOL_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_INT8
#define GB_XTYPE int8_t
#define GB_XCODE GB_INT8_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_INT16
#define GB_XTYPE int16_t
#define GB_XCODE GB_INT16_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_INT32
#define GB_XTYPE int32_t
#define GB_XCODE GB_INT32_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_INT64
#define GB_XTYPE int64_t
#define GB_XCODE GB_INT64_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_UINT8
#define GB_XTYPE uint8_t
#define GB_XCODE GB_UINT8_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_UINT16
#define GB_XTYPE uint16_t
#define GB_XCODE GB_UINT16_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_UINT32
#define GB_XTYPE uint32_t
#define GB_XCODE GB_UINT32_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_UINT64
#define GB_XTYPE uint64_t
#define GB_XCODE GB_UINT64_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_FP32
#define GB_XTYPE float
#define GB_XCODE GB_FP32_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_FP64
#define GB_XTYPE double
#define GB_XCODE GB_FP64_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GxB_Vector_extractElement_FC32
#define GB_XTYPE GxB_FC32_t
#define GB_XCODE GB_FC32_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GxB_Vector_extractElement_FC64
#define GB_XTYPE GxB_FC64_t
#define GB_XCODE GB_FC64_code
#include "GB_Vector_extractElement.c"

#define GB_UDT_EXTRACT
#define GB_EXTRACT_ELEMENT GrB_Vector_extractElement_UDT
#define GB_XTYPE void
#define GB_XCODE GB_UDT_code
#include "GB_Vector_extractElement.c"

#define GB_EXTRACT_ELEMENT GxB_Vector_isStoredElement
#include "GB_Vector_extractElement.c"

