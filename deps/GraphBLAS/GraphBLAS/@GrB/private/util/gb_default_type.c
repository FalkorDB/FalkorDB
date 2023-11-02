//------------------------------------------------------------------------------
// gb_default_type: determine the default type for a binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// When a binary operator has inputs A and B, with no type specified, the type
// is determined from the types of A and B.

// If A or B are boolean, the type is taken from the other operand.
// If either A or B are signed, then the type is signed.

#include "gb_interface.h"

GrB_Type gb_default_type        // return the default type to use
(
    const GrB_Type atype,       // type of the A matrix
    const GrB_Type btype        // type of the B matrix
)
{

    if (atype == NULL || btype == NULL)
    {

        // undefined type
        return (NULL) ;

    }
    else if (atype == GrB_BOOL)
    { 

        // A is bool: optype determined by B
        return (btype) ;

    }
    else if (btype == GrB_BOOL)
    { 

        // B is bool: optype determined by A
        return (atype) ;

    }
    else if (atype == GrB_INT8)
    { 

        // A is int8: optype must be signed, and at least 8 bits
        if (btype == GrB_UINT8  ) return (GrB_INT8) ;
        if (btype == GrB_UINT16 ) return (GrB_INT16) ;
        if (btype == GrB_UINT32 ) return (GrB_INT32) ;
        if (btype == GrB_UINT64 ) return (GrB_INT64) ;
        return (btype) ;

    }
    else if (atype == GrB_INT16)
    { 

        // A is int16: optype must be signed, and at least 16 bits
        if (btype == GrB_INT8   ||
            btype == GrB_UINT8  ||
            btype == GrB_UINT16 ) return (GrB_INT16) ;
        if (btype == GrB_UINT32 ) return (GrB_INT32) ;
        if (btype == GrB_UINT64 ) return (GrB_INT64) ;
        return (btype) ;

    }
    else if (atype == GrB_INT32)
    { 

        // A is int32: optype must be signed, and at least 32 bits
        if (btype == GrB_INT8   ||
            btype == GrB_INT16  ||
            btype == GrB_UINT8  ||
            btype == GrB_UINT16 ||
            btype == GrB_UINT32 ) return (GrB_INT32) ;
        if (btype == GrB_UINT64 ) return (GrB_INT64) ;
        return (btype) ;

    }
    else if (atype == GrB_INT64)
    { 

        // A is int64: optype must be signed, and at least 64
        // bits (if integer).  float and float complex are OK
        if (btype == GrB_INT8   ||
            btype == GrB_INT16  ||
            btype == GrB_INT32  ||
            btype == GrB_UINT8  ||
            btype == GrB_UINT16 ||
            btype == GrB_UINT32 ||
            btype == GrB_UINT64 ) return (GrB_INT64) ;
        return (btype) ;

    }
    else if (atype == GrB_UINT8)
    { 

        // A is uint8: optype determined by B (which is not bool; see above)
        return (btype) ;

    }
    else if (atype == GrB_UINT16)
    { 

        // A is uint16: optype can be unsigned if B is also unsigned.
        // optype must be at least 16 bits.
        if (btype == GrB_UINT8  ) return (GrB_UINT16) ;
        if (btype == GrB_INT8   ) return (GrB_INT16) ;
        return (btype) ;

    }
    else if (atype == GrB_UINT32)
    { 

        // A is uint32: optype can be unsigned if B is also unsigned.
        // optype must be at least 32 bits.
        if (btype == GrB_UINT8  ||
            btype == GrB_UINT16 ) return (GrB_UINT32) ;
        if (btype == GrB_INT8   ||
            btype == GrB_INT16  ) return (GrB_INT32) ;
        return (btype) ;

    }
    else if (atype == GrB_UINT64)
    { 

        // A is uint64: optype can be unsigned if B is also unsigned.
        // optype must be at least 64 bits.  float and float complex OK.
        if (btype == GrB_UINT8  ||
            btype == GrB_UINT16 ||
            btype == GrB_UINT32 ) return (GrB_UINT64) ;
        if (btype == GrB_INT8   ||
            btype == GrB_INT16  ||
            btype == GrB_INT32  ) return (GrB_INT64) ;
        return (btype) ;

    }
    else if (atype == GrB_FP32)
    { 

        // A is float: optype must be real or complex
        if (btype == GrB_INT8   ||
            btype == GrB_INT16  ||
            btype == GrB_INT32  ||
            btype == GrB_INT64  ||
            btype == GrB_UINT8  ||
            btype == GrB_UINT16 ||
            btype == GrB_UINT32 ||
            btype == GrB_UINT64 ) return (GrB_FP32) ;
        return (btype) ;

    }
    else if (atype == GrB_FP64)
    { 

        // A is double: optype must be double or double complex
        if (btype == GxB_FC32   ||
            btype == GxB_FC64   ) return (GxB_FC64) ;
        return (GrB_FP64) ;

    }
    else if (atype == GxB_FC32)
    { 

        // A is float complex: optype must be float complex
        // or double complex
        if (btype == GrB_FP64   ||
            btype == GxB_FC64   ) return (GxB_FC64) ;
        return (GxB_FC32) ;

    }
    else if (atype == GxB_FC64)
    { 

        // A is double complex: optype must be double complex
        return (GxB_FC64) ;

    }
    else
    {

        // unknown type
        return (NULL) ;

    }
}

