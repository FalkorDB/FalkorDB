//------------------------------------------------------------------------------
// GB_cast: definitions for GB_cast_* methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CAST_H
#define GB_CAST_H

//------------------------------------------------------------------------------
// pointer casting function, returned by GB_cast_factory.
//------------------------------------------------------------------------------

typedef void (*GB_cast_function) (void *, const void *, size_t) ;

GB_cast_function GB_cast_factory   // returns pointer to function to cast x to z
(
    const GB_Type_code code1,      // the type of z, the output value
    const GB_Type_code code2       // the type of x, the input value
) ;

//------------------------------------------------------------------------------
// GB_cast_scalar: typecast or copy a scalar
//------------------------------------------------------------------------------

static inline void GB_cast_scalar  // z = x with typecasting from xcode to zcode
(
    void *z,                    // output scalar z of type zcode
    GB_Type_code zcode,         // type of z
    const void *x,              // input scalar x of type xcode
    GB_Type_code xcode,         // type of x
    size_t size                 // size of x and z if they have the same type
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (z != NULL) ;
    ASSERT (x != NULL) ;

    //--------------------------------------------------------------------------
    // copy or typecast the scalar
    //--------------------------------------------------------------------------

    if (zcode == xcode)
    { 
        // no typecasting; copy x into z, works for any types
        memcpy (z, x, size) ;
    }
    else
    { 
        // typecast x into z, works for built-in types only
        GB_cast_function cast_X_to_Z = GB_cast_factory (zcode, xcode) ;
        cast_X_to_Z (z, x, size) ;
    }
}

//------------------------------------------------------------------------------
// GB_cast_one: return 1, typecasted to any type
//------------------------------------------------------------------------------

static inline void GB_cast_one  // z = 1 with typecasting zcode
(
    void *z,                    // output scalar z of type zcode
    GB_Type_code zcode          // type of z
)
{ 
    GB_cast_function cast_one = GB_cast_factory (zcode, GB_UINT8_code) ;
    uint8_t one = 1 ;
    cast_one (z, (GB_void *) &one, sizeof (uint8_t)) ;
}

//------------------------------------------------------------------------------

GrB_Info GB_cast_array          // typecast an array
(
    GB_void *Cx,                // output array
    const GB_Type_code ccode,   // type code for Cx
    GrB_Matrix A,
    const int A_nthreads        // number of threads to use
) ;

GrB_Info GB_cast_matrix         // copy or typecast the values from A into C
(
    GrB_Matrix C,
    GrB_Matrix A
) ;

void GB_cast_int                // parallel memcpy/cast of integer arrays
(
    void *dest,                 // destination
    GB_Type_code dest_code,     // destination type: int32/64, or uint32/64
    const void *src,            // source
    GB_Type_code src_code,      // source type: int32/64, or uint32/64
    size_t n,                   // # of entries to copy
    int nthreads_max            // max # of threads to use
) ;

#endif

