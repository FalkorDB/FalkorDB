//------------------------------------------------------------------------------
// GB_cast: definitions for GB_cast_* methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CAST_H
#define GB_CAST_H

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
    const GB_Type_code code1,   // type code for Cx
    GrB_Matrix A,
    const int nthreads          // number of threads to use
) ;

GrB_Info GB_cast_matrix         // copy or typecast the values from A into C
(
    GrB_Matrix C,
    GrB_Matrix A
) ;

#endif

