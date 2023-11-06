//------------------------------------------------------------------------------
// GrB_Semiring_new: create a new semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A GraphBLAS Semiring consists of two components: "add" and "multiply".
// These components imply three domains: ztype, xtype, and ytype.

// The "add" is an associative and commutative monoid, which is a binary
// operator that works on a single type, ztype = add(ztype,ztype).  The add
// monoid also includes an identity value, called "zero", so that
// add(x,zero)=add(zero,x)=x.  For most algebras, this "zero" is a plain zero
// in the usual sense, but this is not the case for all algebras.  For example,
// for the max-plus algebra, the "add" operator is the function max(a,b), and
// the "zero" for this operator is -infinity since max(a,-inf)=max(-inf,a)=a.

// The "multiply" is a binary operator z = multiply(x,y).  It has no
// restrictions, except that the type of z must exactly match the ztype
// of the add monoid.  That is, the types for the multiply operator are
// ztype = multiply (xtype, ytype).  When the semiring is applied to two
// matrices A and B, where (A,B) appear in that order in the method, the
// multiply operator is always applied as z = multiply (A(i,j),B(i,j)).  The
// two input operands always appear in that order.  That is, the multiply
// operator is not assumed to be commutative.

#include "GB.h"
#include "GB_Semiring_new.h"

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE (semiring, header_size) ;   \
}

GrB_Info GrB_Semiring_new           // create a semiring
(
    GrB_Semiring *semiring,         // handle of semiring to create
    GrB_Monoid add,                 // additive monoid of the semiring
    GrB_BinaryOp multiply           // multiply operator of the semiring
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    GB_WHERE1 ("GrB_Semiring_new (&semiring, add, multiply)") ;

    GB_RETURN_IF_NULL (semiring) ;

    (*semiring) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (add) ;
    GB_RETURN_IF_NULL_OR_FAULTY (multiply) ;
    ASSERT_MONOID_OK (add, "semiring->add", GB0) ;
    ASSERT_BINARYOP_OK (multiply, "semiring->multiply", GB0) ;

    //--------------------------------------------------------------------------
    // allocate the semiring
    //--------------------------------------------------------------------------

    size_t header_size ;
    (*semiring) = GB_MALLOC (1, struct GB_Semiring_opaque, &header_size) ;
    if (*semiring == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    (*semiring)->header_size = header_size ;

    //--------------------------------------------------------------------------
    // create the semiring
    //--------------------------------------------------------------------------

    GB_OK (GB_Semiring_new (*semiring, add, multiply)) ;
    return (GrB_SUCCESS) ;
}

