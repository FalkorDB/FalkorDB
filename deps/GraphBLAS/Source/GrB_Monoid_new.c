//------------------------------------------------------------------------------
// GrB_Monoid_new:  create a new monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a new monoid with binary operator, z=op(x.y).  The three types of x,
// y, and z must all be the same, and the identity value must also have the
// same type.  No typecasting is done for the identity value.

#include "GB.h"
#include "GB_Monoid_new.h"

#define GB_MONOID_NEW(prefix,type,T)                                        \
GrB_Info GB_EVAL3 (prefix, _Monoid_new_, T) /* create a new monoid */       \
(                                                                           \
    GrB_Monoid *monoid,             /* handle of monoid to create    */     \
    GrB_BinaryOp op,                /* binary operator of the monoid */     \
    type identity                   /* identity value of the monoid  */     \
)                                                                           \
{                                                                           \
    GB_WHERE1 ("GrB_Monoid_new_" GB_STR(T) " (&monoid, op, identity)") ;    \
    type id = identity ;                                                    \
    return (GB_Monoid_new (monoid, op, &id, NULL, GB_ ## T ## _code,        \
        Werk)) ;                                                         \
}

GB_MONOID_NEW (GrB, bool      , BOOL   )
GB_MONOID_NEW (GrB, int8_t    , INT8   )
GB_MONOID_NEW (GrB, uint8_t   , UINT8  )
GB_MONOID_NEW (GrB, int16_t   , INT16  )
GB_MONOID_NEW (GrB, uint16_t  , UINT16 )
GB_MONOID_NEW (GrB, int32_t   , INT32  )
GB_MONOID_NEW (GrB, uint32_t  , UINT32 )
GB_MONOID_NEW (GrB, int64_t   , INT64  )
GB_MONOID_NEW (GrB, uint64_t  , UINT64 )
GB_MONOID_NEW (GrB, float     , FP32   )
GB_MONOID_NEW (GrB, double    , FP64   )
GB_MONOID_NEW (GxB, GxB_FC32_t, FC32   )
GB_MONOID_NEW (GxB, GxB_FC64_t, FC64   )

GrB_Info GrB_Monoid_new_UDT         // create a monoid with a user-defined type
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    void *identity                  // identity value of monoid
)
{ 
    GB_WHERE1 ("GrB_Monoid_new_UDT (&monoid, op, identity)") ;
    return (GB_Monoid_new (monoid, op, identity, NULL, GB_UDT_code, Werk)) ;
}

