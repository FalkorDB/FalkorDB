//------------------------------------------------------------------------------
// GxB_Monoid_terminal_new:  create a new monoid with a terminal value
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Identical to GrB_Monoid_new, except that a terminal value is specified.  No
// typecasting is done for the terminal value.  Its type must match the
// identity value.

#include "GB.h"
#include "GB_Monoid_new.h"

#define GB_MONOID_TERMINAL_NEW(type,T)                                      \
GrB_Info GB_EVAL2 (GXB (Monoid_terminal_new_), T) /* create a new monoid */ \
(                                                                           \
    GrB_Monoid *monoid,             /* handle of monoid to create    */     \
    GrB_BinaryOp op,                /* binary operator of the monoid */     \
    type identity,                  /* identity value of the monoid  */     \
    type terminal                   /* terminal value of the monoid  */     \
)                                                                           \
{                                                                           \
    GB_WHERE1 ("GxB_Monoid_terminal_new" GB_STR(T)                          \
        " (&monoid, op, identity, terminal)") ;                             \
    type id = identity ;                                                    \
    type tr = terminal ;                                                    \
    return (GB_Monoid_new (monoid, op, &id, &tr, GB_ ## T ## _code, Werk)) ;\
}

GB_MONOID_TERMINAL_NEW (bool      , BOOL   )
GB_MONOID_TERMINAL_NEW (int8_t    , INT8   )
GB_MONOID_TERMINAL_NEW (uint8_t   , UINT8  )
GB_MONOID_TERMINAL_NEW (int16_t   , INT16  )
GB_MONOID_TERMINAL_NEW (uint16_t  , UINT16 )
GB_MONOID_TERMINAL_NEW (int32_t   , INT32  )
GB_MONOID_TERMINAL_NEW (uint32_t  , UINT32 )
GB_MONOID_TERMINAL_NEW (int64_t   , INT64  )
GB_MONOID_TERMINAL_NEW (uint64_t  , UINT64 )
GB_MONOID_TERMINAL_NEW (float     , FP32   )
GB_MONOID_TERMINAL_NEW (double    , FP64   )
GB_MONOID_TERMINAL_NEW (GxB_FC32_t, FC32   )
GB_MONOID_TERMINAL_NEW (GxB_FC64_t, FC64   )

GrB_Info GxB_Monoid_terminal_new_UDT        // create a monoid with a user type
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    void *identity,                 // identity value of the monoid
    void *terminal                  // terminal value of the monoid
)
{ 
    GB_WHERE1 ("GxB_Monoid_terminal_new_UDT "
        "(&monoid, op, identity, terminal)") ;
    GB_RETURN_IF_NULL (terminal) ;
    return (GB_Monoid_new (monoid, op, identity, terminal, GB_UDT_code,
        Werk)) ;
}

