//------------------------------------------------------------------------------
// GB_casting.c: unary typecasting functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// typecasting from double to integer
//------------------------------------------------------------------------------

extern int8_t   GB_cast_to_int8_t   (double x) ;
extern int16_t  GB_cast_to_int16_t  (double x) ;
extern int32_t  GB_cast_to_int32_t  (double x) ;
extern int64_t  GB_cast_to_int64_t  (double x) ;
extern uint8_t  GB_cast_to_uint8_t  (double x) ;
extern uint16_t GB_cast_to_uint16_t (double x) ;
extern uint32_t GB_cast_to_uint32_t (double x) ;
extern uint64_t GB_cast_to_uint64_t (double x) ;

//------------------------------------------------------------------------------
// unary typecast operators, used in GB_cast_factory.c.
//------------------------------------------------------------------------------

#define GB_CAST_FUNCTION(ztype,xtype)                                   \
extern void GB (_cast_ ## ztype ## _ ## xtype)                          \
(                                                                       \
    void *z,            /* typecasted output, of type ztype */          \
    const void *x,      /* input value to typecast, of type xtype */    \
    size_t s            /* size of type, for GB_copy_user_user only */  \
) ;

//------------------------------------------------------------------------------
// typecast to boolean
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (bool      , bool      )
GB_CAST_FUNCTION (bool      , int8_t    )
GB_CAST_FUNCTION (bool      , int16_t   )
GB_CAST_FUNCTION (bool      , int32_t   )
GB_CAST_FUNCTION (bool      , int64_t   )
GB_CAST_FUNCTION (bool      , uint8_t   )
GB_CAST_FUNCTION (bool      , uint16_t  )
GB_CAST_FUNCTION (bool      , uint32_t  )
GB_CAST_FUNCTION (bool      , uint64_t  )
GB_CAST_FUNCTION (bool      , float     )
GB_CAST_FUNCTION (bool      , double    )
GB_CAST_FUNCTION (bool      , GxB_FC32_t)
GB_CAST_FUNCTION (bool      , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to int8_t
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (int8_t    , bool      )
GB_CAST_FUNCTION (int8_t    , int8_t    )
GB_CAST_FUNCTION (int8_t    , int16_t   )
GB_CAST_FUNCTION (int8_t    , int32_t   )
GB_CAST_FUNCTION (int8_t    , int64_t   )
GB_CAST_FUNCTION (int8_t    , uint8_t   )
GB_CAST_FUNCTION (int8_t    , uint16_t  )
GB_CAST_FUNCTION (int8_t    , uint32_t  )
GB_CAST_FUNCTION (int8_t    , uint64_t  )
GB_CAST_FUNCTION (int8_t    , float     )
GB_CAST_FUNCTION (int8_t    , double    )
GB_CAST_FUNCTION (int8_t    , GxB_FC32_t)
GB_CAST_FUNCTION (int8_t    , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to int16_t
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (int16_t   , bool      )
GB_CAST_FUNCTION (int16_t   , int8_t    )
GB_CAST_FUNCTION (int16_t   , int16_t   )
GB_CAST_FUNCTION (int16_t   , int32_t   )
GB_CAST_FUNCTION (int16_t   , int64_t   )
GB_CAST_FUNCTION (int16_t   , uint8_t   )
GB_CAST_FUNCTION (int16_t   , uint16_t  )
GB_CAST_FUNCTION (int16_t   , uint32_t  )
GB_CAST_FUNCTION (int16_t   , uint64_t  )
GB_CAST_FUNCTION (int16_t   , float     )
GB_CAST_FUNCTION (int16_t   , double    )
GB_CAST_FUNCTION (int16_t   , GxB_FC32_t)
GB_CAST_FUNCTION (int16_t   , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to int32_t
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (int32_t   , bool      )
GB_CAST_FUNCTION (int32_t   , int8_t    )
GB_CAST_FUNCTION (int32_t   , int16_t   )
GB_CAST_FUNCTION (int32_t   , int32_t   )
GB_CAST_FUNCTION (int32_t   , int64_t   )
GB_CAST_FUNCTION (int32_t   , uint8_t   )
GB_CAST_FUNCTION (int32_t   , uint16_t  )
GB_CAST_FUNCTION (int32_t   , uint32_t  )
GB_CAST_FUNCTION (int32_t   , uint64_t  )
GB_CAST_FUNCTION (int32_t   , float     )
GB_CAST_FUNCTION (int32_t   , double    )
GB_CAST_FUNCTION (int32_t   , GxB_FC32_t)
GB_CAST_FUNCTION (int32_t   , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to int64_t
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (int64_t   , bool      )
GB_CAST_FUNCTION (int64_t   , int8_t    )
GB_CAST_FUNCTION (int64_t   , int16_t   )
GB_CAST_FUNCTION (int64_t   , int32_t   )
GB_CAST_FUNCTION (int64_t   , int64_t   )
GB_CAST_FUNCTION (int64_t   , uint8_t   )
GB_CAST_FUNCTION (int64_t   , uint16_t  )
GB_CAST_FUNCTION (int64_t   , uint32_t  )
GB_CAST_FUNCTION (int64_t   , uint64_t  )
GB_CAST_FUNCTION (int64_t   , float     )
GB_CAST_FUNCTION (int64_t   , double    )
GB_CAST_FUNCTION (int64_t   , GxB_FC32_t)
GB_CAST_FUNCTION (int64_t   , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to uint8_t
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (uint8_t   , bool      )
GB_CAST_FUNCTION (uint8_t   , int8_t    )
GB_CAST_FUNCTION (uint8_t   , int16_t   )
GB_CAST_FUNCTION (uint8_t   , int32_t   )
GB_CAST_FUNCTION (uint8_t   , int64_t   )
GB_CAST_FUNCTION (uint8_t   , uint8_t   )
GB_CAST_FUNCTION (uint8_t   , uint16_t  )
GB_CAST_FUNCTION (uint8_t   , uint32_t  )
GB_CAST_FUNCTION (uint8_t   , uint64_t  )
GB_CAST_FUNCTION (uint8_t   , float     )
GB_CAST_FUNCTION (uint8_t   , double    )
GB_CAST_FUNCTION (uint8_t   , GxB_FC32_t)
GB_CAST_FUNCTION (uint8_t   , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to uint16_t
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (uint16_t  , bool      )
GB_CAST_FUNCTION (uint16_t  , int8_t    )
GB_CAST_FUNCTION (uint16_t  , int16_t   )
GB_CAST_FUNCTION (uint16_t  , int32_t   )
GB_CAST_FUNCTION (uint16_t  , int64_t   )
GB_CAST_FUNCTION (uint16_t  , uint8_t   )
GB_CAST_FUNCTION (uint16_t  , uint16_t  )
GB_CAST_FUNCTION (uint16_t  , uint32_t  )
GB_CAST_FUNCTION (uint16_t  , uint64_t  )
GB_CAST_FUNCTION (uint16_t  , float     )
GB_CAST_FUNCTION (uint16_t  , double    )
GB_CAST_FUNCTION (uint16_t  , GxB_FC32_t)
GB_CAST_FUNCTION (uint16_t  , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to uint32_t
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (uint32_t  , bool      )
GB_CAST_FUNCTION (uint32_t  , int8_t    )
GB_CAST_FUNCTION (uint32_t  , int16_t   )
GB_CAST_FUNCTION (uint32_t  , int32_t   )
GB_CAST_FUNCTION (uint32_t  , int64_t   )
GB_CAST_FUNCTION (uint32_t  , uint8_t   )
GB_CAST_FUNCTION (uint32_t  , uint16_t  )
GB_CAST_FUNCTION (uint32_t  , uint32_t  )
GB_CAST_FUNCTION (uint32_t  , uint64_t  )
GB_CAST_FUNCTION (uint32_t  , float     )
GB_CAST_FUNCTION (uint32_t  , double    )
GB_CAST_FUNCTION (uint32_t  , GxB_FC32_t)
GB_CAST_FUNCTION (uint32_t  , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to uint64_t
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (uint64_t  , bool      )
GB_CAST_FUNCTION (uint64_t  , int8_t    )
GB_CAST_FUNCTION (uint64_t  , int16_t   )
GB_CAST_FUNCTION (uint64_t  , int32_t   )
GB_CAST_FUNCTION (uint64_t  , int64_t   )
GB_CAST_FUNCTION (uint64_t  , uint8_t   )
GB_CAST_FUNCTION (uint64_t  , uint16_t  )
GB_CAST_FUNCTION (uint64_t  , uint32_t  )
GB_CAST_FUNCTION (uint64_t  , uint64_t  )
GB_CAST_FUNCTION (uint64_t  , float     )
GB_CAST_FUNCTION (uint64_t  , double    )
GB_CAST_FUNCTION (uint64_t  , GxB_FC32_t)
GB_CAST_FUNCTION (uint64_t  , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to float
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (float     , bool      )
GB_CAST_FUNCTION (float     , int8_t    )
GB_CAST_FUNCTION (float     , int16_t   )
GB_CAST_FUNCTION (float     , int32_t   )
GB_CAST_FUNCTION (float     , int64_t   )
GB_CAST_FUNCTION (float     , uint8_t   )
GB_CAST_FUNCTION (float     , uint16_t  )
GB_CAST_FUNCTION (float     , uint32_t  )
GB_CAST_FUNCTION (float     , uint64_t  )
GB_CAST_FUNCTION (float     , float     )
GB_CAST_FUNCTION (float     , double    )
GB_CAST_FUNCTION (float     , GxB_FC32_t)
GB_CAST_FUNCTION (float     , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to double
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (double    , bool      )
GB_CAST_FUNCTION (double    , int8_t    )
GB_CAST_FUNCTION (double    , int16_t   )
GB_CAST_FUNCTION (double    , int32_t   )
GB_CAST_FUNCTION (double    , int64_t   )
GB_CAST_FUNCTION (double    , uint8_t   )
GB_CAST_FUNCTION (double    , uint16_t  )
GB_CAST_FUNCTION (double    , uint32_t  )
GB_CAST_FUNCTION (double    , uint64_t  )
GB_CAST_FUNCTION (double    , float     )
GB_CAST_FUNCTION (double    , double    )
GB_CAST_FUNCTION (double    , GxB_FC32_t)
GB_CAST_FUNCTION (double    , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to float complex
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (GxB_FC32_t, bool      )
GB_CAST_FUNCTION (GxB_FC32_t, int8_t    )
GB_CAST_FUNCTION (GxB_FC32_t, int16_t   )
GB_CAST_FUNCTION (GxB_FC32_t, int32_t   )
GB_CAST_FUNCTION (GxB_FC32_t, int64_t   )
GB_CAST_FUNCTION (GxB_FC32_t, uint8_t   )
GB_CAST_FUNCTION (GxB_FC32_t, uint16_t  )
GB_CAST_FUNCTION (GxB_FC32_t, uint32_t  )
GB_CAST_FUNCTION (GxB_FC32_t, uint64_t  )
GB_CAST_FUNCTION (GxB_FC32_t, float     )
GB_CAST_FUNCTION (GxB_FC32_t, double    )
GB_CAST_FUNCTION (GxB_FC32_t, GxB_FC32_t)
GB_CAST_FUNCTION (GxB_FC32_t, GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to double complex
//------------------------------------------------------------------------------

GB_CAST_FUNCTION (GxB_FC64_t, bool      )
GB_CAST_FUNCTION (GxB_FC64_t, int8_t    )
GB_CAST_FUNCTION (GxB_FC64_t, int16_t   )
GB_CAST_FUNCTION (GxB_FC64_t, int32_t   )
GB_CAST_FUNCTION (GxB_FC64_t, int64_t   )
GB_CAST_FUNCTION (GxB_FC64_t, uint8_t   )
GB_CAST_FUNCTION (GxB_FC64_t, uint16_t  )
GB_CAST_FUNCTION (GxB_FC64_t, uint32_t  )
GB_CAST_FUNCTION (GxB_FC64_t, uint64_t  )
GB_CAST_FUNCTION (GxB_FC64_t, float     )
GB_CAST_FUNCTION (GxB_FC64_t, double    )
GB_CAST_FUNCTION (GxB_FC64_t, GxB_FC32_t)
GB_CAST_FUNCTION (GxB_FC64_t, GxB_FC64_t)

