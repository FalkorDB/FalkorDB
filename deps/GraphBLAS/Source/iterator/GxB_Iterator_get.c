//------------------------------------------------------------------------------
// GxB_Iterator_get_TYPE: get value of the current entry for any iterator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

// On input, the prior call to GxB_*Iterator_*seek*, or GxB_*Iterator_*next*
// must have returned GrB_SUCCESS, indicating that the iterator is at a valid
// current entry for either a matrix or vector.

// Returns the value of the current entry at the position determined by the
// iterator.  No typecasting is permitted; the method name must match the
// type of the matrix or vector.

#undef GxB_Iterator_get_BOOL
#undef GxB_Iterator_get_INT8
#undef GxB_Iterator_get_INT16
#undef GxB_Iterator_get_INT32
#undef GxB_Iterator_get_INT64
#undef GxB_Iterator_get_UINT8
#undef GxB_Iterator_get_UINT16
#undef GxB_Iterator_get_UINT32
#undef GxB_Iterator_get_UINT64
#undef GxB_Iterator_get_FP32
#undef GxB_Iterator_get_FP64
#undef GxB_Iterator_get_FC32
#undef GxB_Iterator_get_FC64
#undef GxB_Iterator_get_UDT

#if !defined ( GBMATLAB )

// These methods are not tested by the GraphBLAS/Test or GraphBLAS/Tcov
// test coverage suite, because the conflict with libmwgraphblas.so inside
// MATLAB.

bool GxB_Iterator_get_BOOL (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, bool)) ;
}

int8_t GxB_Iterator_get_INT8 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, int8_t)) ;
}

int16_t GxB_Iterator_get_INT16 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, int16_t)) ;
}

int32_t GxB_Iterator_get_INT32 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, int32_t)) ;
}

int64_t GxB_Iterator_get_INT64 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, int64_t)) ;
}

uint8_t GxB_Iterator_get_UINT8 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, uint8_t)) ;
}

uint16_t GxB_Iterator_get_UINT16 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, uint16_t)) ;
}

uint32_t GxB_Iterator_get_UINT32 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, uint32_t)) ;
}

uint64_t GxB_Iterator_get_UINT64 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, uint64_t)) ;
}

float GxB_Iterator_get_FP32 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, float)) ;
}

double GxB_Iterator_get_FP64 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, double)) ;
}

GxB_FC32_t GxB_Iterator_get_FC32 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, GxB_FC32_t)) ;
}

GxB_FC64_t GxB_Iterator_get_FC64 (GxB_Iterator iterator)
{
    return (GB_Iterator_get (iterator, GxB_FC64_t)) ;
}

void GxB_Iterator_get_UDT (GxB_Iterator iterator, void *value)
{
    memcpy (value, ((const uint8_t *) (iterator->Ax)) +
        (iterator->iso ? 0 : (iterator->type_size * iterator->p)),
        iterator->type_size) ;
}

#endif

