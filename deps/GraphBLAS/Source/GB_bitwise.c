//------------------------------------------------------------------------------
// GB_bitwise.c: declaring functions from GB_bitwise.h
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

// z = bitget (x,k)
extern int8_t   GB_bitget_int8   (int8_t   x, int8_t   k) ;
extern int16_t  GB_bitget_int16  (int16_t  x, int16_t  k) ;
extern int32_t  GB_bitget_int32  (int32_t  x, int32_t  k) ;
extern int64_t  GB_bitget_int64  (int64_t  x, int64_t  k) ;
extern uint8_t  GB_bitget_uint8  (uint8_t  x, uint8_t  k) ;
extern uint16_t GB_bitget_uint16 (uint16_t x, uint16_t k) ;
extern uint32_t GB_bitget_uint32 (uint32_t x, uint32_t k) ;
extern uint64_t GB_bitget_uint64 (uint64_t x, uint64_t k) ;

// z = bitset (x,k)
extern int8_t   GB_bitset_int8   (int8_t   x, int8_t   k) ;
extern int16_t  GB_bitset_int16  (int16_t  x, int16_t  k) ;
extern int32_t  GB_bitset_int32  (int32_t  x, int32_t  k) ;
extern int64_t  GB_bitset_int64  (int64_t  x, int64_t  k) ;
extern uint8_t  GB_bitset_uint8  (uint8_t  x, uint8_t  k) ;
extern uint16_t GB_bitset_uint16 (uint16_t x, uint16_t k) ;
extern uint32_t GB_bitset_uint32 (uint32_t x, uint32_t k) ;
extern uint64_t GB_bitset_uint64 (uint64_t x, uint64_t k) ;

// z = bitclr (x,k)
extern int8_t   GB_bitclr_int8   (int8_t   x, int8_t   k) ;
extern int16_t  GB_bitclr_int16  (int16_t  x, int16_t  k) ;
extern int32_t  GB_bitclr_int32  (int32_t  x, int32_t  k) ;
extern int64_t  GB_bitclr_int64  (int64_t  x, int64_t  k) ;
extern uint8_t  GB_bitclr_uint8  (uint8_t  x, uint8_t  k) ;
extern uint16_t GB_bitclr_uint16 (uint16_t x, uint16_t k) ;
extern uint32_t GB_bitclr_uint32 (uint32_t x, uint32_t k) ;
extern uint64_t GB_bitclr_uint64 (uint64_t x, uint64_t k) ;

// z = bitshift (x,k)
extern int8_t   GB_bitshift_int8   (int8_t   x, int8_t k) ;
extern int16_t  GB_bitshift_int16  (int16_t  x, int8_t k) ;
extern int32_t  GB_bitshift_int32  (int32_t  x, int8_t k) ;
extern int64_t  GB_bitshift_int64  (int64_t  x, int8_t k) ;
extern uint8_t  GB_bitshift_uint8  (uint8_t  x, int8_t k) ;
extern uint16_t GB_bitshift_uint16 (uint16_t x, int8_t k) ;
extern uint32_t GB_bitshift_uint32 (uint32_t x, int8_t k) ;
extern uint64_t GB_bitshift_uint64 (uint64_t x, int8_t k) ;

