//------------------------------------------------------------------------------
// gbargminmax: argmin or argmax of a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:

// [x,p] = gbargminmax (A, minmax, dim)

// where minmax is 0 for min or 1 for max, and where dim = 1 to compute the
// argmin/max of each column of A, dim = 2 to compute the argmin/max of each
// row of A, or dim = 0 to compute the argmin/max of all of A.  For dim = 1 or
// 2, x and p are vectors of the same size.  For dim = 0, x is a scalar and p
// is 2-by-1, containing the row and column index of the argmin/max of A.

#include "gb_interface.h"

#define USAGE "usage: [x,p] = gbargminmax (A, minmax, dim)"

//------------------------------------------------------------------------------
// tuple types
//------------------------------------------------------------------------------

// The gb_tuple_* types pair a row or column index (k) with a value v.

typedef struct { int64_t k ; bool     v ; } gb_tuple_bool ;
typedef struct { int64_t k ; int8_t   v ; } gb_tuple_int8 ;
typedef struct { int64_t k ; int16_t  v ; } gb_tuple_int16 ;
typedef struct { int64_t k ; int32_t  v ; } gb_tuple_int32 ;
typedef struct { int64_t k ; int64_t  v ; } gb_tuple_int64 ;
typedef struct { int64_t k ; uint8_t  v ; } gb_tuple_uint8 ;
typedef struct { int64_t k ; uint16_t v ; } gb_tuple_uint16 ;
typedef struct { int64_t k ; uint32_t v ; } gb_tuple_uint32 ;
typedef struct { int64_t k ; uint64_t v ; } gb_tuple_uint64 ;
typedef struct { int64_t k ; float    v ; } gb_tuple_fp32 ;
typedef struct { int64_t k ; double   v ; } gb_tuple_fp64 ;

#define BOOL_K   "typedef struct { int64_t k ; bool     v ; } gb_tuple_bool ;"
#define INT8_K   "typedef struct { int64_t k ; int8_t   v ; } gb_tuple_int8 ;"
#define INT16_K  "typedef struct { int64_t k ; int16_t  v ; } gb_tuple_int16 ;"
#define INT32_K  "typedef struct { int64_t k ; int32_t  v ; } gb_tuple_int32 ;"
#define INT64_K  "typedef struct { int64_t k ; int64_t  v ; } gb_tuple_int64 ;"
#define UINT8_K  "typedef struct { int64_t k ; uint8_t  v ; } gb_tuple_uint8 ;"
#define UINT16_K "typedef struct { int64_t k ; uint16_t v ; } gb_tuple_uint16 ;"
#define UINT32_K "typedef struct { int64_t k ; uint32_t v ; } gb_tuple_uint32 ;"
#define UINT64_K "typedef struct { int64_t k ; uint64_t v ; } gb_tuple_uint64 ;"
#define FP32_K   "typedef struct { int64_t k ; float    v ; } gb_tuple_fp32 ;"
#define FP64_K   "typedef struct { int64_t k ; double   v ; } gb_tuple_fp64 ;"

// The gb_tuple3_* types have both row and column indices, with a value v.

typedef struct { int64_t i,j ; bool     v ; } gb_tuple3_bool ;
typedef struct { int64_t i,j ; int8_t   v ; } gb_tuple3_int8 ;
typedef struct { int64_t i,j ; int16_t  v ; } gb_tuple3_int16 ;
typedef struct { int64_t i,j ; int32_t  v ; } gb_tuple3_int32 ;
typedef struct { int64_t i,j ; int64_t  v ; } gb_tuple3_int64 ;
typedef struct { int64_t i,j ; uint8_t  v ; } gb_tuple3_uint8 ;
typedef struct { int64_t i,j ; uint16_t v ; } gb_tuple3_uint16 ;
typedef struct { int64_t i,j ; uint32_t v ; } gb_tuple3_uint32 ;
typedef struct { int64_t i,j ; uint64_t v ; } gb_tuple3_uint64 ;
typedef struct { int64_t i,j ; float    v ; } gb_tuple3_fp32 ;
typedef struct { int64_t i,j ; double   v ; } gb_tuple3_fp64 ;

#define BOOL_IJ   "typedef struct { int64_t i,j ; bool     v ; } gb_tuple3_bool ;"
#define INT8_IJ   "typedef struct { int64_t i,j ; int8_t   v ; } gb_tuple3_int8 ;"
#define INT16_IJ  "typedef struct { int64_t i,j ; int16_t  v ; } gb_tuple3_int16 ;"
#define INT32_IJ  "typedef struct { int64_t i,j ; int32_t  v ; } gb_tuple3_int32 ;"
#define INT64_IJ  "typedef struct { int64_t i,j ; int64_t  v ; } gb_tuple3_int64 ;"
#define UINT8_IJ  "typedef struct { int64_t i,j ; uint8_t  v ; } gb_tuple3_uint8 ;"
#define UINT16_IJ "typedef struct { int64_t i,j ; uint16_t v ; } gb_tuple3_uint16 ;"
#define UINT32_IJ "typedef struct { int64_t i,j ; uint32_t v ; } gb_tuple3_uint32 ;"
#define UINT64_IJ "typedef struct { int64_t i,j ; uint64_t v ; } gb_tuple3_uint64 ;"
#define FP32_IJ   "typedef struct { int64_t i,j ; float    v ; } gb_tuple3_fp32 ;"
#define FP64_IJ   "typedef struct { int64_t i,j ; double   v ; } gb_tuple3_fp64 ;"

//------------------------------------------------------------------------------
// gb_make_* index binary functions
//------------------------------------------------------------------------------

// These functions take a value v from a matrix and combine it with its row/col
// index k to make a tuple (k,v).

    void gb_make_bool (gb_tuple_bool *z,
        const bool *x, uint64_t ix, uint64_t jx,
        const void *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_bool (gb_tuple_bool *z,
        const bool *x, uint64_t ix, uint64_t jx,
        const void *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_BOOL \
   "void gb_make_bool (gb_tuple_bool *z,                \n" \
   "    const bool *x, uint64_t ix, uint64_t jx,        \n" \
   "    const void *y, uint64_t iy, uint64_t jy,        \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

    void gb_make_int8 (gb_tuple_int8 *z,
        const int8_t *x, uint64_t ix, uint64_t jx,
        const void   *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_int8 (gb_tuple_int8 *z,
        const int8_t *x, uint64_t ix, uint64_t jx,
        const void   *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_INT8 \
   "void gb_make_int8 (gb_tuple_int8 *z,                \n" \
   "    const int8_t *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void   *y, uint64_t iy, uint64_t jy,      \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

    void gb_make_int16 (gb_tuple_int16 *z,
        const int16_t *x, uint64_t ix, uint64_t jx,
        const void    *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_int16 (gb_tuple_int16 *z,
        const int16_t *x, uint64_t ix, uint64_t jx,
        const void    *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_INT16 \
   "void gb_make_int16 (gb_tuple_int16 *z,              \n" \
   "    const int16_t *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void    *y, uint64_t iy, uint64_t jy,     \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

    void gb_make_int32 (gb_tuple_int32 *z,
        const int32_t *x, uint64_t ix, uint64_t jx,
        const void    *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_int32 (gb_tuple_int32 *z,
        const int32_t *x, uint64_t ix, uint64_t jx,
        const void    *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_INT32 \
   "void gb_make_int32 (gb_tuple_int32 *z,              \n" \
   "    const int32_t *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void    *y, uint64_t iy, uint64_t jy,     \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

    void gb_make_int64 (gb_tuple_int64 *z,
        const int64_t *x, uint64_t ix, uint64_t jx,
        const void    *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_int64 (gb_tuple_int64 *z,
        const int64_t *x, uint64_t ix, uint64_t jx,
        const void    *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_INT64 \
   "void gb_make_int64 (gb_tuple_int64 *z,              \n" \
   "    const int64_t *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void    *y, uint64_t iy, uint64_t jy,     \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

    void gb_make_uint8 (gb_tuple_uint8 *z,
        const uint8_t *x, uint64_t ix, uint64_t jx,
        const void    *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_uint8 (gb_tuple_uint8 *z,
        const uint8_t *x, uint64_t ix, uint64_t jx,
        const void    *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_UINT8 \
   "void gb_make_uint8 (gb_tuple_uint8 *z,              \n" \
   "    const uint8_t *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void    *y, uint64_t iy, uint64_t jy,     \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

    void gb_make_uint16 (gb_tuple_uint16 *z,
        const uint16_t *x, uint64_t ix, uint64_t jx,
        const void     *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_uint16 (gb_tuple_uint16 *z,
        const uint16_t *x, uint64_t ix, uint64_t jx,
        const void     *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_UINT16 \
   "void gb_make_uint16 (gb_tuple_uint16 *z,            \n" \
   "    const uint16_t *x, uint64_t ix, uint64_t jx,    \n" \
   "    const void     *y, uint64_t iy, uint64_t jy,    \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

    void gb_make_uint32 (gb_tuple_uint32 *z,
        const uint32_t *x, uint64_t ix, uint64_t jx,
        const void     *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_uint32 (gb_tuple_uint32 *z,
        const uint32_t *x, uint64_t ix, uint64_t jx,
        const void     *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_UINT32 \
   "void gb_make_uint32 (gb_tuple_uint32 *z,            \n" \
   "    const uint32_t *x, uint64_t ix, uint64_t jx,    \n" \
   "    const void     *y, uint64_t iy, uint64_t jy,    \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

    void gb_make_uint64 (gb_tuple_uint64 *z,
        const uint64_t *x, uint64_t ix, uint64_t jx,
        const void     *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_uint64 (gb_tuple_uint64 *z,
        const uint64_t *x, uint64_t ix, uint64_t jx,
        const void     *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_UINT64 \
   "void gb_make_uint64 (gb_tuple_uint64 *z,            \n" \
   "    const uint64_t *x, uint64_t ix, uint64_t jx,    \n" \
   "    const void     *y, uint64_t iy, uint64_t jy,    \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

    void gb_make_fp32 (gb_tuple_fp32 *z,
        const float *x, uint64_t ix, uint64_t jx,
        const void  *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_fp32 (gb_tuple_fp32 *z,
        const float *x, uint64_t ix, uint64_t jx,
        const void  *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_FP32 \
   "void gb_make_fp32 (gb_tuple_fp32 *z,                \n" \
   "    const float *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void  *y, uint64_t iy, uint64_t jy,       \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

    void gb_make_fp64 (gb_tuple_fp64 *z,
        const double *x, uint64_t ix, uint64_t jx,
        const void   *y, uint64_t iy, uint64_t jy,
        const void *theta) ;
    void gb_make_fp64 (gb_tuple_fp64 *z,
        const double *x, uint64_t ix, uint64_t jx,
        const void   *y, uint64_t iy, uint64_t jy,
        const void *theta)
    {
        z->k = (int64_t) jx + 1 ;
        z->v = (*x) ;
    }

#define MAKE_FP64 \
   "void gb_make_fp64 (gb_tuple_fp64 *z,                \n" \
   "    const double *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void   *y, uint64_t iy, uint64_t jy,      \n" \
   "    const void *theta)                              \n" \
   "{                                                   \n" \
   "    z->k = (int64_t) jx + 1 ;                       \n" \
   "    z->v = (*x) ;                                   \n" \
   "}                                                   \n"

//------------------------------------------------------------------------------
// gb_make3a_* index unary functions
//------------------------------------------------------------------------------

// These unary functions take a 2-tuple (k,v) and combine it with another index
// index i to make a 3-tuple (i,k,v).  It is only used for vectors or n-by-1
// matrices, so jx is always zero.

    void gb_make3a_bool (gb_tuple3_bool *z,
        const gb_tuple_bool *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_bool (gb_tuple3_bool *z,
        const gb_tuple_bool *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_BOOL \
   "void gb_make3a_bool (gb_tuple3_bool *z,                     \n" \
   "    const gb_tuple_bool *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_bool (gb_tuple3_bool *z,                     \n" \
   "    const gb_tuple_bool *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3a_int8 (gb_tuple3_int8 *z,
        const gb_tuple_int8 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_int8 (gb_tuple3_int8 *z,
        const gb_tuple_int8 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_INT8 \
   "void gb_make3a_int8 (gb_tuple3_int8 *z,                     \n" \
   "    const gb_tuple_int8 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_int8 (gb_tuple3_int8 *z,                     \n" \
   "    const gb_tuple_int8 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3a_int16 (gb_tuple3_int16 *z,
        const gb_tuple_int16 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_int16 (gb_tuple3_int16 *z,
        const gb_tuple_int16 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_INT16 \
   "void gb_make3a_int16 (gb_tuple3_int16 *z,                   \n" \
   "    const gb_tuple_int16 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_int16 (gb_tuple3_int16 *z,                   \n" \
   "    const gb_tuple_int16 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3a_int32 (gb_tuple3_int32 *z,
        const gb_tuple_int32 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_int32 (gb_tuple3_int32 *z,
        const gb_tuple_int32 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_INT32 \
   "void gb_make3a_int32 (gb_tuple3_int32 *z,                   \n" \
   "    const gb_tuple_int32 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_int32 (gb_tuple3_int32 *z,                   \n" \
   "    const gb_tuple_int32 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3a_int64 (gb_tuple3_int64 *z,
        const gb_tuple_int64 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_int64 (gb_tuple3_int64 *z,
        const gb_tuple_int64 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_INT64 \
   "void gb_make3a_int64 (gb_tuple3_int64 *z,                   \n" \
   "    const gb_tuple_int64 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_int64 (gb_tuple3_int64 *z,                   \n" \
   "    const gb_tuple_int64 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3a_uint8 (gb_tuple3_uint8 *z,
        const gb_tuple_uint8 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_uint8 (gb_tuple3_uint8 *z,
        const gb_tuple_uint8 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_UINT8 \
   "void gb_make3a_uint8 (gb_tuple3_uint8 *z,                   \n" \
   "    const gb_tuple_uint8 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_uint8 (gb_tuple3_uint8 *z,                   \n" \
   "    const gb_tuple_uint8 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3a_uint16 (gb_tuple3_uint16 *z,
        const gb_tuple_uint16 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_uint16 (gb_tuple3_uint16 *z,
        const gb_tuple_uint16 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_UINT16 \
   "void gb_make3a_uint16 (gb_tuple3_uint16 *z,                 \n" \
   "    const gb_tuple_uint16 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_uint16 (gb_tuple3_uint16 *z,                 \n" \
   "    const gb_tuple_uint16 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3a_uint32 (gb_tuple3_uint32 *z,
        const gb_tuple_uint32 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_uint32 (gb_tuple3_uint32 *z,
        const gb_tuple_uint32 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_UINT32 \
   "void gb_make3a_uint32 (gb_tuple3_uint32 *z,                 \n" \
   "    const gb_tuple_uint32 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_uint32 (gb_tuple3_uint32 *z,                 \n" \
   "    const gb_tuple_uint32 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3a_uint64 (gb_tuple3_uint64 *z,
        const gb_tuple_uint64 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_uint64 (gb_tuple3_uint64 *z,
        const gb_tuple_uint64 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_UINT64 \
   "void gb_make3a_uint64 (gb_tuple3_uint64 *z,                 \n" \
   "    const gb_tuple_uint64 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_uint64 (gb_tuple3_uint64 *z,                 \n" \
   "    const gb_tuple_uint64 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3a_fp32 (gb_tuple3_fp32 *z,
        const gb_tuple_fp32 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_fp32 (gb_tuple3_fp32 *z,
        const gb_tuple_fp32 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_FP32 \
   "void gb_make3a_fp32 (gb_tuple3_fp32 *z,                     \n" \
   "    const gb_tuple_fp32 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_fp32 (gb_tuple3_fp32 *z,                     \n" \
   "    const gb_tuple_fp32 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3a_fp64 (gb_tuple3_fp64 *z,
        const gb_tuple_fp64 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3a_fp64 (gb_tuple3_fp64 *z,
        const gb_tuple_fp64 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = (int64_t) ix + 1 ;
        z->j = x->k ;
        z->v = x->v ;
    }

#define MAKE3a_FP64 \
   "void gb_make3a_fp64 (gb_tuple3_fp64 *z,                     \n" \
   "    const gb_tuple_fp64 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3a_fp64 (gb_tuple3_fp64 *z,                     \n" \
   "    const gb_tuple_fp64 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = (int64_t) ix + 1 ;                               \n" \
   "    z->j = x->k ;                                           \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

//------------------------------------------------------------------------------
// gb_make3b_* index unary functions
//------------------------------------------------------------------------------

// These unary functions take a 2-tuple (k,v) and combine it with another index
// index i to make a 3-tuple (k,i,v).  It is only used for vectors or n-by-1
// matrices, so jx is always zero.

    void gb_make3b_bool (gb_tuple3_bool *z,
        const gb_tuple_bool *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_bool (gb_tuple3_bool *z,
        const gb_tuple_bool *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_BOOL \
   "void gb_make3b_bool (gb_tuple3_bool *z,                     \n" \
   "    const gb_tuple_bool *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_bool (gb_tuple3_bool *z,                     \n" \
   "    const gb_tuple_bool *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3b_int8 (gb_tuple3_int8 *z,
        const gb_tuple_int8 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_int8 (gb_tuple3_int8 *z,
        const gb_tuple_int8 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_INT8 \
   "void gb_make3b_int8 (gb_tuple3_int8 *z,                     \n" \
   "    const gb_tuple_int8 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_int8 (gb_tuple3_int8 *z,                     \n" \
   "    const gb_tuple_int8 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3b_int16 (gb_tuple3_int16 *z,
        const gb_tuple_int16 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_int16 (gb_tuple3_int16 *z,
        const gb_tuple_int16 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_INT16 \
   "void gb_make3b_int16 (gb_tuple3_int16 *z,                   \n" \
   "    const gb_tuple_int16 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_int16 (gb_tuple3_int16 *z,                   \n" \
   "    const gb_tuple_int16 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3b_int32 (gb_tuple3_int32 *z,
        const gb_tuple_int32 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_int32 (gb_tuple3_int32 *z,
        const gb_tuple_int32 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_INT32 \
   "void gb_make3b_int32 (gb_tuple3_int32 *z,                   \n" \
   "    const gb_tuple_int32 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_int32 (gb_tuple3_int32 *z,                   \n" \
   "    const gb_tuple_int32 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3b_int64 (gb_tuple3_int64 *z,
        const gb_tuple_int64 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_int64 (gb_tuple3_int64 *z,
        const gb_tuple_int64 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_INT64 \
   "void gb_make3b_int64 (gb_tuple3_int64 *z,                   \n" \
   "    const gb_tuple_int64 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_int64 (gb_tuple3_int64 *z,                   \n" \
   "    const gb_tuple_int64 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3b_uint8 (gb_tuple3_uint8 *z,
        const gb_tuple_uint8 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_uint8 (gb_tuple3_uint8 *z,
        const gb_tuple_uint8 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_UINT8 \
   "void gb_make3b_uint8 (gb_tuple3_uint8 *z,                   \n" \
   "    const gb_tuple_uint8 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_uint8 (gb_tuple3_uint8 *z,                   \n" \
   "    const gb_tuple_uint8 *x, uint64_t ix, uint64_t jx,      \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3b_uint16 (gb_tuple3_uint16 *z,
        const gb_tuple_uint16 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_uint16 (gb_tuple3_uint16 *z,
        const gb_tuple_uint16 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_UINT16 \
   "void gb_make3b_uint16 (gb_tuple3_uint16 *z,                 \n" \
   "    const gb_tuple_uint16 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_uint16 (gb_tuple3_uint16 *z,                 \n" \
   "    const gb_tuple_uint16 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3b_uint32 (gb_tuple3_uint32 *z,
        const gb_tuple_uint32 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_uint32 (gb_tuple3_uint32 *z,
        const gb_tuple_uint32 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_UINT32 \
   "void gb_make3b_uint32 (gb_tuple3_uint32 *z,                 \n" \
   "    const gb_tuple_uint32 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_uint32 (gb_tuple3_uint32 *z,                 \n" \
   "    const gb_tuple_uint32 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3b_uint64 (gb_tuple3_uint64 *z,
        const gb_tuple_uint64 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_uint64 (gb_tuple3_uint64 *z,
        const gb_tuple_uint64 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_UINT64 \
   "void gb_make3b_uint64 (gb_tuple3_uint64 *z,                 \n" \
   "    const gb_tuple_uint64 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_uint64 (gb_tuple3_uint64 *z,                 \n" \
   "    const gb_tuple_uint64 *x, uint64_t ix, uint64_t jx,     \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3b_fp32 (gb_tuple3_fp32 *z,
        const gb_tuple_fp32 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_fp32 (gb_tuple3_fp32 *z,
        const gb_tuple_fp32 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_FP32 \
   "void gb_make3b_fp32 (gb_tuple3_fp32 *z,                     \n" \
   "    const gb_tuple_fp32 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_fp32 (gb_tuple3_fp32 *z,                     \n" \
   "    const gb_tuple_fp32 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

    void gb_make3b_fp64 (gb_tuple3_fp64 *z,
        const gb_tuple_fp64 *x, uint64_t ix, uint64_t jx,
        const void *y) ;
    void gb_make3b_fp64 (gb_tuple3_fp64 *z,
        const gb_tuple_fp64 *x, uint64_t ix, uint64_t jx,
        const void *y)
    {
        z->i = x->k ;
        z->j = (int64_t) ix + 1 ;
        z->v = x->v ;
    }

#define MAKE3b_FP64 \
   "void gb_make3b_fp64 (gb_tuple3_fp64 *z,                     \n" \
   "    const gb_tuple_fp64 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y) ;                                        \n" \
   "void gb_make3b_fp64 (gb_tuple3_fp64 *z,                     \n" \
   "    const gb_tuple_fp64 *x, uint64_t ix, uint64_t jx,       \n" \
   "    const void *y)                                          \n" \
   "{                                                           \n" \
   "    z->i = x->k ;                                           \n" \
   "    z->j = (int64_t) ix + 1 ;                               \n" \
   "    z->v = x->v ;                                           \n" \
   "}                                                           \n" \

//------------------------------------------------------------------------------
// gb_max_* functions:
//------------------------------------------------------------------------------

// These functions find the max of two 2-tuples.  The tuple with the larger
// value v is selected.  In case of ties, pick the one with the smaller index
// k.

    void gb_max_bool (gb_tuple_bool *z, const gb_tuple_bool *x, const gb_tuple_bool *y) ;
    void gb_max_bool (gb_tuple_bool *z, const gb_tuple_bool *x, const gb_tuple_bool *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_BOOL \
   "void gb_max_bool (gb_tuple_bool *z, const gb_tuple_bool *x, const gb_tuple_bool *y)\n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_max_int8 (gb_tuple_int8 *z, const gb_tuple_int8 *x, const gb_tuple_int8 *y) ;
    void gb_max_int8 (gb_tuple_int8 *z, const gb_tuple_int8 *x, const gb_tuple_int8 *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_INT8 \
   "void gb_max_int8 (gb_tuple_int8 *z, const gb_tuple_int8 *x, const gb_tuple_int8 *y)\n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_max_int16 (gb_tuple_int16 *z, const gb_tuple_int16 *x, const gb_tuple_int16 *y);
    void gb_max_int16 (gb_tuple_int16 *z, const gb_tuple_int16 *x, const gb_tuple_int16 *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_INT16 \
   "void gb_max_int16 (gb_tuple_int16 *z, const gb_tuple_int16 *x, const gb_tuple_int16 *y)\n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_max_int32 (gb_tuple_int32 *z, const gb_tuple_int32 *x, const gb_tuple_int32 *y);
    void gb_max_int32 (gb_tuple_int32 *z, const gb_tuple_int32 *x, const gb_tuple_int32 *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_INT32 \
   "void gb_max_int32 (gb_tuple_int32 *z, const gb_tuple_int32 *x, const gb_tuple_int32 *y)\n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_max_int64 (gb_tuple_int64 *z, const gb_tuple_int64 *x, const gb_tuple_int64 *y);
    void gb_max_int64 (gb_tuple_int64 *z, const gb_tuple_int64 *x, const gb_tuple_int64 *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_INT64 \
   "void gb_max_int64 (gb_tuple_int64 *z, const gb_tuple_int64 *x, const gb_tuple_int64 *y)\n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_max_uint8 (gb_tuple_uint8 *z, const gb_tuple_uint8 *x, const gb_tuple_uint8 *y);
    void gb_max_uint8 (gb_tuple_uint8 *z, const gb_tuple_uint8 *x, const gb_tuple_uint8 *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_UINT8 \
   "void gb_max_uint8 (gb_tuple_uint8 *z, const gb_tuple_uint8 *x, const gb_tuple_uint8 *y)\n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_max_uint16 (gb_tuple_uint16 *z, const gb_tuple_uint16 *x, const gb_tuple_uint16 *y) ;
    void gb_max_uint16 (gb_tuple_uint16 *z, const gb_tuple_uint16 *x, const gb_tuple_uint16 *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_UINT16 \
   "void gb_max_uint16 (gb_tuple_uint16 *z, const gb_tuple_uint16 *x, const gb_tuple_uint16 *y) \n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_max_uint32 (gb_tuple_uint32 *z, const gb_tuple_uint32 *x, const gb_tuple_uint32 *y) ;
    void gb_max_uint32 (gb_tuple_uint32 *z, const gb_tuple_uint32 *x, const gb_tuple_uint32 *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_UINT32 \
   "void gb_max_uint32 (gb_tuple_uint32 *z, const gb_tuple_uint32 *x, const gb_tuple_uint32 *y) \n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_max_uint64 (gb_tuple_uint64 *z, const gb_tuple_uint64 *x, const gb_tuple_uint64 *y) ;
    void gb_max_uint64 (gb_tuple_uint64 *z, const gb_tuple_uint64 *x, const gb_tuple_uint64 *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_UINT64 \
   "void gb_max_uint64 (gb_tuple_uint64 *z, const gb_tuple_uint64 *x, const gb_tuple_uint64 *y) \n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_max_fp32 (gb_tuple_fp32 *z, const gb_tuple_fp32 *x, const gb_tuple_fp32 *y) ;
    void gb_max_fp32 (gb_tuple_fp32 *z, const gb_tuple_fp32 *x, const gb_tuple_fp32 *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_FP32 \
   "void gb_max_fp32 (gb_tuple_fp32 *z, const gb_tuple_fp32 *x, const gb_tuple_fp32 *y)\n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_max_fp64 (gb_tuple_fp64 *z, const gb_tuple_fp64 *x, const gb_tuple_fp64 *y) ;
    void gb_max_fp64 (gb_tuple_fp64 *z, const gb_tuple_fp64 *x, const gb_tuple_fp64 *y)
    {
        if (x->v > y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MAX_FP64 \
   "void gb_max_fp64 (gb_tuple_fp64 *z, const gb_tuple_fp64 *x, const gb_tuple_fp64 *y)\n" \
   "{                                                       \n" \
   "    if (x->v > y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

//------------------------------------------------------------------------------
// gb_min_* functions:
//------------------------------------------------------------------------------

// These functions find the max of two 2-tuples.  The tuple with the larger
// value v is selected.  In case of ties, pick the one with the smaller index
// k.

    void gb_min_bool (gb_tuple_bool *z, const gb_tuple_bool *x, const gb_tuple_bool *y) ;
    void gb_min_bool (gb_tuple_bool *z, const gb_tuple_bool *x, const gb_tuple_bool *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_BOOL \
   "void gb_min_bool (gb_tuple_bool *z, const gb_tuple_bool *x, const gb_tuple_bool *y)\n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_min_int8 (gb_tuple_int8 *z, const gb_tuple_int8 *x, const gb_tuple_int8 *y) ;
    void gb_min_int8 (gb_tuple_int8 *z, const gb_tuple_int8 *x, const gb_tuple_int8 *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_INT8 \
   "void gb_min_int8 (gb_tuple_int8 *z, const gb_tuple_int8 *x, const gb_tuple_int8 *y)\n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_min_int16 (gb_tuple_int16 *z, const gb_tuple_int16 *x, const gb_tuple_int16 *y) ;
    void gb_min_int16 (gb_tuple_int16 *z, const gb_tuple_int16 *x, const gb_tuple_int16 *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_INT16 \
   "void gb_min_int16 (gb_tuple_int16 *z, const gb_tuple_int16 *x, const gb_tuple_int16 *y)\n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_min_int32 (gb_tuple_int32 *z, const gb_tuple_int32 *x, const gb_tuple_int32 *y) ;
    void gb_min_int32 (gb_tuple_int32 *z, const gb_tuple_int32 *x, const gb_tuple_int32 *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_INT32 \
   "void gb_min_int32 (gb_tuple_int32 *z, const gb_tuple_int32 *x, const gb_tuple_int32 *y)\n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_min_int64 (gb_tuple_int64 *z, const gb_tuple_int64 *x, const gb_tuple_int64 *y) ;
    void gb_min_int64 (gb_tuple_int64 *z, const gb_tuple_int64 *x, const gb_tuple_int64 *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_INT64 \
   "void gb_min_int64 (gb_tuple_int64 *z, const gb_tuple_int64 *x, const gb_tuple_int64 *y)\n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_min_uint8 (gb_tuple_uint8 *z, const gb_tuple_uint8 *x, const gb_tuple_uint8 *y) ;
    void gb_min_uint8 (gb_tuple_uint8 *z, const gb_tuple_uint8 *x, const gb_tuple_uint8 *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_UINT8 \
   "void gb_min_uint8 (gb_tuple_uint8 *z, const gb_tuple_uint8 *x, const gb_tuple_uint8 *y)\n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_min_uint16 (gb_tuple_uint16 *z, const gb_tuple_uint16 *x, const gb_tuple_uint16 *y) ;
    void gb_min_uint16 (gb_tuple_uint16 *z, const gb_tuple_uint16 *x, const gb_tuple_uint16 *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_UINT16 \
   "void gb_min_uint16 (gb_tuple_uint16 *z, const gb_tuple_uint16 *x, const gb_tuple_uint16 *y) \n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_min_uint32 (gb_tuple_uint32 *z, const gb_tuple_uint32 *x, const gb_tuple_uint32 *y) ;
    void gb_min_uint32 (gb_tuple_uint32 *z, const gb_tuple_uint32 *x, const gb_tuple_uint32 *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_UINT32 \
   "void gb_min_uint32 (gb_tuple_uint32 *z, const gb_tuple_uint32 *x, const gb_tuple_uint32 *y) \n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_min_uint64 (gb_tuple_uint64 *z, const gb_tuple_uint64 *x, const gb_tuple_uint64 *y) ;
    void gb_min_uint64 (gb_tuple_uint64 *z, const gb_tuple_uint64 *x, const gb_tuple_uint64 *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_UINT64 \
   "void gb_min_uint64 (gb_tuple_uint64 *z, const gb_tuple_uint64 *x, const gb_tuple_uint64 *y) \n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_min_fp32 (gb_tuple_fp32 *z, const gb_tuple_fp32 *x, const gb_tuple_fp32 *y) ;
    void gb_min_fp32 (gb_tuple_fp32 *z, const gb_tuple_fp32 *x, const gb_tuple_fp32 *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_FP32 \
   "void gb_min_fp32 (gb_tuple_fp32 *z, const gb_tuple_fp32 *x, const gb_tuple_fp32 *y)\n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

    void gb_min_fp64 (gb_tuple_fp64 *z, const gb_tuple_fp64 *x, const gb_tuple_fp64 *y) ;
    void gb_min_fp64 (gb_tuple_fp64 *z, const gb_tuple_fp64 *x, const gb_tuple_fp64 *y)
    {
        if (x->v < y->v || (x->v == y->v && x->k < y->k))
        {
            z->k = x->k ;
            z->v = x->v ;
        }
        else
        {
            z->k = y->k ;
            z->v = y->v ;
        }
    }

#define MIN_FP64 \
   "void gb_min_fp64 (gb_tuple_fp64 *z, const gb_tuple_fp64 *x, const gb_tuple_fp64 *y)\n" \
   "{                                                       \n" \
   "    if (x->v < y->v || (x->v == y->v && x->k < y->k))   \n" \
   "    {                                                   \n" \
   "        z->k = x->k ;                                   \n" \
   "        z->v = x->v ;                                   \n" \
   "    }                                                   \n" \
   "    else                                                \n" \
   "    {                                                   \n" \
   "        z->k = y->k ;                                   \n" \
   "        z->v = y->v ;                                   \n" \
   "    }                                                   \n" \
   "}                                                       \n"

//------------------------------------------------------------------------------
// gb_max3_* functions:
//------------------------------------------------------------------------------

// These functions find the max of two 3-tuples.  The tuple with the larger
// value v is selected.  In case of ties, pick the one with the smaller index
// i.  If both the value and i tie, pick the one with the smaller j.

    void gb_max3_bool (gb_tuple3_bool *z, const gb_tuple3_bool *x, const gb_tuple3_bool *y) ;
    void gb_max3_bool (gb_tuple3_bool *z, const gb_tuple3_bool *x, const gb_tuple3_bool *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_BOOL \
   "void gb_max3_bool (gb_tuple3_bool *z, const gb_tuple3_bool *x, const gb_tuple3_bool *y) ;   \n" \
   "void gb_max3_bool (gb_tuple3_bool *z, const gb_tuple3_bool *x, const gb_tuple3_bool *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_max3_int8 (gb_tuple3_int8 *z, const gb_tuple3_int8 *x, const gb_tuple3_int8 *y) ;
    void gb_max3_int8 (gb_tuple3_int8 *z, const gb_tuple3_int8 *x, const gb_tuple3_int8 *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_INT8 \
   "void gb_max3_int8 (gb_tuple3_int8 *z, const gb_tuple3_int8 *x, const gb_tuple3_int8 *y) ;   \n" \
   "void gb_max3_int8 (gb_tuple3_int8 *z, const gb_tuple3_int8 *x, const gb_tuple3_int8 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_max3_int16 (gb_tuple3_int16 *z, const gb_tuple3_int16 *x, const gb_tuple3_int16 *y) ;
    void gb_max3_int16 (gb_tuple3_int16 *z, const gb_tuple3_int16 *x, const gb_tuple3_int16 *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_INT16 \
   "void gb_max3_int16 (gb_tuple3_int16 *z, const gb_tuple3_int16 *x, const gb_tuple3_int16 *y) ;   \n" \
   "void gb_max3_int16 (gb_tuple3_int16 *z, const gb_tuple3_int16 *x, const gb_tuple3_int16 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_max3_int32 (gb_tuple3_int32 *z, const gb_tuple3_int32 *x, const gb_tuple3_int32 *y) ;
    void gb_max3_int32 (gb_tuple3_int32 *z, const gb_tuple3_int32 *x, const gb_tuple3_int32 *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_INT32 \
   "void gb_max3_int32 (gb_tuple3_int32 *z, const gb_tuple3_int32 *x, const gb_tuple3_int32 *y) ;   \n" \
   "void gb_max3_int32 (gb_tuple3_int32 *z, const gb_tuple3_int32 *x, const gb_tuple3_int32 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_max3_int64 (gb_tuple3_int64 *z, const gb_tuple3_int64 *x, const gb_tuple3_int64 *y) ;
    void gb_max3_int64 (gb_tuple3_int64 *z, const gb_tuple3_int64 *x, const gb_tuple3_int64 *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_INT64 \
   "void gb_max3_int64 (gb_tuple3_int64 *z, const gb_tuple3_int64 *x, const gb_tuple3_int64 *y) ;   \n" \
   "void gb_max3_int64 (gb_tuple3_int64 *z, const gb_tuple3_int64 *x, const gb_tuple3_int64 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_max3_uint8 (gb_tuple3_uint8 *z, const gb_tuple3_uint8 *x, const gb_tuple3_uint8 *y) ;
    void gb_max3_uint8 (gb_tuple3_uint8 *z, const gb_tuple3_uint8 *x, const gb_tuple3_uint8 *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_UINT8 \
   "void gb_max3_uint8 (gb_tuple3_uint8 *z, const gb_tuple3_uint8 *x, const gb_tuple3_uint8 *y) ;   \n" \
   "void gb_max3_uint8 (gb_tuple3_uint8 *z, const gb_tuple3_uint8 *x, const gb_tuple3_uint8 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_max3_uint16 (gb_tuple3_uint16 *z, const gb_tuple3_uint16 *x, const gb_tuple3_uint16 *y) ;
    void gb_max3_uint16 (gb_tuple3_uint16 *z, const gb_tuple3_uint16 *x, const gb_tuple3_uint16 *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_UINT16 \
   "void gb_max3_uint16 (gb_tuple3_uint16 *z, const gb_tuple3_uint16 *x, const gb_tuple3_uint16 *y) ;   \n" \
   "void gb_max3_uint16 (gb_tuple3_uint16 *z, const gb_tuple3_uint16 *x, const gb_tuple3_uint16 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_max3_uint32 (gb_tuple3_uint32 *z, const gb_tuple3_uint32 *x, const gb_tuple3_uint32 *y) ;
    void gb_max3_uint32 (gb_tuple3_uint32 *z, const gb_tuple3_uint32 *x, const gb_tuple3_uint32 *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_UINT32 \
   "void gb_max3_uint32 (gb_tuple3_uint32 *z, const gb_tuple3_uint32 *x, const gb_tuple3_uint32 *y) ;   \n" \
   "void gb_max3_uint32 (gb_tuple3_uint32 *z, const gb_tuple3_uint32 *x, const gb_tuple3_uint32 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_max3_uint64 (gb_tuple3_uint64 *z, const gb_tuple3_uint64 *x, const gb_tuple3_uint64 *y) ;
    void gb_max3_uint64 (gb_tuple3_uint64 *z, const gb_tuple3_uint64 *x, const gb_tuple3_uint64 *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_UINT64 \
   "void gb_max3_uint64 (gb_tuple3_uint64 *z, const gb_tuple3_uint64 *x, const gb_tuple3_uint64 *y) ;   \n" \
   "void gb_max3_uint64 (gb_tuple3_uint64 *z, const gb_tuple3_uint64 *x, const gb_tuple3_uint64 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_max3_fp32 (gb_tuple3_fp32 *z, const gb_tuple3_fp32 *x, const gb_tuple3_fp32 *y) ;
    void gb_max3_fp32 (gb_tuple3_fp32 *z, const gb_tuple3_fp32 *x, const gb_tuple3_fp32 *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_FP32 \
   "void gb_max3_fp32 (gb_tuple3_fp32 *z, const gb_tuple3_fp32 *x, const gb_tuple3_fp32 *y) ;   \n" \
   "void gb_max3_fp32 (gb_tuple3_fp32 *z, const gb_tuple3_fp32 *x, const gb_tuple3_fp32 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_max3_fp64 (gb_tuple3_fp64 *z, const gb_tuple3_fp64 *x, const gb_tuple3_fp64 *y) ;
    void gb_max3_fp64 (gb_tuple3_fp64 *z, const gb_tuple3_fp64 *x, const gb_tuple3_fp64 *y)
    {
        if (x->v > y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MAX3_FP64 \
   "void gb_max3_fp64 (gb_tuple3_fp64 *z, const gb_tuple3_fp64 *x, const gb_tuple3_fp64 *y) ;   \n" \
   "void gb_max3_fp64 (gb_tuple3_fp64 *z, const gb_tuple3_fp64 *x, const gb_tuple3_fp64 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v > y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

//------------------------------------------------------------------------------
// gb_min3_* functions:
//------------------------------------------------------------------------------

// These functions find the min of two 3-tuples.  The tuple with the larger
// value v is selected.  In case of ties, pick the one with the smaller index
// i.  If both the value and i tie, pick the one with the smaller j.

    void gb_min3_bool (gb_tuple3_bool *z, const gb_tuple3_bool *x, const gb_tuple3_bool *y) ;
    void gb_min3_bool (gb_tuple3_bool *z, const gb_tuple3_bool *x, const gb_tuple3_bool *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_BOOL \
   "void gb_min3_bool (gb_tuple3_bool *z, const gb_tuple3_bool *x, const gb_tuple3_bool *y) ;   \n" \
   "void gb_min3_bool (gb_tuple3_bool *z, const gb_tuple3_bool *x, const gb_tuple3_bool *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_min3_int8 (gb_tuple3_int8 *z, const gb_tuple3_int8 *x, const gb_tuple3_int8 *y) ;
    void gb_min3_int8 (gb_tuple3_int8 *z, const gb_tuple3_int8 *x, const gb_tuple3_int8 *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_INT8 \
   "void gb_min3_int8 (gb_tuple3_int8 *z, const gb_tuple3_int8 *x, const gb_tuple3_int8 *y) ;   \n" \
   "void gb_min3_int8 (gb_tuple3_int8 *z, const gb_tuple3_int8 *x, const gb_tuple3_int8 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_min3_int16 (gb_tuple3_int16 *z, const gb_tuple3_int16 *x, const gb_tuple3_int16 *y) ;
    void gb_min3_int16 (gb_tuple3_int16 *z, const gb_tuple3_int16 *x, const gb_tuple3_int16 *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_INT16 \
   "void gb_min3_int16 (gb_tuple3_int16 *z, const gb_tuple3_int16 *x, const gb_tuple3_int16 *y) ;   \n" \
   "void gb_min3_int16 (gb_tuple3_int16 *z, const gb_tuple3_int16 *x, const gb_tuple3_int16 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_min3_int32 (gb_tuple3_int32 *z, const gb_tuple3_int32 *x, const gb_tuple3_int32 *y) ;
    void gb_min3_int32 (gb_tuple3_int32 *z, const gb_tuple3_int32 *x, const gb_tuple3_int32 *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_INT32 \
   "void gb_min3_int32 (gb_tuple3_int32 *z, const gb_tuple3_int32 *x, const gb_tuple3_int32 *y) ;   \n" \
   "void gb_min3_int32 (gb_tuple3_int32 *z, const gb_tuple3_int32 *x, const gb_tuple3_int32 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_min3_int64 (gb_tuple3_int64 *z, const gb_tuple3_int64 *x, const gb_tuple3_int64 *y) ;
    void gb_min3_int64 (gb_tuple3_int64 *z, const gb_tuple3_int64 *x, const gb_tuple3_int64 *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_INT64 \
   "void gb_min3_int64 (gb_tuple3_int64 *z, const gb_tuple3_int64 *x, const gb_tuple3_int64 *y) ;   \n" \
   "void gb_min3_int64 (gb_tuple3_int64 *z, const gb_tuple3_int64 *x, const gb_tuple3_int64 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_min3_uint8 (gb_tuple3_uint8 *z, const gb_tuple3_uint8 *x, const gb_tuple3_uint8 *y) ;
    void gb_min3_uint8 (gb_tuple3_uint8 *z, const gb_tuple3_uint8 *x, const gb_tuple3_uint8 *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_UINT8 \
   "void gb_min3_uint8 (gb_tuple3_uint8 *z, const gb_tuple3_uint8 *x, const gb_tuple3_uint8 *y) ;   \n" \
   "void gb_min3_uint8 (gb_tuple3_uint8 *z, const gb_tuple3_uint8 *x, const gb_tuple3_uint8 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_min3_uint16 (gb_tuple3_uint16 *z, const gb_tuple3_uint16 *x, const gb_tuple3_uint16 *y) ;
    void gb_min3_uint16 (gb_tuple3_uint16 *z, const gb_tuple3_uint16 *x, const gb_tuple3_uint16 *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_UINT16 \
   "void gb_min3_uint16 (gb_tuple3_uint16 *z, const gb_tuple3_uint16 *x, const gb_tuple3_uint16 *y) ;   \n" \
   "void gb_min3_uint16 (gb_tuple3_uint16 *z, const gb_tuple3_uint16 *x, const gb_tuple3_uint16 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_min3_uint32 (gb_tuple3_uint32 *z, const gb_tuple3_uint32 *x, const gb_tuple3_uint32 *y) ;
    void gb_min3_uint32 (gb_tuple3_uint32 *z, const gb_tuple3_uint32 *x, const gb_tuple3_uint32 *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_UINT32 \
   "void gb_min3_uint32 (gb_tuple3_uint32 *z, const gb_tuple3_uint32 *x, const gb_tuple3_uint32 *y) ;   \n" \
   "void gb_min3_uint32 (gb_tuple3_uint32 *z, const gb_tuple3_uint32 *x, const gb_tuple3_uint32 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_min3_uint64 (gb_tuple3_uint64 *z, const gb_tuple3_uint64 *x, const gb_tuple3_uint64 *y) ;
    void gb_min3_uint64 (gb_tuple3_uint64 *z, const gb_tuple3_uint64 *x, const gb_tuple3_uint64 *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_UINT64 \
   "void gb_min3_uint64 (gb_tuple3_uint64 *z, const gb_tuple3_uint64 *x, const gb_tuple3_uint64 *y) ;   \n" \
   "void gb_min3_uint64 (gb_tuple3_uint64 *z, const gb_tuple3_uint64 *x, const gb_tuple3_uint64 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_min3_fp32 (gb_tuple3_fp32 *z, const gb_tuple3_fp32 *x, const gb_tuple3_fp32 *y) ;
    void gb_min3_fp32 (gb_tuple3_fp32 *z, const gb_tuple3_fp32 *x, const gb_tuple3_fp32 *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_FP32 \
   "void gb_min3_fp32 (gb_tuple3_fp32 *z, const gb_tuple3_fp32 *x, const gb_tuple3_fp32 *y) ;   \n" \
   "void gb_min3_fp32 (gb_tuple3_fp32 *z, const gb_tuple3_fp32 *x, const gb_tuple3_fp32 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

    void gb_min3_fp64 (gb_tuple3_fp64 *z, const gb_tuple3_fp64 *x, const gb_tuple3_fp64 *y) ;
    void gb_min3_fp64 (gb_tuple3_fp64 *z, const gb_tuple3_fp64 *x, const gb_tuple3_fp64 *y)
    {
        if (x->v < y->v || (x->v == y->v &&
           (x->i < y->i || (x->i == y->i && x->j < y->j))))
        {
            z->i = x->i ;
            z->j = x->j ;
            z->v = x->v ;
        }
        else
        {
            z->i = y->i ;
            z->j = y->j ;
            z->v = y->v ;
        }
    }

#define MIN3_FP64 \
   "void gb_min3_fp64 (gb_tuple3_fp64 *z, const gb_tuple3_fp64 *x, const gb_tuple3_fp64 *y) ;   \n" \
   "void gb_min3_fp64 (gb_tuple3_fp64 *z, const gb_tuple3_fp64 *x, const gb_tuple3_fp64 *y)     \n" \
   "{                                                           \n" \
   "    if (x->v < y->v || (x->v == y->v &&                     \n" \
   "       (x->i < y->i || (x->i == y->i && x->j < y->j))))     \n" \
   "    {                                                       \n" \
   "        z->i = x->i ;                                       \n" \
   "        z->j = x->j ;                                       \n" \
   "        z->v = x->v ;                                       \n" \
   "    }                                                       \n" \
   "    else                                                    \n" \
   "    {                                                       \n" \
   "        z->i = y->i ;                                       \n" \
   "        z->j = y->j ;                                       \n" \
   "        z->v = y->v ;                                       \n" \
   "    }                                                       \n" \
   "}                                                           \n"

//------------------------------------------------------------------------------
// gb_getv_* functions:
//------------------------------------------------------------------------------

// v = getv (tuple) extracts the value v from a 2-tuple.

void gb_getv_bool   (bool     *z, const gb_tuple_bool   *x) ;
void gb_getv_int8   (int8_t   *z, const gb_tuple_int8   *x) ;
void gb_getv_int16  (int16_t  *z, const gb_tuple_int16  *x) ;
void gb_getv_int32  (int32_t  *z, const gb_tuple_int32  *x) ;
void gb_getv_int64  (int64_t  *z, const gb_tuple_int64  *x) ;
void gb_getv_uint8  (uint8_t  *z, const gb_tuple_uint8  *x) ;
void gb_getv_uint16 (uint16_t *z, const gb_tuple_uint16 *x) ;
void gb_getv_uint32 (uint32_t *z, const gb_tuple_uint32 *x) ;
void gb_getv_uint64 (uint64_t *z, const gb_tuple_uint64 *x) ;
void gb_getv_fp32   (float    *z, const gb_tuple_fp32   *x) ;
void gb_getv_fp64   (double   *z, const gb_tuple_fp64   *x) ;

void gb_getv_bool   (bool     *z, const gb_tuple_bool   *x) { (*z) = x->v ; }
void gb_getv_int8   (int8_t   *z, const gb_tuple_int8   *x) { (*z) = x->v ; }
void gb_getv_int16  (int16_t  *z, const gb_tuple_int16  *x) { (*z) = x->v ; }
void gb_getv_int32  (int32_t  *z, const gb_tuple_int32  *x) { (*z) = x->v ; }
void gb_getv_int64  (int64_t  *z, const gb_tuple_int64  *x) { (*z) = x->v ; }
void gb_getv_uint8  (uint8_t  *z, const gb_tuple_uint8  *x) { (*z) = x->v ; }
void gb_getv_uint16 (uint16_t *z, const gb_tuple_uint16 *x) { (*z) = x->v ; }
void gb_getv_uint32 (uint32_t *z, const gb_tuple_uint32 *x) { (*z) = x->v ; }
void gb_getv_uint64 (uint64_t *z, const gb_tuple_uint64 *x) { (*z) = x->v ; }
void gb_getv_fp32   (float    *z, const gb_tuple_fp32   *x) { (*z) = x->v ; }
void gb_getv_fp64   (double   *z, const gb_tuple_fp64   *x) { (*z) = x->v ; }

#define GETV_BOOL   "void gb_getv_bool   (bool     *z, const gb_tuple_bool   *x) { (*z) = x->v ; }"
#define GETV_INT8   "void gb_getv_int8   (int8_t   *z, const gb_tuple_int8   *x) { (*z) = x->v ; }"
#define GETV_INT16  "void gb_getv_int16  (int16_t  *z, const gb_tuple_int16  *x) { (*z) = x->v ; }"
#define GETV_INT32  "void gb_getv_int32  (int32_t  *z, const gb_tuple_int32  *x) { (*z) = x->v ; }"
#define GETV_INT64  "void gb_getv_int64  (int64_t  *z, const gb_tuple_int64  *x) { (*z) = x->v ; }"
#define GETV_UINT8  "void gb_getv_uint8  (uint8_t  *z, const gb_tuple_uint8  *x) { (*z) = x->v ; }"
#define GETV_UINT16 "void gb_getv_uint16 (uint16_t *z, const gb_tuple_uint16 *x) { (*z) = x->v ; }"
#define GETV_UINT32 "void gb_getv_uint32 (uint32_t *z, const gb_tuple_uint32 *x) { (*z) = x->v ; }"
#define GETV_UINT64 "void gb_getv_uint64 (uint64_t *z, const gb_tuple_uint64 *x) { (*z) = x->v ; }"
#define GETV_FP32   "void gb_getv_fp32   (float    *z, const gb_tuple_fp32   *x) { (*z) = x->v ; }"
#define GETV_FP64   "void gb_getv_fp64   (double   *z, const gb_tuple_fp64   *x) { (*z) = x->v ; }"

//------------------------------------------------------------------------------
// gb_getk_* functions:
//------------------------------------------------------------------------------

// k = getk (tuple) extracts the index k from a 2-tuple.

void gb_getk_bool   (int64_t *z, const gb_tuple_bool   *x) ;
void gb_getk_int8   (int64_t *z, const gb_tuple_int8   *x) ;
void gb_getk_int16  (int64_t *z, const gb_tuple_int16  *x) ;
void gb_getk_int32  (int64_t *z, const gb_tuple_int32  *x) ;
void gb_getk_int64  (int64_t *z, const gb_tuple_int64  *x) ;
void gb_getk_uint8  (int64_t *z, const gb_tuple_uint8  *x) ;
void gb_getk_uint16 (int64_t *z, const gb_tuple_uint16 *x) ;
void gb_getk_uint32 (int64_t *z, const gb_tuple_uint32 *x) ;
void gb_getk_uint64 (int64_t *z, const gb_tuple_uint64 *x) ;
void gb_getk_fp32   (int64_t *z, const gb_tuple_fp32   *x) ;
void gb_getk_fp64   (int64_t *z, const gb_tuple_fp64   *x) ;

void gb_getk_bool   (int64_t *z, const gb_tuple_bool   *x) { (*z) = x->k ; }
void gb_getk_int8   (int64_t *z, const gb_tuple_int8   *x) { (*z) = x->k ; }
void gb_getk_int16  (int64_t *z, const gb_tuple_int16  *x) { (*z) = x->k ; }
void gb_getk_int32  (int64_t *z, const gb_tuple_int32  *x) { (*z) = x->k ; }
void gb_getk_int64  (int64_t *z, const gb_tuple_int64  *x) { (*z) = x->k ; }
void gb_getk_uint8  (int64_t *z, const gb_tuple_uint8  *x) { (*z) = x->k ; }
void gb_getk_uint16 (int64_t *z, const gb_tuple_uint16 *x) { (*z) = x->k ; }
void gb_getk_uint32 (int64_t *z, const gb_tuple_uint32 *x) { (*z) = x->k ; }
void gb_getk_uint64 (int64_t *z, const gb_tuple_uint64 *x) { (*z) = x->k ; }
void gb_getk_fp32   (int64_t *z, const gb_tuple_fp32   *x) { (*z) = x->k ; }
void gb_getk_fp64   (int64_t *z, const gb_tuple_fp64   *x) { (*z) = x->k ; }

#define GETK_BOOL   "void gb_getk_bool   (int64_t *z, const gb_tuple_bool   *x) { (*z) = x->k ; }"
#define GETK_INT8   "void gb_getk_int8   (int64_t *z, const gb_tuple_int8   *x) { (*z) = x->k ; }"
#define GETK_INT16  "void gb_getk_int16  (int64_t *z, const gb_tuple_int16  *x) { (*z) = x->k ; }"
#define GETK_INT32  "void gb_getk_int32  (int64_t *z, const gb_tuple_int32  *x) { (*z) = x->k ; }"
#define GETK_INT64  "void gb_getk_int64  (int64_t *z, const gb_tuple_int64  *x) { (*z) = x->k ; }"
#define GETK_UINT8  "void gb_getk_uint8  (int64_t *z, const gb_tuple_uint8  *x) { (*z) = x->k ; }"
#define GETK_UINT16 "void gb_getk_uint16 (int64_t *z, const gb_tuple_uint16 *x) { (*z) = x->k ; }"
#define GETK_UINT32 "void gb_getk_uint32 (int64_t *z, const gb_tuple_uint32 *x) { (*z) = x->k ; }"
#define GETK_UINT64 "void gb_getk_uint64 (int64_t *z, const gb_tuple_uint64 *x) { (*z) = x->k ; }"
#define GETK_FP32   "void gb_getk_fp32   (int64_t *z, const gb_tuple_fp32   *x) { (*z) = x->k ; }"
#define GETK_FP64   "void gb_getk_fp64   (int64_t *z, const gb_tuple_fp64   *x) { (*z) = x->k ; }"

//------------------------------------------------------------------------------
// gbargminmax: mexFunction to compute the argmin/max of each row/column of A
//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin == 3 && nargout == 2, USAGE) ;
    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    bool is_min = (bool) (mxGetScalar (pargin [1]) == 0) ;
    int dim = (int) mxGetScalar (pargin [2]) ;
    CHECK_ERROR (dim < 0 || dim > 2, "invalid dim") ;

    //--------------------------------------------------------------------------
    // get the matrix properties
    //--------------------------------------------------------------------------

    uint64_t nrows, ncols, nvals ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    GrB_Type A_type ;
    OK (GxB_Matrix_type (&A_type, A)) ;
    int fmt ;
    OK (GrB_Matrix_get_INT32 (A, &fmt, GxB_FORMAT)) ;

    //--------------------------------------------------------------------------
    // types, ops, and semirings for argmin and argmax
    //--------------------------------------------------------------------------

    GrB_Type Tuple = NULL, Tuple3 = NULL ;
    GxB_IndexBinaryOp Iop = NULL ;
    GrB_IndexUnaryOp Make3 = NULL ;
    GrB_BinaryOp Bop = NULL, MonOp = NULL, Mon3Op = NULL ;
    GrB_Monoid Monoid = NULL, Monoid3 = NULL ;
    GrB_Semiring Semiring = NULL ;
    GrB_Scalar Theta = NULL ;
    GrB_UnaryOp Getv = NULL, Getk = NULL ;
    GrB_Matrix x = NULL, p = NULL, c = NULL, y = NULL, z = NULL ;
    GrB_Scalar s = NULL ;

    OK (GrB_Scalar_new (&Theta, GrB_BOOL)) ;
    OK (GrB_Scalar_setElement_BOOL (Theta, 0)) ;

    if (A_type == GrB_BOOL)
    {

        //----------------------------------------------------------------------
        // boolean
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_bool), "gb_tuple_bool", BOOL_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_bool,
            Tuple, GrB_BOOL, GrB_BOOL, GrB_BOOL, "gb_make_bool", MAKE_BOOL)) ;
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_bool id ;
        memset (&id, 0, sizeof (gb_tuple_bool)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = true ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_bool,
                Tuple, Tuple, Tuple, "gb_min_bool", MIN_BOOL)) ;
        }
        else
        {
            id.v = false ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_bool,
                Tuple, Tuple, Tuple, "gb_max_bool", MAX_BOOL)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_bool),
                "gb_tuple3_bool", BOOL_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_bool,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_bool", MAKE3a_BOOL)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_bool,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_bool", MAKE3b_BOOL)) ;
            }
            gb_tuple3_bool id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_bool)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = true ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_min3_bool,
                    Tuple3, Tuple3, Tuple3, "gb_min3_bool", MIN3_BOOL)) ;
            }
            else
            {
                id3.v = false ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_max3_bool,
                    Tuple3, Tuple3, Tuple3, "gb_max3_bool", MAX3_BOOL)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_bool,
                GrB_INT64, Tuple, "gb_getk_bool", GETK_BOOL)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_bool,
                GrB_BOOL, Tuple, "gb_getv_bool", GETV_BOOL)) ;
        }

    }
    else if (A_type == GrB_INT8)
    {

        //----------------------------------------------------------------------
        // int8
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_int8), "gb_tuple_int8", INT8_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_int8,
            Tuple, GrB_INT8, GrB_BOOL, GrB_BOOL, "gb_make_int8", MAKE_INT8)) ;
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_int8 id ;
        memset (&id, 0, sizeof (gb_tuple_int8)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = INT8_MAX ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_int8,
                Tuple, Tuple, Tuple, "gb_min_int8", MIN_INT8)) ;
        }
        else
        {
            id.v = INT8_MIN ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_int8,
                Tuple, Tuple, Tuple, "gb_max_int8", MAX_INT8)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_int8),
                "gb_tuple3_int8", INT8_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_int8,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_int8", MAKE3a_INT8)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_int8,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_int8", MAKE3b_INT8)) ;
            }
            gb_tuple3_int8 id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_int8)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = INT8_MAX ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_min3_int8,
                    Tuple3, Tuple3, Tuple3, "gb_min3_int8", MIN3_INT8)) ;
            }
            else
            {
                id3.v = INT8_MIN ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_max3_int8,
                    Tuple3, Tuple3, Tuple3, "gb_max3_int8", MAX3_INT8)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_int8,
                GrB_INT64, Tuple, "gb_getk_int8", GETK_INT8)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_int8,
                GrB_INT8, Tuple, "gb_getv_int8", GETV_INT8)) ;
        }

    }
    else if (A_type == GrB_INT16)
    {

        //----------------------------------------------------------------------
        // int16
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_int16),
            "gb_tuple_int16", INT16_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_int16,
            Tuple, GrB_INT16, GrB_BOOL, GrB_BOOL, "gb_make_int16", MAKE_INT16)) ;
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_int16 id ;
        memset (&id, 0, sizeof (gb_tuple_int16)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = INT16_MAX ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_int16,
                Tuple, Tuple, Tuple, "gb_min_int16", MIN_INT16)) ;
        }
        else
        {
            id.v = INT16_MIN ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_int16,
                Tuple, Tuple, Tuple, "gb_max_int16", MAX_INT16)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_int16),
                "gb_tuple3_int16", INT16_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_int16,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_int16", MAKE3a_INT16)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_int16,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_int16", MAKE3b_INT16)) ;
            }
            gb_tuple3_int16 id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_int16)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = INT16_MAX ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_min3_int16,
                    Tuple3, Tuple3, Tuple3, "gb_min3_int16", MIN3_INT16)) ;
            }
            else
            {
                id3.v = INT16_MIN ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_max3_int16,
                    Tuple3, Tuple3, Tuple3, "gb_max3_int16", MAX3_INT16)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_int16,
                GrB_INT64, Tuple, "gb_getk_int16", GETK_INT16)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_int16,
                GrB_INT16, Tuple, "gb_getv_int16", GETV_INT16)) ;
        }

    }
    else if (A_type == GrB_INT32)
    {

        //----------------------------------------------------------------------
        // int32
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_int32),
            "gb_tuple_int32", INT32_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_int32,
            Tuple, GrB_INT32, GrB_BOOL, GrB_BOOL, "gb_make_int32", MAKE_INT32)) ;
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_int32 id ;
        memset (&id, 0, sizeof (gb_tuple_int32)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = INT32_MAX ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_int32,
                Tuple, Tuple, Tuple, "gb_min_int32", MIN_INT32)) ;
        }
        else
        {
            id.v = INT32_MIN ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_int32,
                Tuple, Tuple, Tuple, "gb_max_int32", MAX_INT32)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_int32),
                "gb_tuple3_int32", INT32_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_int32,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_int32", MAKE3a_INT32)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_int32,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_int32", MAKE3b_INT32)) ;
            }
            gb_tuple3_int32 id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_int32)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = INT32_MAX ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_min3_int32,
                    Tuple3, Tuple3, Tuple3, "gb_min3_int32", MIN3_INT32)) ;
            }
            else
            {
                id3.v = INT32_MIN ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_max3_int32,
                    Tuple3, Tuple3, Tuple3, "gb_max3_int32", MAX3_INT32)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_int32,
                GrB_INT64, Tuple, "gb_getk_int32", GETK_INT32)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_int32,
                GrB_INT32, Tuple, "gb_getv_int32", GETV_INT32)) ;
        }

    }
    else if (A_type == GrB_INT64)
    {

        //----------------------------------------------------------------------
        // int64
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_int64),
            "gb_tuple_int64", INT64_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_int64,
            Tuple, GrB_INT64, GrB_BOOL, GrB_BOOL, "gb_make_int64", MAKE_INT64)) ;
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_int64 id ;
        memset (&id, 0, sizeof (gb_tuple_int64)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = INT64_MAX ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_int64,
                Tuple, Tuple, Tuple, "gb_min_int64", MIN_INT64)) ;
        }
        else
        {
            id.v = INT64_MIN ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_int64,
                Tuple, Tuple, Tuple, "gb_max_int64", MAX_INT64)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_int64),
                "gb_tuple3_int64", INT64_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_int64,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_int64", MAKE3a_INT64)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_int64,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_int64", MAKE3b_INT64)) ;
            }
            gb_tuple3_int64 id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_int64)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = INT64_MAX ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_min3_int64,
                    Tuple3, Tuple3, Tuple3, "gb_min3_int64", MIN3_INT64)) ;
            }
            else
            {
                id3.v = INT64_MIN ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_max3_int64,
                    Tuple3, Tuple3, Tuple3, "gb_max3_int64", MAX3_INT64)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_int64,
                GrB_INT64, Tuple, "gb_getk_int64", GETK_INT64)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_int64,
                GrB_INT64, Tuple, "gb_getv_int64", GETV_INT64)) ;
        }

    }
    else if (A_type == GrB_UINT8)
    {

        //----------------------------------------------------------------------
        // uint8
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_uint8),
            "gb_tuple_uint8", UINT8_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_uint8,
            Tuple, GrB_UINT8, GrB_BOOL, GrB_BOOL, "gb_make_uint8", MAKE_UINT8)) ;
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_uint8 id ;
        memset (&id, 0, sizeof (gb_tuple_uint8)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = UINT8_MAX ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_uint8,
                Tuple, Tuple, Tuple, "gb_min_uint8", MIN_UINT8)) ;
        }
        else
        {
            id.v = 0 ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_uint8,
                Tuple, Tuple, Tuple, "gb_max_uint8", MAX_UINT8)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_uint8),
                "gb_tuple3_uint8", UINT8_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_uint8,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_uint8", MAKE3a_UINT8)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_uint8,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_uint8", MAKE3b_UINT8)) ;
            }
            gb_tuple3_uint8 id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_uint8)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = UINT8_MAX ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_min3_uint8,
                    Tuple3, Tuple3, Tuple3, "gb_min3_uint8", MIN3_UINT8)) ;
            }
            else
            {
                id3.v = 0 ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_max3_uint8,
                    Tuple3, Tuple3, Tuple3, "gb_max3_uint8", MAX3_UINT8)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_uint8,
                GrB_INT64, Tuple, "gb_getk_uint8", GETK_UINT8)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_uint8,
                GrB_UINT8, Tuple, "gb_getv_uint8", GETV_UINT8)) ;
        }

    }
    else if (A_type == GrB_UINT16)
    {

        //----------------------------------------------------------------------
        // uint16
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_uint16),
            "gb_tuple_uint16", UINT16_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_uint16,
            Tuple, GrB_UINT16, GrB_BOOL, GrB_BOOL, "gb_make_uint16", MAKE_UINT16));
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_uint16 id ;
        memset (&id, 0, sizeof (gb_tuple_uint16)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = UINT16_MAX ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_uint16,
                Tuple, Tuple, Tuple, "gb_min_uint16", MIN_UINT16)) ;
        }
        else
        {
            id.v = 0 ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_uint16,
                Tuple, Tuple, Tuple, "gb_max_uint16", MAX_UINT16)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_uint16),
                "gb_tuple3_uint16", UINT16_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_uint16,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_uint16", MAKE3a_UINT16)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_uint16,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_uint16", MAKE3b_UINT16)) ;
            }
            gb_tuple3_uint16 id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_uint16)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = UINT16_MAX ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function)gb_min3_uint16,
                    Tuple3, Tuple3, Tuple3, "gb_min3_uint16", MIN3_UINT16)) ;
            }
            else
            {
                id3.v = 0 ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function)gb_max3_uint16,
                    Tuple3, Tuple3, Tuple3, "gb_max3_uint16", MAX3_UINT16)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_uint16,
                GrB_INT64, Tuple, "gb_getk_uint16", GETK_UINT16)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_uint16,
                GrB_UINT16, Tuple, "gb_getv_uint16", GETV_UINT16)) ;
        }

    }
    else if (A_type == GrB_UINT32)
    {

        //----------------------------------------------------------------------
        // uint32
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_uint32),
            "gb_tuple_uint32", UINT32_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_uint32,
            Tuple, GrB_UINT32, GrB_BOOL, GrB_BOOL, "gb_make_uint32", MAKE_UINT32));
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_uint32 id ;
        memset (&id, 0, sizeof (gb_tuple_uint32)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = UINT32_MAX ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_uint32,
                Tuple, Tuple, Tuple, "gb_min_uint32", MIN_UINT32)) ;
        }
        else
        {
            id.v = 0 ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_uint32,
                Tuple, Tuple, Tuple, "gb_max_uint32", MAX_UINT32)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_uint32),
                "gb_tuple3_uint32", UINT32_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_uint32,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_uint32", MAKE3a_UINT32)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_uint32,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_uint32", MAKE3b_UINT32)) ;
            }
            gb_tuple3_uint32 id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_uint32)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = UINT32_MAX ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function)gb_min3_uint32,
                    Tuple3, Tuple3, Tuple3, "gb_min3_uint32", MIN3_UINT32)) ;
            }
            else
            {
                id3.v = 0 ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function)gb_max3_uint32,
                    Tuple3, Tuple3, Tuple3, "gb_max3_uint32", MAX3_UINT32)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_uint32,
                GrB_INT64, Tuple, "gb_getk_uint32", GETK_UINT32)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_uint32,
                GrB_UINT32, Tuple, "gb_getv_uint32", GETV_UINT32)) ;
        }

    }
    else if (A_type == GrB_UINT64)
    {

        //----------------------------------------------------------------------
        // uint64
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_uint64),
            "gb_tuple_uint64", UINT64_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_uint64,
            Tuple, GrB_UINT64, GrB_BOOL, GrB_BOOL, "gb_make_uint64", MAKE_UINT64));
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_uint64 id ;
        memset (&id, 0, sizeof (gb_tuple_uint64)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = UINT64_MAX ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_uint64,
                Tuple, Tuple, Tuple, "gb_min_uint64", MIN_UINT64)) ;
        }
        else
        {
            id.v = 0 ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_uint64,
                Tuple, Tuple, Tuple, "gb_max_uint64", MAX_UINT64)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_uint64),
                "gb_tuple3_uint64", UINT64_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_uint64,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_uint64", MAKE3a_UINT64)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_uint64,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_uint64", MAKE3b_UINT64)) ;
            }
            gb_tuple3_uint64 id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_uint64)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = UINT64_MAX ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function)gb_min3_uint64,
                    Tuple3, Tuple3, Tuple3, "gb_min3_uint64", MIN3_UINT64)) ;
            }
            else
            {
                id3.v = 0 ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function)gb_max3_uint64,
                    Tuple3, Tuple3, Tuple3, "gb_max3_uint64", MAX3_UINT64)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_uint64,
                GrB_INT64, Tuple, "gb_getk_uint64", GETK_UINT64)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_uint64,
                GrB_UINT64, Tuple, "gb_getv_uint64", GETV_UINT64)) ;
        }

    }
    else if (A_type == GrB_FP32)
    {

        //----------------------------------------------------------------------
        // fp32
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_fp32), "gb_tuple_fp32", FP32_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_fp32,
            Tuple, GrB_FP32, GrB_BOOL, GrB_BOOL, "gb_make_fp32", MAKE_FP32)) ;
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_fp32 id ;
        memset (&id, 0, sizeof (gb_tuple_fp32)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = (float) INFINITY ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_fp32,
                Tuple, Tuple, Tuple, "gb_min_fp32", MIN_FP32)) ;
        }
        else
        {
            id.v = (float) (-INFINITY) ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_fp32,
                Tuple, Tuple, Tuple, "gb_max_fp32", MAX_FP32)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_fp32),
                "gb_tuple3_fp32", FP32_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_fp32,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_fp32", MAKE3a_FP32)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_fp32,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_fp32", MAKE3b_FP32)) ;
            }
            gb_tuple3_fp32 id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_fp32)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = (float) INFINITY ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_min3_fp32,
                    Tuple3, Tuple3, Tuple3, "gb_min3_fp32", MIN3_FP32)) ;
            }
            else
            {
                id3.v = (float) (-INFINITY) ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_max3_fp32,
                    Tuple3, Tuple3, Tuple3, "gb_max3_fp32", MAX3_FP32)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_fp32,
                GrB_INT64, Tuple, "gb_getk_fp32", GETK_FP32)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_fp32,
                GrB_FP32, Tuple, "gb_getv_fp32", GETV_FP32)) ;
        }

    }
    else if (A_type == GrB_FP64)
    {

        //----------------------------------------------------------------------
        // fp64
        //----------------------------------------------------------------------

        OK (GxB_Type_new (&Tuple, sizeof (gb_tuple_fp64), "gb_tuple_fp64", FP64_K)) ;
        OK (GxB_IndexBinaryOp_new (&Iop,
            (GxB_index_binary_function) gb_make_fp64,
            Tuple, GrB_FP64, GrB_BOOL, GrB_BOOL, "gb_make_fp64", MAKE_FP64)) ;
        OK (GxB_BinaryOp_new_IndexOp (&Bop, Iop, Theta)) ;
        gb_tuple_fp64 id ;
        memset (&id, 0, sizeof (gb_tuple_fp64)) ;
        id.k = INT64_MAX ;
        if (is_min)
        {
            id.v = (double) INFINITY ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_min_fp64,
                Tuple, Tuple, Tuple, "gb_min_fp64", MIN_FP64)) ;
        }
        else
        {
            id.v = (double) (-INFINITY) ;
            OK (GxB_BinaryOp_new (&MonOp, (GxB_binary_function) gb_max_fp64,
                Tuple, Tuple, Tuple, "gb_max_fp64", MAX_FP64)) ;
        }
        OK (GrB_Monoid_new_UDT (&Monoid, MonOp, &id)) ;
        OK (GrB_Semiring_new (&Semiring, Monoid, Bop)) ;
        if (dim == 0)
        {
            OK (GxB_Type_new (&Tuple3, sizeof (gb_tuple3_fp64),
                "gb_tuple3_fp64", FP64_IJ)) ;
            if (fmt == GxB_BY_ROW)
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3a_fp64,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3a_fp64", MAKE3a_FP64)) ;
            }
            else
            {
                OK (GxB_IndexUnaryOp_new (&Make3,
                    (GxB_index_unary_function) gb_make3b_fp64,
                    Tuple3, Tuple, GrB_BOOL, "gb_make3b_fp64", MAKE3b_FP64)) ;
            }
            gb_tuple3_fp64 id3 ;
            memset (&id3, 0, sizeof (gb_tuple3_fp64)) ;
            id3.i = INT64_MAX ;
            id3.j = INT64_MAX ;
            if (is_min)
            {
                id3.v = (double) INFINITY ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_min3_fp64,
                    Tuple3, Tuple3, Tuple3, "gb_min3_fp64", MIN3_FP64)) ;
            }
            else
            {
                id3.v = (double) (-INFINITY) ;
                OK (GxB_BinaryOp_new (&Mon3Op, (GxB_binary_function) gb_max3_fp64,
                    Tuple3, Tuple3, Tuple3, "gb_max3_fp64", MAX3_FP64)) ;
            }
            OK (GrB_Monoid_new_UDT (&Monoid3, Mon3Op, &id3)) ;
        }
        else
        {
            OK (GxB_UnaryOp_new (&Getk, (GxB_unary_function) gb_getk_fp64,
                GrB_INT64, Tuple, "gb_getk_fp64", GETK_FP64)) ;
            OK (GxB_UnaryOp_new (&Getv, (GxB_unary_function) gb_getv_fp64,
                GrB_FP64, Tuple, "gb_getv_fp64", GETV_FP64)) ;
        }

    }
    else
    {
        ERROR ("unsupported type") ;
    }

    //--------------------------------------------------------------------------
    // compute the argmin/max
    //--------------------------------------------------------------------------

    if (dim == 0)
    {

        //----------------------------------------------------------------------
        // scalar argmin/max of all of A
        //----------------------------------------------------------------------

        if (fmt == GxB_BY_ROW)
        {
            // A is held by row
            // y = zeros (ncols,1) ;
            OK (GrB_Matrix_new (&y, GrB_BOOL, ncols, 1)) ;
            OK (GrB_Matrix_assign_BOOL (y, NULL, NULL, 0,
                GrB_ALL, ncols, GrB_ALL, 1, NULL)) ;

            // c = A*y using the argmin/argmax semiring
            OK (GrB_Matrix_new (&c, Tuple, nrows, 1)) ;
            OK (GrB_mxm (c, NULL, NULL, Semiring, A, y, NULL)) ;

            // create z
            OK (GrB_Matrix_new (&z, Tuple3, nrows, 1)) ;
        }
        else
        {
            // A is held by column (the default for MATLAB)
            // y = zeros (nrows,1) ;
            OK (GrB_Matrix_new (&y, GrB_BOOL, nrows, 1)) ;
            OK (GrB_Matrix_assign_BOOL (y, NULL, NULL, 0,
                GrB_ALL, nrows, GrB_ALL, 1, NULL)) ;

            // c = A'*y using the argmin/argmax semiring
            OK (GrB_Matrix_new (&c, Tuple, ncols, 1)) ;
            OK (GrB_mxm (c, NULL, NULL, Semiring, A, y, GrB_DESC_T0)) ;

            // create z
            OK (GrB_Matrix_new (&z, Tuple3, ncols, 1)) ;
        }

        // z = make3 (c)
        OK (GrB_Matrix_apply_IndexOp_BOOL (z, NULL, NULL, Make3, c, 0, NULL)) ;

        // s = max3 (z)
        OK (GrB_Scalar_new (&s, Tuple3)) ;
        OK (GrB_Matrix_reduce_Monoid_Scalar (s, NULL, Monoid3, z, NULL)) ;

        // x = s.v, p = {s.i, s.j}
        OK (GrB_Matrix_new (&x, A_type, 1, 1)) ;
        OK (GrB_Matrix_new (&p, GrB_INT64, 2, 1)) ;
        OK (GrB_Scalar_nvals (&nvals, s)) ;
        int64_t si = INT64_MAX, sj = INT64_MAX ;
        if (nvals > 0)
        {
            if (A_type == GrB_BOOL)
            {
                gb_tuple3_bool result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_BOOL (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            else if (A_type == GrB_INT8)
            {
                gb_tuple3_int8 result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_INT8 (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            else if (A_type == GrB_INT16)
            {
                gb_tuple3_int16 result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_INT16 (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            else if (A_type == GrB_INT32)
            {
                gb_tuple3_int32 result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_INT32 (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            else if (A_type == GrB_INT64)
            {
                gb_tuple3_int64 result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_INT64 (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            else if (A_type == GrB_UINT8)
            {
                gb_tuple3_uint8 result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_UINT8 (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            else if (A_type == GrB_UINT16)
            {
                gb_tuple3_uint16 result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_UINT16 (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            else if (A_type == GrB_UINT32)
            {
                gb_tuple3_uint32 result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_UINT32 (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            else if (A_type == GrB_UINT64)
            {
                gb_tuple3_uint64 result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_UINT64 (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            else if (A_type == GrB_FP32)
            {
                gb_tuple3_fp32 result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_FP32 (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            else // if (A_type == GrB_FP64)
            {
                gb_tuple3_fp64 result ;
                OK (GrB_Scalar_extractElement_UDT (&result, s)) ;
                OK (GrB_Matrix_setElement_FP64 (x, result.v, 0, 0)) ;
                si = result.i ;
                sj = result.j ;
            }
            OK (GrB_Matrix_setElement_INT64 (p, si, 0, 0)) ;
            OK (GrB_Matrix_setElement_INT64 (p, sj, 1, 0)) ;
        }

    }
    else
    {

        if (dim == 1)
        {

            //------------------------------------------------------------------
            // argmin/max of each column of A
            //------------------------------------------------------------------

            // y = zeros (nrows,1) ;
            OK (GrB_Matrix_new (&y, GrB_BOOL, nrows, 1)) ;
            OK (GrB_Matrix_assign_BOOL (y, NULL, NULL, 0,
                GrB_ALL, nrows, GrB_ALL, 1, NULL)) ;

            // c = A'*y using the argmin/argmax semiring
            OK (GrB_Matrix_new (&c, Tuple, ncols, 1)) ;
            OK (GrB_mxm (c, NULL, NULL, Semiring, A, y, GrB_DESC_T0)) ;

            // create x and p
            OK (GrB_Matrix_new (&x, A_type, ncols, 1)) ;
            OK (GrB_Matrix_new (&p, GrB_INT64, ncols, 1)) ;

        }
        else
        {

            //------------------------------------------------------------------
            // argmin/max of each row of A
            //------------------------------------------------------------------

            // y = zeros (ncols,1) ;
            OK (GrB_Matrix_new (&y, GrB_BOOL, ncols, 1)) ;
            OK (GrB_Matrix_assign_BOOL (y, NULL, NULL, 0,
                GrB_ALL, ncols, GrB_ALL, 1, NULL)) ;

            // c = A*y using the argmin/argmax semiring
            OK (GrB_Matrix_new (&c, Tuple, nrows, 1)) ;
            OK (GrB_mxm (c, NULL, NULL, Semiring, A, y, NULL)) ;

            // create x and p
            OK (GrB_Matrix_new (&x, A_type, nrows, 1)) ;
            OK (GrB_Matrix_new (&p, GrB_INT64, nrows, 1)) ;
        }

        // x = getv (c)
        OK (GrB_Matrix_apply (x, NULL, NULL, Getv, c, NULL)) ;
        // p = getk (c)
        OK (GrB_Matrix_apply (p, NULL, NULL, Getk, c, NULL)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    OK (GrB_Type_free (&Tuple)) ;
    OK (GrB_Type_free (&Tuple3)) ;
    OK (GxB_IndexBinaryOp_free (&Iop)) ;
    OK (GrB_IndexUnaryOp_free (&Make3)) ;
    OK (GrB_BinaryOp_free (&Bop)) ;
    OK (GrB_BinaryOp_free (&MonOp)) ;
    OK (GrB_BinaryOp_free (&Mon3Op)) ;
    OK (GrB_Monoid_free (&Monoid)) ;
    OK (GrB_Monoid_free (&Monoid3)) ;
    OK (GrB_Semiring_free (&Semiring)) ;
    OK (GrB_UnaryOp_free (&Getv)) ;
    OK (GrB_UnaryOp_free (&Getk)) ;
    OK (GrB_Matrix_free (&y)) ;
    OK (GrB_Matrix_free (&c)) ;
    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&z)) ;
    OK (GrB_Scalar_free (&Theta)) ;
    OK (GrB_Scalar_free (&s)) ;

    pargout [0] = gb_export (&x, KIND_GRB) ;
    pargout [1] = gb_export (&p, KIND_GRB) ;
    gb_wrapup ( ) ;
}

