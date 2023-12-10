//------------------------------------------------------------------------------
// GB_casting.h: define the unary typecasting functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GJ_cast* methods are only used in JIT kernels.

#ifndef GB_CASTING_H
#define GB_CASTING_H

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
// typecasting from double to integer
//------------------------------------------------------------------------------

// The GraphBLAS C API states that typecasting follows the rules of the C
// language.  However, the C11 language specification states that results
// are undefined when typecasting a float or double to an integer value that is
// outside the range of the integer type.  GraphBLAS handles this case by
// typecasting a float or double that is larger than the maximum integer to the
// max integer, and a value less than the minimum integer to the min integer.
// NaN's are typecasted to the integer value zero.

inline int8_t GB_cast_to_int8_t (double x)
{ 
    if (isnan (x)) return (0) ;
    if (x <= (double) INT8_MIN) return (INT8_MIN) ;
    if (x >= (double) INT8_MAX) return (INT8_MAX) ;
    return ((int8_t) x) ;
}

#define GJ_cast_to_int8_DEFN                                \
"int8_t GJ_cast_to_int8 (double x)                      \n" \
"{                                                      \n" \
"    if (isnan (x)) return (0) ;                        \n" \
"    if (x <= (double) INT8_MIN) return (INT8_MIN) ;    \n" \
"    if (x >= (double) INT8_MAX) return (INT8_MAX) ;    \n" \
"    return ((int8_t) x) ;                              \n" \
"}"

inline int16_t GB_cast_to_int16_t (double x)
{
    if (isnan (x)) return (0) ;
    if (x <= (double) INT16_MIN) return (INT16_MIN) ;
    if (x >= (double) INT16_MAX) return (INT16_MAX) ;
    return ((int16_t) x) ;
}

#define  GJ_cast_to_int16_DEFN                              \
"int16_t GJ_cast_to_int16 (double x)                    \n" \
"{                                                      \n" \
"    if (isnan (x)) return (0) ;                        \n" \
"    if (x <= (double) INT16_MIN) return (INT16_MIN) ;  \n" \
"    if (x >= (double) INT16_MAX) return (INT16_MAX) ;  \n" \
"    return ((int16_t) x) ;                             \n" \
"}"

inline int32_t GB_cast_to_int32_t (double x)
{ 
    if (isnan (x)) return (0) ;
    if (x <= (double) INT32_MIN) return (INT32_MIN) ;
    if (x >= (double) INT32_MAX) return (INT32_MAX) ;
    return ((int32_t) x) ;
}

#define  GJ_cast_to_int32_DEFN                              \
"int32_t GJ_cast_to_int32 (double x)                    \n" \
"{                                                      \n" \
"    if (isnan (x)) return (0) ;                        \n" \
"    if (x <= (double) INT32_MIN) return (INT32_MIN) ;  \n" \
"    if (x >= (double) INT32_MAX) return (INT32_MAX) ;  \n" \
"    return ((int32_t) x) ;                             \n" \
"}"

inline int64_t GB_cast_to_int64_t (double x)
{ 
    if (isnan (x)) return (0) ;
    if (x <= (double) INT64_MIN) return (INT64_MIN) ;
    if (x >= (double) INT64_MAX) return (INT64_MAX) ;
    return ((int64_t) x) ;
}

#define  GJ_cast_to_int64_DEFN                              \
"int64_t GJ_cast_to_int64 (double x)                    \n" \
"{                                                      \n" \
"    if (isnan (x)) return (0) ;                        \n" \
"    if (x <= (double) INT64_MIN) return (INT64_MIN) ;  \n" \
"    if (x >= (double) INT64_MAX) return (INT64_MAX) ;  \n" \
"    return ((int64_t) x) ;                             \n" \
"}"

inline uint8_t GB_cast_to_uint8_t (double x)
{ 
    if (isnan (x) || x <= 0) return (0) ;
    if (x >= (double) UINT8_MAX) return (UINT8_MAX) ;
    return ((uint8_t) x) ;
}

#define  GJ_cast_to_uint8_DEFN                              \
"uint8_t GJ_cast_to_uint8 (double x)                    \n" \
"{                                                      \n" \
"    if (isnan (x) || x <= 0) return (0) ;              \n" \
"    if (x >= (double) UINT8_MAX) return (UINT8_MAX) ;  \n" \
"    return ((uint8_t) x) ;                             \n" \
"}"

inline uint16_t GB_cast_to_uint16_t (double x)
{ 
    if (isnan (x) || x <= 0) return (0) ;
    if (x >= (double) UINT16_MAX) return (UINT16_MAX) ;
    return ((uint16_t) x) ;
}

#define   GJ_cast_to_uint16_DEFN                                \
"uint16_t GJ_cast_to_uint16 (double x)                      \n" \
"{                                                          \n" \
"    if (isnan (x) || x <= 0) return (0) ;                  \n" \
"    if (x >= (double) UINT16_MAX) return (UINT16_MAX) ;    \n" \
"    return ((uint16_t) x) ;                                \n" \
"}"

inline uint32_t GB_cast_to_uint32_t (double x)
{ 
    if (isnan (x) || x <= 0) return (0) ;
    if (x >= (double) UINT32_MAX) return (UINT32_MAX) ;
    return ((uint32_t) x) ;
}

#define   GJ_cast_to_uint32_DEFN                                \
"uint32_t GJ_cast_to_uint32 (double x)                      \n" \
"{                                                          \n" \
"    if (isnan (x) || x <= 0) return (0) ;                  \n" \
"    if (x >= (double) UINT32_MAX) return (UINT32_MAX) ;    \n" \
"    return ((uint32_t) x) ;                                \n" \
"}"

inline uint64_t GB_cast_to_uint64_t (double x)
{ 
    if (isnan (x) || x <= 0) return (0) ;
    if (x >= (double) UINT64_MAX) return (UINT64_MAX) ;
    return ((uint64_t) x) ;
}

#define   GJ_cast_to_uint64_DEFN                                \
"uint64_t GJ_cast_to_uint64 (double x)                      \n" \
"{                                                          \n" \
"    if (isnan (x) || x <= 0) return (0) ;                  \n" \
"    if (x >= (double) UINT64_MAX) return (UINT64_MAX) ;    \n" \
"    return ((uint64_t) x) ;                                \n" \
"}"

//------------------------------------------------------------------------------
// unary typecast operators, used in GB_cast_factory.c.
//------------------------------------------------------------------------------

// construct the typecast functions: z = (ztype) x, where x has type xtype.

// GB_cast_ztype_xtype casts a single value ((xtype) x) into the result
// ((ztype) z).  Functions with the xtype and ztype the same, such as
// GB_cast_double_double, simply copy their input to their output.

// The s parameter is not used in these functions.  It is present because one
// function returned by GB_cast_factory requires it (GB_copy_user_user).

void GB_copy_user_user (void *z, const void *x, size_t s) ;

#define GB_CAST_FUNCTION(ztype,xtype)                                   \
inline void GB (_cast_ ## ztype ## _ ## xtype)                          \
(                                                                       \
    void *z,            /* typecasted output, of type ztype */          \
    const void *x,      /* input value to typecast, of type xtype */    \
    size_t s            /* size of type, for GB_copy_user_user only */  \
)                                                                       \
{                                                                       \
    xtype xx = (*((xtype *) x)) ;                                       \
    ztype zz = GB_CAST (ztype, xx) ;                                    \
    (*((ztype *) z)) = zz ;                                             \
}

//------------------------------------------------------------------------------
// typecast to boolean
//------------------------------------------------------------------------------

// Typecasting a NaN to a bool results in 'true', as defined by the C11
// standard (NaN converts to true, since Nan != 0 is true).  GraphBLAS follows
// the C11 standard in this case.

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (bool      , bool      )
#undef  GB_CAST
#define GB_CAST(ztype,x) (x != 0)
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

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#undef  GB_CAST
#define GB_CAST(ztype,x) (GB_crealf (x) != 0 || GB_cimagf (x) != 0)
GB_CAST_FUNCTION (bool      , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) (GB_creal (x) != 0 || GB_cimag (x) != 0)
GB_CAST_FUNCTION (bool      , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to int8_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (int8_t    , bool      )
GB_CAST_FUNCTION (int8_t    , int8_t    )
GB_CAST_FUNCTION (int8_t    , int16_t   )
GB_CAST_FUNCTION (int8_t    , int32_t   )
GB_CAST_FUNCTION (int8_t    , int64_t   )
GB_CAST_FUNCTION (int8_t    , uint8_t   )
GB_CAST_FUNCTION (int8_t    , uint16_t  )
GB_CAST_FUNCTION (int8_t    , uint32_t  )
GB_CAST_FUNCTION (int8_t    , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int8_t ((double) x)
GB_CAST_FUNCTION (int8_t    , float     )
GB_CAST_FUNCTION (int8_t    , double    )
#undef  GB_CAST

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#define GB_CAST(ztype,x) GB_cast_to_int8_t ((double) GB_crealf (x))
GB_CAST_FUNCTION (int8_t    , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int8_t (GB_creal (x))
GB_CAST_FUNCTION (int8_t    , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to int16_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (int16_t   , bool      )
GB_CAST_FUNCTION (int16_t   , int8_t    )
GB_CAST_FUNCTION (int16_t   , int16_t   )
GB_CAST_FUNCTION (int16_t   , int32_t   )
GB_CAST_FUNCTION (int16_t   , int64_t   )
GB_CAST_FUNCTION (int16_t   , uint8_t   )
GB_CAST_FUNCTION (int16_t   , uint16_t  )
GB_CAST_FUNCTION (int16_t   , uint32_t  )
GB_CAST_FUNCTION (int16_t   , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int16_t ((double) x)
GB_CAST_FUNCTION (int16_t   , float     )
GB_CAST_FUNCTION (int16_t   , double    )

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int16_t ((double) GB_crealf (x))
GB_CAST_FUNCTION (int16_t   , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int16_t (GB_creal (x))
GB_CAST_FUNCTION (int16_t   , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to int32_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (int32_t   , bool      )
GB_CAST_FUNCTION (int32_t   , int8_t    )
GB_CAST_FUNCTION (int32_t   , int16_t   )
GB_CAST_FUNCTION (int32_t   , int32_t   )
GB_CAST_FUNCTION (int32_t   , int64_t   )
GB_CAST_FUNCTION (int32_t   , uint8_t   )
GB_CAST_FUNCTION (int32_t   , uint16_t  )
GB_CAST_FUNCTION (int32_t   , uint32_t  )
GB_CAST_FUNCTION (int32_t   , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int32_t ((double) x)
GB_CAST_FUNCTION (int32_t   , float     )
GB_CAST_FUNCTION (int32_t   , double    )

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int32_t ((double) GB_crealf (x))
GB_CAST_FUNCTION (int32_t   , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int32_t (GB_creal (x))
GB_CAST_FUNCTION (int32_t   , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to int64_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (int64_t   , bool      )
GB_CAST_FUNCTION (int64_t   , int8_t    )
GB_CAST_FUNCTION (int64_t   , int16_t   )
GB_CAST_FUNCTION (int64_t   , int32_t   )
GB_CAST_FUNCTION (int64_t   , int64_t   )
GB_CAST_FUNCTION (int64_t   , uint8_t   )
GB_CAST_FUNCTION (int64_t   , uint16_t  )
GB_CAST_FUNCTION (int64_t   , uint32_t  )
GB_CAST_FUNCTION (int64_t   , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int64_t ((double) x)
GB_CAST_FUNCTION (int64_t   , float     )
GB_CAST_FUNCTION (int64_t   , double    )

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int64_t ((double) GB_crealf (x))
GB_CAST_FUNCTION (int64_t   , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int64_t (GB_creal (x))
GB_CAST_FUNCTION (int64_t   , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to uint8_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (uint8_t   , bool      )
GB_CAST_FUNCTION (uint8_t   , int8_t    )
GB_CAST_FUNCTION (uint8_t   , int16_t   )
GB_CAST_FUNCTION (uint8_t   , int32_t   )
GB_CAST_FUNCTION (uint8_t   , int64_t   )
GB_CAST_FUNCTION (uint8_t   , uint8_t   )
GB_CAST_FUNCTION (uint8_t   , uint16_t  )
GB_CAST_FUNCTION (uint8_t   , uint32_t  )
GB_CAST_FUNCTION (uint8_t   , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint8_t ((double) x)
GB_CAST_FUNCTION (uint8_t   , float     )
GB_CAST_FUNCTION (uint8_t   , double    )

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint8_t ((double) GB_crealf (x))
GB_CAST_FUNCTION (uint8_t   , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint8_t (GB_creal (x))
GB_CAST_FUNCTION (uint8_t   , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to uint16_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (uint16_t  , bool      )
GB_CAST_FUNCTION (uint16_t  , int8_t    )
GB_CAST_FUNCTION (uint16_t  , int16_t   )
GB_CAST_FUNCTION (uint16_t  , int32_t   )
GB_CAST_FUNCTION (uint16_t  , int64_t   )
GB_CAST_FUNCTION (uint16_t  , uint8_t   )
GB_CAST_FUNCTION (uint16_t  , uint16_t  )
GB_CAST_FUNCTION (uint16_t  , uint32_t  )
GB_CAST_FUNCTION (uint16_t  , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint16_t ((double) x)
GB_CAST_FUNCTION (uint16_t  , float     )
GB_CAST_FUNCTION (uint16_t  , double    )

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint16_t ((double) GB_crealf (x))
GB_CAST_FUNCTION (uint16_t  , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint16_t (GB_creal (x))
GB_CAST_FUNCTION (uint16_t  , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to uint32_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (uint32_t  , bool      )
GB_CAST_FUNCTION (uint32_t  , int8_t    )
GB_CAST_FUNCTION (uint32_t  , int16_t   )
GB_CAST_FUNCTION (uint32_t  , int32_t   )
GB_CAST_FUNCTION (uint32_t  , int64_t   )
GB_CAST_FUNCTION (uint32_t  , uint8_t   )
GB_CAST_FUNCTION (uint32_t  , uint16_t  )
GB_CAST_FUNCTION (uint32_t  , uint32_t  )
GB_CAST_FUNCTION (uint32_t  , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint32_t ((double) x)
GB_CAST_FUNCTION (uint32_t  , float     )
GB_CAST_FUNCTION (uint32_t  , double    )

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint32_t ((double) GB_crealf (x))
GB_CAST_FUNCTION (uint32_t  , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint32_t (GB_creal (x))
GB_CAST_FUNCTION (uint32_t  , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to uint64_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (uint64_t  , bool      )
GB_CAST_FUNCTION (uint64_t  , int8_t    )
GB_CAST_FUNCTION (uint64_t  , int16_t   )
GB_CAST_FUNCTION (uint64_t  , int32_t   )
GB_CAST_FUNCTION (uint64_t  , int64_t   )
GB_CAST_FUNCTION (uint64_t  , uint8_t   )
GB_CAST_FUNCTION (uint64_t  , uint16_t  )
GB_CAST_FUNCTION (uint64_t  , uint32_t  )
GB_CAST_FUNCTION (uint64_t  , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint64_t ((double) x)
GB_CAST_FUNCTION (uint64_t  , float     )
GB_CAST_FUNCTION (uint64_t  , double    )

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint64_t ((double) GB_crealf (x))
GB_CAST_FUNCTION (uint64_t  , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint64_t (GB_creal (x))
GB_CAST_FUNCTION (uint64_t  , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to float
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
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

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_crealf (x)
GB_CAST_FUNCTION (float     , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) ((float) GB_creal (x))
GB_CAST_FUNCTION (float     , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to double
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
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

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA
#undef  GB_CAST
#define GB_CAST(ztype,x) ((double) GB_crealf (x))
GB_CAST_FUNCTION (double    , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_creal (x)
GB_CAST_FUNCTION (double    , GxB_FC64_t)
#endif

//------------------------------------------------------------------------------
// typecast to float complex
//------------------------------------------------------------------------------

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA

#undef  GB_CAST
#define GB_CAST(ztype,x) GB_CMPLX32 ((float) x, (float) 0)
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
#undef  GB_CAST
#define GB_CAST(ztype,x) x
GB_CAST_FUNCTION (GxB_FC32_t, GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_CMPLX32 ((float) GB_creal (x), (float) GB_cimag (x))
GB_CAST_FUNCTION (GxB_FC32_t, GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to double complex
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) GB_CMPLX64 ((double) x, (double) 0)
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
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_CMPLX64 ((double) GB_crealf (x), (double) GB_cimagf (x))
GB_CAST_FUNCTION (GxB_FC64_t, GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) x
GB_CAST_FUNCTION (GxB_FC64_t, GxB_FC64_t)

#endif

#undef  GB_CAST
#undef  GB_CAST_FUNCTION

//------------------------------------------------------------------------------
// GB_mcast: cast a mask entry from any native type to boolean
//------------------------------------------------------------------------------

// The mask matrix M must be one of the native data types, which have sizes of
// 1, 2, 4, 8, or 16 bytes.  The value could be properly typecasted to bool,
// but this requires a function pointer to the proper GB_cast_function.
// Instead, it is faster to simply use type punning, based on the size of the
// data type, and use the inline GB_mcast function instead.

// By default, the pre-generated kernels in Source/FactoryKernels and the
// generic kernels use GB_void as the data type for the M matrix, and then call
// this function to evaluate the value of a mask entry M(i,j).  The JIT kernels
// have a specific matrix type for M, and they #undef these definitions and
// replace them with specific types and a simpler test of the M(i,j) value.

#define GB_M_TYPE GB_void
#define GB_MCAST(Mx,pM,msize) GB_mcast ((GB_void *) Mx, pM, msize)

static inline bool GB_mcast         // return the value of M(i,j)
(
    const GB_void *restrict Mx,     // mask values
    const int64_t pM,               // extract boolean value of Mx [pM]
    const size_t msize              // size of each data type
)
{
    if (Mx == NULL)
    {
        // If Mx is NULL, then values in the mask matrix M are ignored, and
        // only the structural pattern is used.  This function is only called
        // for entries M(i,j) in the structure of M, so the result is always
        // true if Mx is NULL.
        return (true) ;
    }
    else
    {
        // check the value of M(i,j)
        switch (msize)
        {
            default:
            case GB_1BYTE: return ((*(uint8_t  *) (Mx +((pM)*1))) != 0) ;
            case GB_2BYTE: return ((*(uint16_t *) (Mx +((pM)*2))) != 0) ;
            case GB_4BYTE: return ((*(uint32_t *) (Mx +((pM)*4))) != 0) ;
            case GB_8BYTE: return ((*(uint64_t *) (Mx +((pM)*8))) != 0) ;
            case GB_16BYTE:
            {
                const uint64_t *restrict Zx = (uint64_t *) Mx ;
                return (Zx [2*pM] != 0 || Zx [2*pM+1] != 0) ;
            }
        }
    }
}

#endif
