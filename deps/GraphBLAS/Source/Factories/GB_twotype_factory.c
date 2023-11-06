//------------------------------------------------------------------------------
// GB_twotype_factory.c: 2-type switch factory
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is a generic switch factory for creating 169 workers that operate on
// two built-in data types (13 types each), to be #include'd in another file.
// GB_WORKER(op,zname,ztype,xname,xtype) is a macro defined in the #including
// file, where ztype and xtype are the built-in types corresponding to code1
// and code2, respectively or (void *) for a user-defined type.  The last
// statement of GB_WORKER should be a break or return since it does not appear
// here.

// User-defined types are not handled.

switch (code1)
{
    case GB_BOOL_code   :

        switch (code2)
        {
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _bool, bool, _bool,   bool      )
            #endif
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _bool, bool, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _bool, bool, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _bool, bool, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _bool, bool, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _bool, bool, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _bool, bool, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _bool, bool, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _bool, bool, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _bool, bool, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _bool, bool, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _bool, bool, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _bool, bool, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_INT8_code   :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _int8, int8_t, _bool,   bool      )
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _int8, int8_t, _int8,   int8_t    )
            #endif
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _int8, int8_t, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _int8, int8_t, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _int8, int8_t, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _int8, int8_t, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _int8, int8_t, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _int8, int8_t, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _int8, int8_t, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _int8, int8_t, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _int8, int8_t, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _int8, int8_t, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _int8, int8_t, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_INT16_code  :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _int16, int16_t, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _int16, int16_t, _int8,   int8_t    )
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _int16, int16_t, _int16,  int16_t   )
            #endif
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _int16, int16_t, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _int16, int16_t, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _int16, int16_t, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _int16, int16_t, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _int16, int16_t, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _int16, int16_t, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _int16, int16_t, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _int16, int16_t, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _int16, int16_t, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _int16, int16_t, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_INT32_code  :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _int32, int32_t, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _int32, int32_t, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _int32, int32_t, _int16,  int16_t   )
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _int32, int32_t, _int32,  int32_t   )
            #endif
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _int32, int32_t, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _int32, int32_t, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _int32, int32_t, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _int32, int32_t, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _int32, int32_t, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _int32, int32_t, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _int32, int32_t, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _int32, int32_t, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _int32, int32_t, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_INT64_code  :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _int64, int64_t, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _int64, int64_t, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _int64, int64_t, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _int64, int64_t, _int32,  int32_t   )
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _int64, int64_t, _int64,  int64_t   )
            #endif
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _int64, int64_t, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _int64, int64_t, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _int64, int64_t, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _int64, int64_t, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _int64, int64_t, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _int64, int64_t, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _int64, int64_t, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _int64, int64_t, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_UINT8_code  :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _int64,  int64_t   )
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _uint8,  uint8_t   )
            #endif
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _uint8, uint8_t, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_UINT16_code :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _uint8,  uint8_t   )
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _uint16, uint16_t  )
            #endif
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _uint16, uint16_t, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_UINT32_code :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _uint16, uint16_t  )
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _uint32, uint32_t  )
            #endif
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _uint32, uint32_t, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_UINT64_code :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _uint32, uint32_t  )
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _uint64, uint64_t  )
            #endif
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _uint64, uint64_t, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_FP32_code   :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _fp32, float, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _fp32, float, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _fp32, float, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _fp32, float, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _fp32, float, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _fp32, float, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _fp32, float, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _fp32, float, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _fp32, float, _uint64, uint64_t  )
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _fp32, float, _fp32,   float     )
            #endif
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _fp32, float, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _fp32, float, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _fp32, float, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_FP64_code   :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _fp64, double, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _fp64, double, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _fp64, double, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _fp64, double, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _fp64, double, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _fp64, double, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _fp64, double, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _fp64, double, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _fp64, double, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _fp64, double, _fp32,   float     )
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _fp64, double, _fp64,   double    )
            #endif
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _fp64, double, _fc32,   GxB_FC32_t)
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _fp64, double, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_FC32_code   :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _fc32,   GxB_FC32_t)
            #endif
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _fc32, GxB_FC32_t, _fc64,   GxB_FC64_t)
            #endif
            default: ;
        }
        break ;

    case GB_FC64_code   :

        switch (code2)
        {
            case GB_BOOL_code   : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _bool,   bool      )
            case GB_INT8_code   : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _int8,   int8_t    )
            case GB_INT16_code  : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _int16,  int16_t   )
            case GB_INT32_code  : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _int32,  int32_t   )
            case GB_INT64_code  : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _int64,  int64_t   )
            case GB_UINT8_code  : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _uint8,  uint8_t   )
            case GB_UINT16_code : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _uint16, uint16_t  )
            case GB_UINT32_code : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _uint32, uint32_t  )
            case GB_UINT64_code : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _uint64, uint64_t  )
            case GB_FP32_code   : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _fp32,   float     )
            case GB_FP64_code   : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _fp64,   double    )
            #if !defined ( SUITESPARSE_CUDA )
            // TODO: does not yet work in CUDA
            case GB_FC32_code   : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _fc32,   GxB_FC32_t)
            #if !defined ( GB_EXCLUDE_SAME_TYPES )
            case GB_FC64_code   : GB_WORKER (GB_OPNAME, _fc64, GxB_FC64_t, _fc64,   GxB_FC64_t)
            #endif
            #endif
            default: ;
        }
        break ;

    default: ;
}

#undef GB_OPNAME

