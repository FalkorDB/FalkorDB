//------------------------------------------------------------------------------
// GB_ops.h: built-in unary and binary operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_OPS_H
#define GB_OPS_H

//------------------------------------------------------------------------------
// define all built-in unary and binary operators
//------------------------------------------------------------------------------

// nonzombie function for generic case
inline void GB_nonzombie_func (bool *z, const void *x,
    int64_t i, GrB_Index j, const void *y)
{
    (*z) = (i >= 0) ;    
}

#define GB_FUNC_T(op,xtype) GB (GB_EVAL4 (_func_, op, _, xtype))
#define GB_FUNC(op) GB_FUNC_T (op, GB_XTYPE)

#define GB_TYPE             bool
#define GB_XTYPE            BOOL
#define GB_BITS             1
#define GB_REAL
#define GB_BOOLEAN
#include "GB_ops_template.h"

#define GB_TYPE             int8_t
#define GB_XTYPE            INT8
#define GB_BITS             8
#define GB_REAL
#define GB_SIGNED_INT
#include "GB_ops_template.h"

#define GB_TYPE             int16_t
#define GB_XTYPE            INT16
#define GB_BITS             16
#define GB_REAL
#define GB_SIGNED_INT
#include "GB_ops_template.h"

#define GB_TYPE             int32_t
#define GB_XTYPE            INT32
#define GB_BITS             32
#define GB_REAL
#define GB_SIGNED_INT
#define GB_SIGNED_INDEX
#include "GB_ops_template.h"

#define GB_TYPE             int64_t
#define GB_XTYPE            INT64
#define GB_BITS             64
#define GB_REAL
#define GB_SIGNED_INT
#define GB_SIGNED_INDEX
#define GB_SIGNED_INDEX64
#include "GB_ops_template.h"

#define GB_TYPE             uint8_t
#define GB_XTYPE            UINT8
#define GB_BITS             8
#define GB_REAL
#define GB_UNSIGNED_INT
#include "GB_ops_template.h"

#define GB_TYPE             uint16_t
#define GB_XTYPE            UINT16
#define GB_BITS             16
#define GB_REAL
#define GB_UNSIGNED_INT
#include "GB_ops_template.h"

#define GB_TYPE             uint32_t
#define GB_XTYPE            UINT32
#define GB_BITS             32
#define GB_REAL
#define GB_UNSIGNED_INT
#include "GB_ops_template.h"

#define GB_TYPE             uint64_t
#define GB_XTYPE            UINT64
#define GB_BITS             64
#define GB_REAL
#define GB_UNSIGNED_INT
#include "GB_ops_template.h"

#define GB_TYPE             float
#define GB_XTYPE            FP32
#define GB_BITS             32
#define GB_REAL
#define GB_FLOATING_POINT
#define GB_FLOAT
#include "GB_ops_template.h"

#define GB_TYPE             double
#define GB_XTYPE            FP64
#define GB_BITS             64
#define GB_REAL
#define GB_FLOATING_POINT
#define GB_DOUBLE
#include "GB_ops_template.h"

#if !defined ( GBCUDA_CPLUSPLUS )
// TODO: does not yet work in CUDA

#define GB_TYPE             GxB_FC32_t
#define GB_XTYPE            FC32
#define GB_BITS             64
#define GB_COMPLEX
#define GB_FLOATING_POINT
#define GB_FLOAT_COMPLEX
#include "GB_ops_template.h"

#define GB_TYPE             GxB_FC64_t
#define GB_XTYPE            FC64
#define GB_BITS             128
#define GB_COMPLEX
#define GB_FLOATING_POINT
#define GB_DOUBLE_COMPLEX
#include "GB_ops_template.h"

#endif

#endif

