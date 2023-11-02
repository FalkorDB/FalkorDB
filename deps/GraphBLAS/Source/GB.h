//------------------------------------------------------------------------------
// GB.h: definitions visible only inside GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_H
#define GB_H

#include "GB_Template.h"
#include "GB_Global.h"
#include "GB_printf.h"
#include "GB_assert.h"
#if defined ( SUITESPARSE_CUDA )
#include "rmm_wrap.h"
#endif
#include "GB_static_header.h"
#include "GB_positional.h"
#include "GB_casting.h"
#include "GB_math.h"
#include "GB_bitwise.h"
#include "GB_check.h"
#include "GB_nnz.h"
#include "GB_omp.h"
#include "GB_memory.h"
#include "GB_iso.h"
#include "GB_Pending_n.h"
#include "GB_nvals.h"
#include "GB_aliased.h"
#include "GB_new.h"
#include "GB_clear.h"
#include "GB_resize.h"
#include "GB_dup.h"
#include "GB_code_compatible.h"
#include "GB_compatible.h"
#include "GB_task_methods.h"
#include "GB_transplant.h"
#include "GB_type.h"
#include "GB_slice.h"
#include "GB_uint64_multiply.h"
#include "GB_int64_multiply.h"
#include "GB_size_t_multiply.h"
#include "GB_extractTuples.h"
#include "GB_cumsum.h"
#include "GB_Descriptor_get.h"
#include "GB_Element.h"
#include "GB_op.h"
#include "GB_hyper.h"
#include "GB_ok.h"
#include "GB_cast.h"
#include "GB_wait.h"
#include "GB_convert.h"
#include "GB_ops.h"
#include "GB_where.h"
#include "GB_Context.h"
#include "GB_cuda_gateway.h"
#include "GB_saxpy3task_struct.h"
#include "GB_callbacks.h"
#include "GB_factory.h"
#endif

