//------------------------------------------------------------------------------
// GraphBLAS/CUDA/memory/GB_rmm_free:  thread-safe wrapper for rmm_wrap_free
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// fixme for CUDA: there will need to be one unique malloc and a unique free
// method for each GPU.

#include "GB.h"

void GB_rmm_free (void *p)
{
    GB_OPENMP_LOCK_SET (2)      // rmm_wrap_malloc/rmm_wrap_free
    rmm_wrap_free (p) ;
    GB_OPENMP_LOCK_UNSET (2)    // rmm_wrap_malloc/rmm_wrap_free
}
