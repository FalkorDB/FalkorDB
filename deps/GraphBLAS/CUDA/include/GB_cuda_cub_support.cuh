//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_cub_support.cuh: definitions for CUB
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Supporting definitions for using cub:: methods

#pragma once

//------------------------------------------------------------------------------
// GB_CUB_BLOCK_WORKSPACE: declare workspace W of type _T for cub::Block*
//------------------------------------------------------------------------------

#define GB_CUB_BLOCK_WORKSPACE(W,_T,blockdim,items_per_thread)                \
using BlockLoad  = cub::BlockLoad <_T, blockdim, items_per_thread> ;          \
using BlockScan  = cub::BlockScan <_T, blockdim, cub::BLOCK_SCAN_WARP_SCANS> ;\
using BlockStore = cub::BlockStore <_T, blockdim, items_per_thread> ;         \
__shared__ union                                                              \
{                                                                             \
    typename BlockLoad::TempStorage load ;                                    \
    typename BlockScan::TempStorage scan ;                                    \
    typename BlockStore::TempStorage store ;                                  \
} W ;

