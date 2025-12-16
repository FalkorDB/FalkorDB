//------------------------------------------------------------------------------
// rmm_wrap/rmm_wrap.hpp
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#pragma once

#include "stddef.h"
#include <cuda.h>
//#include <rmm/cuda_stream.hpp>
//#include <rmm/device_buffer.hpp>
//#include <rmm/detail/aligned.hpp>
#include <rmm/mr/host/host_memory_resource.hpp>
#include <rmm/mr/host/new_delete_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
// #include <rmm/cuda_stream_view.hpp>
// #include <rmm/cuda_stream_pool.hpp>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <unordered_map>
#include "rmm_wrap.h"

typedef rmm::mr::new_delete_resource host_mr;
typedef rmm::mr::pinned_memory_resource pinned_mr;
typedef rmm::mr::cuda_memory_resource device_mr;
typedef rmm::mr::managed_memory_resource managed_mr;
typedef rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr;

typedef rmm::mr::pool_memory_resource<host_mr> host_pool_mr;
typedef rmm::mr::pool_memory_resource<pinned_mr> host_pinned_pool_mr;
typedef rmm::mr::pool_memory_resource<device_mr> device_pool_mr;
typedef rmm::mr::pool_memory_resource<managed_mr> managed_pool_mr;

typedef std::unordered_map< std::size_t, std::size_t> alloc_map;

// typedef rmm::cuda_stream_pool cuda_stream_pool;
// typedef rmm::cuda_stream_view cuda_stream_view;

