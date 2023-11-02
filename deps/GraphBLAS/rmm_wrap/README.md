# rmm_wrap is a C-callable wrapper for the NVIDIA Rapids Memory Manager.

SPDX-License-Identifier: Apache-2.0

rmm_wrap defines a single global object, the RMM_Wrap_Handle that holds
an RMM (Rapids Memory Manager) memory resource and a hash map (C++
std:unordered_map).  This allows rmm_wrap to provide 7 functions to a C
application:

Create/destroy an RMM resource:

    rmm_wrap_initialize: create the RMM resource
    rmm_wrap_finalize: destroy the RMM resource

C-style malloc/calloc/realloc/free methods:

    rmm_wrap_malloc:  malloc a block of memory using RMM
    rmm_wrap_calloc:  calloc a block of memory using RMM
    rmm_wrap_realloc: realloc a block of allocated by this RMM wrapper
    rmm_wrap_free:    free a block of memory allocated by this RMM wrapper

PMR-based allocate/deallocate methods (C-callable):

    rmm_wrap_allocate (std::size_t *size)
    rmm_wrap_deallocate (void *p, std::size_t size)

Files in this package:

    CMakeLists.txt      compiles the rmm_wrap library
    README.md           this file
    rmm_wrap.cpp        rmm_wrap_* functions
    rmm_wrap.h          definitions for an external C program
    rmm_wrap.hpp        internal definitions for rmm_wrap only
    rmm_wrap_test.c     tests for the rmm_wrap library

