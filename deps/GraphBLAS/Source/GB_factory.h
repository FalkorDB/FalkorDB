//------------------------------------------------------------------------------
// GB_factory.h: for testing factory kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_FACTORY_H
#define GB_FACTORY_H

// For testing in MATLAB, factory kernels can be disabled
#ifdef GBMATLAB
extern bool GB_factory_kernels_enabled ;
#define GB_IF_FACTORY_KERNELS_ENABLED if (GB_factory_kernels_enabled)
#else
#define GB_IF_FACTORY_KERNELS_ENABLED
#endif

#endif

