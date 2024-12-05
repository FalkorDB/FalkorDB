//------------------------------------------------------------------------------
// GB_defaults.h: default parameter settings
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_DEFAULTS_H
#define GB_DEFAULTS_H

//------------------------------------------------------------------------------
// default options
//------------------------------------------------------------------------------

// These parameters define the content of values that can be
// used as inputs to GxB_*Option_set.

#define GB_HYPER_SWITCH_DEFAULT (0.0625)
#define GB_HYPER_HASH_DEFAULT (1024)

// by default, give each thread at least 64K units of work to do
#define GB_CHUNK_DEFAULT (64*1024)

// initial size of the pending tuples
#define GB_PENDING_INIT 256

#endif

