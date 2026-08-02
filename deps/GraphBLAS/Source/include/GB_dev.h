//------------------------------------------------------------------------------
// GB_dev.h: definitions for code development
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_DEV_H
#define GB_DEV_H

//------------------------------------------------------------------------------
// code development settings: by default, all settings should be commented out
//------------------------------------------------------------------------------

// To turn on Debug for a single file of GraphBLAS, add '#define GB_DEBUG' at
// the top of the file, before any other #include statements.  GraphBLAS will
// be exceedingly slow if these flags are set; they are for development only>

// to turn on Debug for all of GraphBLAS, uncomment this line:
// #define GB_DEBUG

// to turn on a very verbose memory trace:
// #define GB_MEMDUMP

// to enable the debug-only global memtable:
// #define GB_MEMTABLE_DEBUG

#endif

