//------------------------------------------------------------------------------
// GB_file.h: portable file I/O
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_FILE_H
#define GB_FILE_H

#if (defined (_WIN64) || defined (_WIN32))
#define GB_WINDOWS 1
#else
#define GB_WINDOWS 0
#endif

// to pipe stdout to /dev/null
#if GB_WINDOWS
#define GB_DEV_NULL " > nul"
#else
#define GB_DEV_NULL " > /dev/null"
#endif

bool GB_file_mkdir (char *path) ;

void *GB_file_dlopen (char *library_name) ;

void *GB_file_dlsym (void *dl_handle, char *symbol) ;

void GB_file_dlclose (void *dl_handle) ;

#endif

