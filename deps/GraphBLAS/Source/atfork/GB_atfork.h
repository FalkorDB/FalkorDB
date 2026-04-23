//------------------------------------------------------------------------------
// GB_atfork.h: methods to support forking a process
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ATFORK_H
#define GB_ATFORK_H

#include "GB.h"
#include "jitifyer/GB_jitifyer.h"

void *GB_child_malloc (size_t size) ;
void GB_child_free (void *p) ;
int GB_child_printf (const char *restrict format, ...) ;
int GB_child_flush (void) ;
void GxB_atfork_child (void) ;


#endif

