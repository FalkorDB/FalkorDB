//------------------------------------------------------------------------------
// GxB_atfork_parent: actions GraphBLAS must after a child is forked
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is suitable for passing to pthread_atfork
// (see https://man7.org/linux/man-pages/man3/pthread_atfork.3.html ) as the
// 2nd parameter.  The parent must call this method after it forks a child,
// via pthread_atfork or by calling it directly (which is suitable for use in
// non-POSIX systems such as Windows).

// Currently, this method does nothing.

//------------------------------------------------------------------------------

#include "GB_atfork.h"

//------------------------------------------------------------------------------
// GxB_atfork_parent
//------------------------------------------------------------------------------

void GxB_atfork_parent (void)
{
    ; // do nothing
}

