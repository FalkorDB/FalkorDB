//-----------------------------------------------------------------------------
// LAGraph/src/test/include/LAGraph_test.h:  defintions for testing LAGraph
//-----------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

// Contributed by Timothy A. Davis, Texas A&M University

//-----------------------------------------------------------------------------

#ifndef LAGRAPH_TEST_H
#define LAGRAPH_TEST_H

//------------------------------------------------------------------------------
// LAGraph include files
//------------------------------------------------------------------------------

#include <LAGraph.h>
#include <LG_test.h>
#include <acutest.h>
#include <graph_zachary_karate.h>

#if LAGRAPH_SUITESPARSE
// to allow tests to call GrB_init twice
void GB_Global_GrB_init_called_set (bool GrB_init_called) ;
#endif

//------------------------------------------------------------------------------
// test macros
//------------------------------------------------------------------------------

// The tests are compiled with -DLGDIR=/home/me/LAGraph, or whatever, where
// /home/me/LAGraph is the top-level LAGraph source directory.
#define LG_XSTR(x) LG_STR(x)
#define LG_STR(x) #x
#define LG_SOURCE_DIR LG_XSTR (LGDIR)

#define LG_DATA_DIR LG_SOURCE_DIR "/data/"

#define OK(method) TEST_CHECK (method == 0)

//------------------------------------------------------------------------------
// typename: return the name of a type
//------------------------------------------------------------------------------

static inline const char *typename (GrB_Type type)
{
    if      (type == GrB_BOOL  ) return ("GrB_BOOL") ;
    else if (type == GrB_INT8  ) return ("GrB_INT8") ;
    else if (type == GrB_INT16 ) return ("GrB_INT16") ;
    else if (type == GrB_INT32 ) return ("GrB_INT32") ;
    else if (type == GrB_INT64 ) return ("GrB_INT64") ;
    else if (type == GrB_UINT8 ) return ("GrB_UINT8") ;
    else if (type == GrB_UINT16) return ("GrB_UINT16") ;
    else if (type == GrB_UINT32) return ("GrB_UINT32") ;
    else if (type == GrB_UINT64) return ("GrB_UINT64") ;
    else if (type == GrB_FP32  ) return ("GrB_FP32") ;
    else if (type == GrB_FP64  ) return ("GrB_FP64") ;
    #if 0
    else if (type == GxB_FC32  ) return ("GxB_FC32") ;
    else if (type == GxB_FC64  ) return ("GxB_FC64") ;
    #endif
    return (NULL) ;
}

#endif

