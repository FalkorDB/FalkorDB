//------------------------------------------------------------------------------
// LAGraph_SWrite: write a sequence of serialized objects to a file
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

#include "LG_internal.h"
#include "LAGraphX.h"

#define FPRINT(params)                                  \
{                                                       \
    int result = fprintf params ;                       \
    LG_ASSERT_MSG (result >= 0, -1002, "file not written") ;  \
}

//------------------------------------------------------------------------------
// LAGraph_SWrite_HeaderStart
//------------------------------------------------------------------------------

int LAGraph_SWrite_HeaderStart  // write the first part of the JSON header
(
    FILE *f,                    // file to write to
    const char *name,           // name of this collection of matrices
    char *msg
)
{
    // check inputs
    LG_CLEAR_MSG ;
    LG_ASSERT (f != NULL && name != NULL, GrB_NULL_POINTER) ;

    // write the first part of the JSON header to the file
    FPRINT ((f, "{\n    \"LAGraph\": [%d,%d,%d],\n    \"GraphBLAS\": [ ",
        LAGRAPH_VERSION_MAJOR, LAGRAPH_VERSION_MINOR, LAGRAPH_VERSION_UPDATE)) ;

    #if LAGRAPH_SUITESPARSE

        // SuiteSparse:GraphBLAS v6.0.0 or later
        char *library ;
        int ver [3] ;
        GRB_TRY (GxB_get (GxB_LIBRARY_NAME, &library)) ;
        GRB_TRY (GxB_get (GxB_LIBRARY_VERSION, ver)) ;
        FPRINT ((f, "\"%s\", [%d,%d,%d] ],\n", library,
            ver [0], ver [1], ver [2])) ;

    #else

        // some other GraphBLAS library: call it "vanilla 1.0.0"
        FPRINT ((f, "\"%s\", [%d,%d,%d] ],\n", "vanilla", 1, 0, 0)) ;

    #endif

    // write name of this collection and start the list of items
    FPRINT ((f, "    \"%s\":\n    [\n", name)) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// LAGraph_SWrite_HeaderItem
//------------------------------------------------------------------------------

int LAGraph_SWrite_HeaderItem   // write a single item to the JSON header
(
    // inputs:
    FILE *f,                    // file to write to
    LAGraph_Contents_kind kind, // matrix, vector, or text
    const char *name,           // name of the matrix/vector/text; matrices from
                                // sparse.tamu.edu use the form "Group/Name"
    const char *type,           // name of type of the matrix/vector
    // todo: vectors and text not yet supported by LAGraph_SWrite_HeaderItem
    int compression,            // text compression method
    GrB_Index blob_size,        // exact size of serialized blob for this item
    char *msg
)
{
    // check inputs
    LG_CLEAR_MSG ;
    LG_ASSERT (f != NULL, GrB_NULL_POINTER) ;

    // write the JSON information for this item
    FPRINT ((f, "        { \"")) ;
    switch (kind)
    {
        case LAGraph_matrix_kind :
            FPRINT ((f, "GrB_Matrix\": \"%s\", \"type\": \"%s", name, type)) ;
            break ;

        // todo: handle vectors and text
#if 0
        case LAGraph_vector_kind :
            FPRINT((f, "GrB_Vector\": \"%s\", \"type\": \"%s", name, type)) ;
            break ;

        case LAGraph_text_kind :
            FPRINT ((f, "text\": \"%s\", \"compression\": \"", name)) ;
            switch (compression)
            {
                case -1   : FPRINT ((f, "none"   )) ; break ;
                case 0    : FPRINT ((f, "default")) ; break ;
                case 1000 : FPRINT ((f, "lz4"    )) ; break ;
                case 2000 : FPRINT ((f, "lz4hc:0")) ; break ;
                case 2001 : FPRINT ((f, "lz4hc:1")) ; break ;
                case 2002 : FPRINT ((f, "lz4hc:2")) ; break ;
                case 2003 : FPRINT ((f, "lz4hc:3")) ; break ;
                case 2004 : FPRINT ((f, "lz4hc:4")) ; break ;
                case 2005 : FPRINT ((f, "lz4hc:5")) ; break ;
                case 2006 : FPRINT ((f, "lz4hc:6")) ; break ;
                case 2007 : FPRINT ((f, "lz4hc:7")) ; break ;
                case 2008 : FPRINT ((f, "lz4hc:8")) ; break ;
                case 2009 : FPRINT ((f, "lz4hc:9")) ; break ;
                default   : LG_ASSERT_MSG (false, GrB_INVALID_VALUE,
                    "invalid compression") ; break ;
            }
            break ;
#endif

        default :
            LG_ASSERT_MSG (false, GrB_INVALID_VALUE, "invalid kind") ;
            break ;
    }

    FPRINT ((f, "\", \"bytes\": %" PRIu64 " },\n", blob_size)) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// LAGraph_SWrite_HeaderEnd
//------------------------------------------------------------------------------

int LAGraph_SWrite_HeaderEnd    // write the end of the JSON header
(
    FILE *f,                    // file to write to
    char *msg
)
{
    // check inputs
    LG_CLEAR_MSG ;
    LG_ASSERT (f != NULL, GrB_NULL_POINTER) ;

    // finalize the JSON header string
    FPRINT ((f, "        null\n    ]\n}\n")) ;

    // write a final zero byte to terminate the JSON string
    fputc (0, f) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// LAGraph_SWrite_Item
//------------------------------------------------------------------------------

int LAGraph_SWrite_Item  // write the serialized blob of a matrix/vector/text
(
    // input:
    FILE *f,                // file to write to
    const void *blob,       // serialized blob from G*B_Matrix_serialize
    GrB_Index blob_size,    // exact size of the serialized blob
    char *msg
)
{
    // check inputs
    LG_CLEAR_MSG ;
    LG_ASSERT (f != NULL && blob != NULL, GrB_NULL_POINTER) ;

    // write the blob
    size_t blob_written = fwrite (blob, sizeof (uint8_t), blob_size, f) ;
    LG_ASSERT_MSG (blob_written == blob_size, -1001,
        "file not written properly") ;
    return (GrB_SUCCESS) ;
}
