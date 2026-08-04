//------------------------------------------------------------------------------
// gbmex_serialize: serialize a matrix into a blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_serialize is an interface to GxB_Matrix_serialize.

// Usage:

// blob = gbmex_serialize (ghb, A, method, level)

// The blob is returned as the opaque content of an n-by-1 uint8 GrB or GhB
// matrix.

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                       \
    GrB_Matrix_free (&A_to_free) ;      \
    GrB_Descriptor_free (&desc) ;

#undef  FREE_ALL
#define FREE_ALL                        \
    FREE_WORK ;                         \
    gb_free ((void **) &blob, arena) ;  \
    GrB_Vector_free (&Blob) ;

#define USAGE "usage: blob = GrB.serialize (A, method, level)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Matrix *Blob_opaque = NULL, A = NULL, A_to_free = NULL ;
    GrB_Vector Blob = NULL ;
    GrB_Descriptor desc = NULL ;
    void *blob = NULL ;

    GBMX_USAGE (nargin >= 2 && nargin <= 4 && nargout <= 1, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&Blob_opaque) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    char method_name [LEN+2] ;

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [1]) ;

    int method = GxB_COMPRESSION_DEFAULT ;
    int level = 0 ;     // use whatever is the default for the method

    if (nargin > 2)
    { 
        gbmx_mxstring_to_string (method_name, LEN, pargin [2], "method") ;
    }

    // get the method level
    if (nargin > 3)
    { 
        level = (int) mxGetScalar (pargin [3]) ;
    }
    if (level < 0 || level > 999) level = 0 ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;

    //--------------------------------------------------------------------------
    // create descriptor
    //--------------------------------------------------------------------------

    bool debug = false ;
    if (nargin > 2)
    { 
        // create the descriptor
        OK (GxB_Descriptor_new_arena (&desc, arena)) ;
        // get the method
        if (MATCH (method_name, "none"))
        { 
            method = GxB_COMPRESSION_NONE ;
        }
        else if (MATCH (method_name, "lz4"))
        { 
            method = GxB_COMPRESSION_LZ4 ;
        }
        else if (MATCH (method_name, "lz4hc"))
        { 
            method = GxB_COMPRESSION_LZ4HC ;
        }
        else if (MATCH (method_name, "default") || MATCH (method_name, "zstd"))
        { 
            // the default is ZSTD, with level 1
            method = GxB_COMPRESSION_ZSTD ;
        }
        else if (MATCH (method_name, "debug"))
        { 
            // use GrB_Matrix_serializeSize and GrB_Matrix_serialize, just
            // for testing
            debug = true ;
        }
        else
        { 
            ERROR ("unknown method", GrB_INVALID_VALUE) ;
        }
        // set the descriptor
        OK (GrB_Descriptor_set_INT32 (desc, method + level, GxB_COMPRESSION)) ;
    }

    //--------------------------------------------------------------------------
    // serialize the matrix into the blob (in arena 0)
    //--------------------------------------------------------------------------

    uint64_t blob_memsize = 0 ;

    if (debug)
    { 
        // debug GrB_Matrix_serializeSize and GrB_Matrix_serialize
        OK (GrB_Matrix_serializeSize (&blob_memsize, A)) ;
        blob = gb_malloc (blob_memsize, arena) ;
        OK (GrB_Matrix_serialize (blob, &blob_memsize, A)) ;
        // shrink the blob to its actual size
        // blob = realloc (blob, blob_memsize) ;    // this is skipped
    }
    else
    { 
        // use GxB_Matrix_serialize by default
        OK (GxB_Matrix_serialize_arena (&blob, &blob_memsize, A, arena, desc)) ;
    }

    //--------------------------------------------------------------------------
    // transfer the blob into the output Blob vector
    //--------------------------------------------------------------------------

    OK (GxB_Vector_new_arena (&Blob, GrB_UINT8, blob_memsize, arena, arena)) ;
    OK (GxB_Vector_load (Blob, &blob, GrB_UINT8, blob_memsize, blob_memsize,
        GrB_DEFAULT + arena, NULL)) ;
    ASSERT (blob == NULL) ;

    //--------------------------------------------------------------------------
    // free workspace and return results
    //--------------------------------------------------------------------------

    FREE_WORK ;

    OK (gb_export (Blob_opaque, (GrB_Matrix *) &Blob, KIND_GRB, ghb, err)) ;
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct ((GrB_Matrix *) &Blob) ;
    }

    gb_wrapup ( ) ;
}

