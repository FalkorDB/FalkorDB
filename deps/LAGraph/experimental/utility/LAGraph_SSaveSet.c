//------------------------------------------------------------------------------
// LAGraph_SSaveSet: save a set of matrices to a *.lagraph file
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

// LAGraph_SSaveSet saves a set of matrices to a *.lagraph file.
// The file is created, written to with the JSON header and the serialized
// matrices, and then closed.  If using SuiteSparse:GraphBLAS, the highest
// level of compression is used (LZ4HC:9).

// Use LAGraph_SSLoadSet to load the matrices back in from the file.

// This method will not work without SuiteSparse:GraphBLAS, because the C API
// has no GrB* method for querying the GrB_Type (or its name as a string) of a
// matrix.

//------------------------------------------------------------------------------

#define LG_FREE_WORK                                \
{                                                   \
    fclose (f) ;                                    \
    f = NULL ;                                      \
    GrB_free (&desc) ;                              \
    LAGraph_SFreeContents (&Contents, nmatrices) ;  \
}

#define LG_FREE_ALL                                 \
{                                                   \
    LG_FREE_WORK ;                                  \
}

#include "LG_internal.h"
#include "LAGraphX.h"

//------------------------------------------------------------------------------
// LAGraph_SSaveSet
//------------------------------------------------------------------------------

int LAGraph_SSaveSet            // save a set of matrices from a *.lagraph file
(
    // inputs:
    char *filename,             // name of file to write to
    GrB_Matrix *Set,            // array of GrB_Matrix of size nmatrices
    GrB_Index nmatrices,        // # of matrices to write to *.lagraph file
    char *collection,           // name of this collection of matrices
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    FILE *f = NULL ;

    LAGraph_Contents *Contents = NULL ;
    GrB_Descriptor desc = NULL ;

    LG_ASSERT (filename != NULL && Set != NULL && collection != NULL,
        GrB_NULL_POINTER) ;

    #if LAGRAPH_SUITESPARSE
    GRB_TRY (GrB_Descriptor_new (&desc)) ;
    GRB_TRY (GxB_set (desc, GxB_COMPRESSION, GxB_COMPRESSION_LZ4HC + 9)) ;
    #endif

    f = fopen (filename, "wb") ;
    LG_ASSERT_MSG (f != NULL, -1001, "unable to create output file") ;

    //--------------------------------------------------------------------------
    // serialize all the matrices
    //--------------------------------------------------------------------------

    // allocate an Contents array of size nmatrices to hold the contents
    LG_TRY (LAGraph_Calloc ((void **) &Contents, nmatrices,
        sizeof (LAGraph_Contents), msg)) ;

    for (GrB_Index i = 0 ; i < nmatrices ; i++)
    {
        #if LAGRAPH_SUITESPARSE
        {
            GRB_TRY (GxB_Matrix_serialize (&(Contents [i].blob),
                (GrB_Index *)&(Contents [i].blob_size), Set [i], desc)) ;
        }
        #else
        {
            GrB_Index estimate ;
            GRB_TRY (GrB_Matrix_serializeSize (&estimate, Set [i])) ;
            Contents [i].blob_size = estimate ;
            LAGRAPH_TRY (LAGraph_Malloc ((void **) &(Contents [i].blob),
                estimate, sizeof (uint8_t), msg)) ;
            GRB_TRY (GrB_Matrix_serialize (Contents [i].blob,
                (GrB_Index *) &(Contents [i].blob_size), Set [i])) ;
            LG_TRY (LAGraph_Realloc ((void **) &(Contents [i].blob),
                (size_t) Contents [i].blob_size,
                estimate, sizeof (uint8_t), msg)) ;
        }
        #endif
    }

    //--------------------------------------------------------------------------
    // write the header
    //--------------------------------------------------------------------------

    LG_TRY (LAGraph_SWrite_HeaderStart (f, collection, msg)) ;
    for (GrB_Index i = 0 ; i < nmatrices ; i++)
    {
        char typename [GxB_MAX_NAME_LEN] ;
        LG_TRY (LAGraph_Matrix_TypeName (typename, Set [i], msg)) ;
        char matrix_name [256] ;
        snprintf (matrix_name, 256, "A_%" PRIu64, i) ;
        LG_TRY (LAGraph_SWrite_HeaderItem (f, LAGraph_matrix_kind,
            matrix_name, typename, 0, Contents [i].blob_size, msg)) ;
    }
    LG_TRY (LAGraph_SWrite_HeaderEnd (f, msg)) ;

    //--------------------------------------------------------------------------
    // write all the blobs
    //--------------------------------------------------------------------------

    for (GrB_Index i = 0 ; i < nmatrices ; i++)
    {
        LG_TRY (LAGraph_SWrite_Item (f, Contents [i].blob,
            Contents [i].blob_size, msg)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
