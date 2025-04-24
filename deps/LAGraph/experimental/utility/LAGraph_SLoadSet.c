//------------------------------------------------------------------------------
// LAGraph_SLoadSet: load a set of matrices from a *.lagraph file
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

// LAgraph_SLoadSet loads a set of GrB_Matrix objects from a *.lagraph file.
// It returns a GrB_Matrix array of size nmatrices.  In the future, it will
// also return a set of GrB_Vectors, and a an array of uncompressed ascii
// texts.  The caller is responsible for freeing the output of this method,
// via:

//      LAGraph_Free ((void **) &collection, NULL) ;
//      LAGraph_SFreeSet (&Set, nmatrices) ;

// See also LAGraph_SRead, which just reads in the serialized objects and
// does not convert them to their corresponding GrB_Matrix, GrB_Vector, or
// uncompressed texts.

//------------------------------------------------------------------------------

#define LG_FREE_WORK                                                \
{                                                                   \
    if (f != NULL && f != stdin) fclose (f) ;                       \
    f = NULL ;                                                      \
    LAGraph_SFreeContents (&Contents, ncontents) ;                  \
}

#define LG_FREE_ALL                                                 \
{                                                                   \
    LG_FREE_WORK ;                                                  \
    LAGraph_SFreeSet (&Set, nmatrices) ;                            \
    LAGraph_Free ((void **) &collection, NULL) ;                    \
}

#include "LG_internal.h"
#include "LAGraphX.h"

//------------------------------------------------------------------------------
// LAGraph_SLoadSet
//------------------------------------------------------------------------------

int LAGraph_SLoadSet            // load a set of matrices from a *.lagraph file
(
    // input:
    char *filename,                 // name of file to read; NULL for stdin
    // outputs:
    GrB_Matrix **Set_handle,        // array of GrB_Matrix of size nmatrices
    GrB_Index *nmatrices_handle,    // # of matrices loaded from *.lagraph file
//  todo: handle vectors and text in LAGraph_SLoadSet
//  GrB_Vector **Set_handle,        // array of GrB_Vector of size nvector
//  GrB_Index **nvectors_handle,    // # of vectors loaded from *.lagraph file
//  char **Text_handle,             // array of pointers to (char *) strings
//  GrB_Index **ntext_handle,       // # of texts loaded from *.lagraph file
    char **collection_handle,       // name of this collection of matrices
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    FILE *f = stdin ;
    char *collection = NULL ;
    GrB_Matrix *Set = NULL ;
    LAGraph_Contents *Contents = NULL ;
    GrB_Index ncontents = 0 ;
    GrB_Index nmatrices = 0 ;
//  GrB_Index nvectors = 0 ;
//  GrB_Index ntexts = 0 ;

    LG_ASSERT (Set_handle != NULL && nmatrices_handle != NULL
        && collection_handle != NULL, GrB_NULL_POINTER) ;

    //--------------------------------------------------------------------------
    // read the file
    //--------------------------------------------------------------------------

    if (filename != NULL)
    {
        f = fopen (filename, "rb") ;
        LG_ASSERT_MSG (f != NULL,
            LAGRAPH_IO_ERROR, "unable to open input file") ;
    }
    LG_TRY (LAGraph_SRead (f, &collection, &Contents, &ncontents, msg)) ;
    if (filename != NULL)
    {
        fclose (f) ;
    }
    f = NULL ;

    //--------------------------------------------------------------------------
    // count the matrices/vectors/texts in the Contents
    //--------------------------------------------------------------------------

    // todo: for now, all Contents are matrices
    nmatrices = ncontents ;

#if 0
    for (GrB_Index i = 0 ; i < ncontents ; i++)
    {
        switch (Contents [i].kind)
        {
            case LAGraph_matrix_kind : nmatrices++ ; break ;
            case LAGraph_vector_kind : nvectors++  ; break ;
            case LAGraph_text_kind   : ntexts++    ; break ;
            default : LG_ASSERT_MSG (false, GrB_INVALID_VALUE, "unknown kind") ;
        }
    }
    if (nvectors > 0 || ntexts > 0)
    {
        // todo: handle vectors and texts
        printf ("Warning: %lu vectors and %lu texts ignored\n",
            nvectors, ntexts) ;
    }
#endif

    //--------------------------------------------------------------------------
    // convert all the matrices (skip vectors and text content for now)
    //--------------------------------------------------------------------------

    LG_TRY (LAGraph_Calloc ((void **) &Set, nmatrices, sizeof (GrB_Matrix),
        msg)) ;

    GrB_Index kmatrices = 0 ;
    for (GrB_Index i = 0 ; i < ncontents ; i++)
    {
        // convert Contents [i]
        void *blob = Contents [i].blob ;
        size_t blob_size = Contents [i].blob_size ;

        if (Contents [i].kind == LAGraph_matrix_kind)
        {
            // convert Contents [i].typename to a GrB_Type ctype.
            // SuiteSparse:GraphBLAS allows this to be NULL for built-in types.
            GrB_Type ctype = NULL ;
            LG_TRY (LAGraph_TypeFromName (&ctype, Contents [i].type_name, msg));
            GRB_TRY (GrB_Matrix_deserialize (&(Set [kmatrices]), ctype, blob,
                blob_size)) ;
            kmatrices++ ;
        }
        // todo: handle vectors and texts
        // else if (Content [i].kind == LAGraph_vector_kind) ...
        // else if (Content [i].kind == LAGraph_text_kind) ...

        // free the ith blob
        LAGraph_Free ((void **) &(Contents [i].blob), NULL) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LG_FREE_WORK ;
    (*Set_handle) = Set ;
    (*collection_handle) = collection ;
    (*nmatrices_handle) = nmatrices ;
    return (GrB_SUCCESS) ;
}
