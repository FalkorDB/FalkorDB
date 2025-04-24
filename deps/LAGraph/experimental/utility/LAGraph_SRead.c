//------------------------------------------------------------------------------
// LAGraph_SRead: read a sequence of serialized objects from a file
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

// LAGraph_SRead reads a set of serialized items from a file. The items are
// returned as-is and not converted into GrB_Matrix, GrB_Vector, or text
// objects.  The user application is responsible for freeing the output of this
// method, via:

//      LAGraph_Free ((void **) &collection, NULL) ;
//      LAGraph_SFreeContents (&Contents, ncontents) ;

// See also LAGraph_SLoadSet, which calls this function and then converts all
// serialized objects into their GrB_Matrix, GrB_Vector, or text components.

//------------------------------------------------------------------------------

#include "LG_internal.h"
#include "LAGraphX.h"

//------------------------------------------------------------------------------
// json.h: JSON parser
//------------------------------------------------------------------------------

#include "json.h"

typedef struct json_value_s  *json_val ;
typedef struct json_object_s *json_obj ; // { json_o start; size length }
typedef struct json_array_s  *json_arr ; // { json_a start; size length }
typedef struct json_string_s *json_str ; // { char *string; size string_size }
typedef struct json_number_s *json_num ; // { char *number; size number_size }

// object element: { char *name; json_val value ; json_o next }
typedef struct json_object_element_s *json_o ;

// array element:  {             json_val value ; json_a next }
typedef struct json_array_element_s  *json_a ;

#define STRMATCH(s,t) (strcmp (s,t) == 0)
#define OK(ok) LG_ASSERT_MSG (ok, LAGRAPH_IO_ERROR, "invalid file")
#define VER(major,minor,sub) (((major)*1000ULL + (minor))*1000ULL + (sub))

//------------------------------------------------------------------------------
// get_int_array_3: get an int array of size 3 from the JSON header
//------------------------------------------------------------------------------

static int get_int_array_3 (json_arr arr, int *x, char *msg)
{
    OK (arr != NULL) ;
    OK (x != NULL) ;
    OK (arr->length == 3) ;
    json_a a = arr->start ;
    json_num num = json_value_as_number (a->value) ;
    OK (num != NULL) ;
    x [0] = (int) strtol (num->number, NULL, 0) ;
    a = a->next ;
    num = json_value_as_number (a->value) ;
    OK (num != NULL) ;
    x [1] = (int) strtol (num->number, NULL, 0) ;
    a = a->next ;
    num = json_value_as_number (a->value) ;
    OK (num != NULL) ;
    x [2] = (int) strtol (num->number, NULL, 0) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// LAGraph_SRead
//------------------------------------------------------------------------------

#undef  LG_FREE_WORK
#define LG_FREE_WORK                                    \
{                                                       \
    if (root != NULL) { free (root) ; }                 \
    root = NULL ;                                       \
    LAGraph_Free ((void **) &json_string, NULL) ;       \
}

#undef  LG_FREE_ALL
#define LG_FREE_ALL                                     \
{                                                       \
    LG_FREE_WORK ;                                      \
    LAGraph_Free ((void **) &collection, NULL) ;        \
    LAGraph_SFreeContents (&Contents, ncontents) ;      \
}

int LAGraph_SRead   // read a set of matrices from a *.lagraph file
(
    FILE *f,                            // file to read from
    // output
    char **collection_handle,           // name of collection
    LAGraph_Contents **Contents_handle, // array of contents
    GrB_Index *ncontents_handle,        // # of items in the Contents array
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    char *json_string = NULL ;
    json_val root = NULL ;
    char *collection = NULL ;
    LAGraph_Contents *Contents = NULL ;
    GrB_Index ncontents = 0 ;

    LG_ASSERT (collection_handle != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (Contents_handle != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (f != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (ncontents_handle != NULL, GrB_NULL_POINTER) ;
    (*collection_handle) = NULL ;
    (*Contents_handle) = NULL ;
    (*ncontents_handle) = 0 ;

    //--------------------------------------------------------------------------
    // load in a json string from the file
    //--------------------------------------------------------------------------

    size_t s = 256, k = 0 ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &json_string, s, sizeof (char),
        msg)) ;
    while (true)
    {
        if (k == s)
        {
            // json_string is full; double its size
            LG_TRY (LAGraph_Realloc ((void **) &json_string, 2*s, s,
                sizeof (char), msg)) ;
            s = 2*s ;
        }
        // get the next character from the file
        int c = fgetc (f) ;
        if (c == EOF || c == '\0')
        {
            // end of the JSON string; remainder is matrix/vector/text data
            json_string [k] = '\0' ;
            break ;
        }
        json_string [k++] = (char) c ;
    }

    //--------------------------------------------------------------------------
    // parse the json string and free it
    //--------------------------------------------------------------------------

    root = json_parse (json_string, k) ;
    LG_ASSERT (root != NULL, GrB_OUT_OF_MEMORY) ;
    LAGraph_Free ((void **) &json_string, NULL) ;

    //--------------------------------------------------------------------------
    // process the JSON header
    //--------------------------------------------------------------------------

    json_obj obj = json_value_as_object (root) ;
    json_o o = obj->start ;
    json_a a = NULL ;
    json_arr arr = NULL ;
    json_str str = NULL ;

    //--------------------------------------------------------------------------
    // check LAGraph version
    //--------------------------------------------------------------------------

    arr = json_value_as_array (o->value) ;
    int lagraph_version [3] ;
    int result = get_int_array_3 (arr, lagraph_version, msg) ;
    OK (result == GrB_SUCCESS) ;
    OK (VER (lagraph_version [0], lagraph_version [1], lagraph_version [2])
     <= VER (LAGRAPH_VERSION_MAJOR, LAGRAPH_VERSION_MINOR,
             LAGRAPH_VERSION_UPDATE)) ;

    //--------------------------------------------------------------------------
    // check GraphBLAS library name and version
    //--------------------------------------------------------------------------

    o = o->next ;
    OK (STRMATCH (o->name->string, "GraphBLAS")) ;
    arr = json_value_as_array (o->value) ;
    OK (arr->length == 2) ;
    a = arr->start ;
    str = json_value_as_string (a->value) ;
    #if LAGRAPH_SUITESPARSE
    OK (STRMATCH (str->string, "SuiteSparse:GraphBLAS")) ;
    #else
    OK (STRMATCH (str->string, "vanilla")) ;
    #endif
    a = a->next ;
    arr = json_value_as_array (a->value) ;

    int graphblas_version [3] ;
    result = get_int_array_3 (arr, graphblas_version, msg) ;
    OK (result == GrB_SUCCESS) ;
    uint64_t library_version =
        VER (graphblas_version [0],
             graphblas_version [1],
             graphblas_version [2]) ;
    #if LAGRAPH_SUITESPARSE
    OK (library_version <= GxB_IMPLEMENTATION) ;
    #else
    OK (library_version <= VER (1,0,0)) ;
    #endif

    //--------------------------------------------------------------------------
    // get the contents and the name of the collection
    //--------------------------------------------------------------------------

    o = o->next ;
    OK (o->value->type == json_type_array) ;
    size_t len = o->name->string_size ;
    LG_TRY (LAGraph_Calloc ((void **) &collection, len+1, sizeof (char), msg)) ;
    strncpy (collection, o->name->string, len) ;

    //--------------------------------------------------------------------------
    // iterate over the contents
    //--------------------------------------------------------------------------

    arr = json_value_as_array (o->value) ;
    OK (arr != NULL) ;
    a = arr->start ;
    len = arr->length ;
    // allocate an Contents array of size len to hold the contents
    LG_TRY (LAGraph_Calloc ((void **) &Contents, len, sizeof (LAGraph_Contents),
        msg)) ;

    for (int i = 0 ; i < len && a != NULL ; i++, a = a->next)
    {

        //----------------------------------------------------------------------
        // get the next item
        //----------------------------------------------------------------------

        if (a->value->type == json_type_null) break ;
        ncontents++ ;
        LAGraph_Contents *Item = &(Contents [i]) ;
        OK (a != NULL) ;
        OK (a->value->type == json_type_object) ;
        json_obj obj = json_value_as_object (a->value) ;
        OK (obj != NULL) ;
        json_o o = obj->start ;
        OK (o != NULL) ;
        OK (o->value->type == json_type_string) ;
        int len = obj->length ;
        OK (len == 3) ;

        //----------------------------------------------------------------------
        // parse the item kind: matrix, vector, or ascii text
        //----------------------------------------------------------------------

        if (STRMATCH (o->name->string, "GrB_Matrix"))
        {
            Item->kind = LAGraph_matrix_kind ;
        }
#if 0
        // todo: handle vectors and text
        else if (STRMATCH (o->name->string, "GrB_Vector"))
        {
            Item->kind = LAGraph_vector_kind ;
        }
        else if (STRMATCH (o->name->string, "text"))
        {
            Item->kind = LAGraph_text_kind ;
        }
#endif
        else
        {
            Item->kind = LAGraph_unknown_kind ;
            OK (false) ;
        }

        //----------------------------------------------------------------------
        // parse the item name
        //----------------------------------------------------------------------

        json_str str = json_value_as_string (o->value) ;
        strncpy (Item->name, str->string, LAGRAPH_MAX_NAME_LEN) ;
        Item->name [LAGRAPH_MAX_NAME_LEN+1] = '\0' ;
        OK (str != NULL) ;
        o = o->next ;
        str = json_value_as_string (o->value) ;
        OK (str != NULL) ;

        //----------------------------------------------------------------------
        // parse the text compression method, or matrix/vector type
        //----------------------------------------------------------------------

#if 0
        // todo: handle text, with optional compression
        if (Item->kind == LAGraph_text_kind)
        {
            // text, uncompressed or compressed
            int c ;
            if      (STRMATCH (str->string, "none"   )) c = -1 ;
            else if (STRMATCH (str->string, "default")) c = 0 ;
            else if (STRMATCH (str->string, "lz4"    )) c = 1000 ;
            else if (STRMATCH (str->string, "lz4hc:0")) c = 2000 ;
            else if (STRMATCH (str->string, "lz4hc:1")) c = 2001 ;
            else if (STRMATCH (str->string, "lz4hc:2")) c = 2002 ;
            else if (STRMATCH (str->string, "lz4hc:3")) c = 2003 ;
            else if (STRMATCH (str->string, "lz4hc:4")) c = 2004 ;
            else if (STRMATCH (str->string, "lz4hc:5")) c = 2005 ;
            else if (STRMATCH (str->string, "lz4hc:6")) c = 2006 ;
            else if (STRMATCH (str->string, "lz4hc:7")) c = 2007 ;
            else if (STRMATCH (str->string, "lz4hc:8")) c = 2008 ;
            else if (STRMATCH (str->string, "lz4hc:9")) c = 2009 ;
            else OK (false) ;
            Item->type_name [0] = '\0' ;    // or set to "char"?
            Item->compression = c ;
        }
        else
#endif
        {
            // serialized matrix or vector
            strncpy (Item->type_name, str->string, LAGRAPH_MAX_NAME_LEN) ;
            Item->type_name [LAGRAPH_MAX_NAME_LEN+1] = '\0' ;
            Item->compression = 0 ;
        }

        //----------------------------------------------------------------------
        // parse the item size
        //----------------------------------------------------------------------

        o = o->next ;
        json_num num = json_value_as_number (o->value) ;
        OK (num != NULL) ;
        Item->blob_size = (GrB_Index) strtoll (num->number, NULL, 0) ;

        //----------------------------------------------------------------------
        // allocate the blob and read it from the file
        //----------------------------------------------------------------------

        LAGRAPH_TRY (LAGraph_Malloc ((void **) &(Item->blob), Item->blob_size,
            sizeof (uint8_t), msg)) ;
        size_t bytes_read = fread (Item->blob, sizeof (uint8_t),
            Item->blob_size, f) ;
        OK (bytes_read == Item->blob_size) ;
    }

    // todo: optional components will be needed for matrices from
    // sparse.tamu.edu (matrix id, author, editor, title, etc)
#if 0
    // optional components
    o = o->next ;
    while (o != NULL)
    {
        printf ("other: [%s]\n", o->name->string) ;
        o = o->next ;
    }
#endif

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LG_FREE_WORK  ;
    (*collection_handle) = collection ;
    (*Contents_handle) = Contents ;
    (*ncontents_handle) = ncontents ;
    return (GrB_SUCCESS) ;
}
