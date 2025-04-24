#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include "json.h"

typedef struct json_value_s  *json_val ;
typedef struct json_object_s *json_obj ; // { json_o start; size length }
typedef struct json_array_s  *json_arr ; // { json_a start; size length }
typedef struct json_string_s *json_str ; // { char *string; size string_size }
typedef struct json_number_s *json_num ; // { char *number; size number_size }
typedef struct json_object_element_s *json_o ; // { char *name; json_val value ; json_o next }
typedef struct json_array_element_s  *json_a ; // {             json_val value ; json_a next }

void json_dump_indent (int depth) ;
bool json_dump_array  (json_val v, int depth) ;
bool json_dump_object (json_val v, int depth) ;
bool json_dump_string (json_val v, int depth) ;
bool json_dump_number (json_val v, int depth) ;

#define MATCH(s1,s2) (strcmp (s1,s2) == 0)
#define ERR(err,msg) if (err) { printf ("%d: ", __LINE__); printf (msg) ; return (false) ; } 
#define OK(ok) ERR(!(ok), "fail") ;

// GraphBLAS.h
#define GxB_MAX_NAME_LEN 128

// in LAGraphX.h
#if LAGRAPH_SUITESPARSE
#define LAGRAPH_MAX_NAME_LEN GxB_MAX_NAME_LEN
#else
#define LAGRAPH_MAX_NAME_LEN 128
#endif

typedef enum
{
    LAGraph_unknown_kind = -1,  // unknown kind
    LAGraph_matrix_kind = 0,    // a serialized GrB_Matrix
    LAGraph_vector_kind = 1,    // a serialized GrB_Vector (SS:GrB only)
    LAGraph_text_kind = 2,      // text (char *), possibly compressed
}
LAGraph_Contents_kind ;

typedef struct
{
    // serialized matrix/vector, or pointer to text, and its size
    void *blob ;
    size_t blob_size ;

    // kind of item: matrix, vector, text, or unknown
    LAGraph_Contents_kind kind ;

    // if kind is text: compression used
    // -1: none, 0: default for library, 1000: LZ4, 200x: LZ4HC:x
    int compression ;

    // name of the object
    char name [LAGRAPH_MAX_NAME_LEN+4] ;

    // if kind is matrix or vector: type name
    char type [LAGRAPH_MAX_NAME_LEN+4] ;
}
LAGraph_Contents ;

bool get_contents (json_a a, LAGraph_Contents *Item) ;
bool get_int_array_3 (json_arr arr, int *x) ;

bool get_contents (json_a a, LAGraph_Contents *Item)
{
    OK (a != NULL) ;
    OK (a->value->type == json_type_object) ;
    json_obj obj = json_value_as_object (a->value) ;
    OK (obj != NULL) ;
    json_o o = obj->start ;
    OK (o != NULL) ;
    OK (o->value->type == json_type_string) ;
    int len = obj->length ;
    OK (len == 3) ;

    if (MATCH (o->name->string, "GrB_Matrix"))
    {
        Item->kind = LAGraph_matrix_kind ;
        printf (">>> matrix\n") ;
    }
    else if (MATCH (o->name->string, "GrB_Vector"))
    {
        Item->kind = LAGraph_vector_kind ;
    }
    else if (MATCH (o->name->string, "text"))
    {
        Item->kind = LAGraph_text_kind ;
        printf (">>> text\n") ;
    }
    else
    {
        Item->kind = LAGraph_unknown_kind ;
        ERR (true, "unknown contents") ;
    }

    Item->blob = NULL ;

    json_num num ;

    json_str str = json_value_as_string (o->value) ;
    strncpy (Item->name, str->string, LAGRAPH_MAX_NAME_LEN) ;
    Item->name [LAGRAPH_MAX_NAME_LEN+1] = '\0' ;
    OK (str != NULL) ;
    printf ("[%s]\n", Item->name) ;
    o = o->next ;
    str = json_value_as_string (o->value) ;
    OK (str != NULL) ;

    if (Item->kind == LAGraph_text_kind)
    {
        // text, uncompressed or compressed
        int c ;
        if      (MATCH (str->string, "none"   )) c = -1 ;
        else if (MATCH (str->string, "default")) c = 0 ;
        else if (MATCH (str->string, "lz4"    )) c = 1000 ;
        else if (MATCH (str->string, "lz4hc:0")) c = 2000 ;
        else if (MATCH (str->string, "lz4hc:1")) c = 2001 ;
        else if (MATCH (str->string, "lz4hc:2")) c = 2002 ;
        else if (MATCH (str->string, "lz4hc:3")) c = 2003 ;
        else if (MATCH (str->string, "lz4hc:4")) c = 2004 ;
        else if (MATCH (str->string, "lz4hc:5")) c = 2005 ;
        else if (MATCH (str->string, "lz4hc:6")) c = 2006 ;
        else if (MATCH (str->string, "lz4hc:7")) c = 2007 ;
        else if (MATCH (str->string, "lz4hc:8")) c = 2008 ;
        else if (MATCH (str->string, "lz4hc:9")) c = 2009 ;
        else ERR (true, "unknown compression") ;
        Item->type [0] = '\0' ;
        Item->compression = c ;
        printf ("[%s]:%d\n", str->string, c) ;
    }
    else
    {
        // serialized matrix or vector
        strncpy (Item->type, str->string, LAGRAPH_MAX_NAME_LEN) ;
        Item->type [LAGRAPH_MAX_NAME_LEN+1] = '\0' ;
        Item->compression = 0 ;
        printf ("(%s)\n", Item->type) ;
    }

    o = o->next ;
    num = json_value_as_number (o->value) ;
    OK (num != NULL) ;
    Item->blob_size = atoi (num->number) ;
    printf ("(%lu)\n", Item->blob_size) ;

    return (true) ;
}

//------------------------------------------------------------------------------
// get_int
//------------------------------------------------------------------------------

bool get_int_array_3 (json_arr arr, int *x)
{
    OK (arr != NULL) ;
    OK (x != NULL) ;
    OK (arr->length == 3) ;
    json_a a = arr->start ;
    json_num num = json_value_as_number (a->value) ;
    OK (num != NULL) ;
    // printf ("[%s.", num->number) ;
    x [0] = atoi (num->number) ;
    a = a->next ;
    num = json_value_as_number (a->value) ;
    OK (num != NULL) ;
    // printf ("%s.", num->number) ;
    x [1] = atoi (num->number) ;
    a = a->next ;
    num = json_value_as_number (a->value) ;
    OK (num != NULL) ;
    // printf ("%s]\n", num->number) ;
    x [2] = atoi (num->number) ;
    return (true) ;
}

//------------------------------------------------------------------------------
// json_dump_indent
//------------------------------------------------------------------------------

void json_dump_indent (int depth)
{
    for (int i = 0 ; i < depth ; i++)
    {
        printf ("....") ;
    }
}

//------------------------------------------------------------------------------
// json_dump_number: dump a json number
//------------------------------------------------------------------------------

bool json_dump_number (json_val v, int depth)
{
    json_dump_indent (depth) ; printf ("number: ") ;
    ERR (v == NULL, "v is NULL\n") ;
    ERR (v->type != json_type_number, "v not number\n") ;
    json_num num = (json_num) v->payload ;
    ERR (num == NULL, "num is NULL\n") ;
    const char *s = (const char *) num->number ;
    printf ("%lu:[%s]\n", num->number_size, s) ;
    return (true) ;
}

//------------------------------------------------------------------------------
// json_dump_string: dump a json string
//------------------------------------------------------------------------------

bool json_dump_string (json_val v, int depth)
{
    json_dump_indent (depth) ; printf ("string:\n") ;
    ERR (v == NULL, "v is NULL\n") ;
    ERR (v->type != json_type_string, "v not string\n") ;
    json_str str = (json_str) v->payload ;
    ERR (str == NULL, "str is NULL\n") ;
    const char *s = (const char *) str->string ;
    printf ("%lu:[%s]\n", str->string_size, s) ;
    return (true) ;
}

//------------------------------------------------------------------------------
// json_dump_value: dump a json value
//------------------------------------------------------------------------------

bool json_dump_value (json_val v, int depth)
{
    ERR (v == NULL, "v is NULL\n") ;
    switch (v->type)
    {
        case json_type_string : return (json_dump_string (v, depth+1)) ;
        case json_type_number : return (json_dump_number (v, depth+1)) ;
        case json_type_object : return (json_dump_object (v, depth+1)) ;
        case json_type_array  : return (json_dump_array (v, depth+1)) ;
        case json_type_true   :
            json_dump_indent (depth+1) ;
            printf ("true\n" ) ;
            break ;
        case json_type_false  :
            json_dump_indent (depth+1) ;
            printf ("false\n") ;
            break ;
        case json_type_null   :
            json_dump_indent (depth+1) ;
            printf ("NULL\n" ) ;
            break ;
        default : ERR (true, "unknown type\n") ; break ;
    }
    return (true) ;
}

//------------------------------------------------------------------------------
// json_dump_array: dump a json array
//------------------------------------------------------------------------------

bool json_dump_array (json_val v, int depth)
{
    ERR (v == NULL, "v is NULL\n") ;
    ERR (v->type != json_type_array, "v not array\n") ;
    json_arr arr = (json_arr) v->payload ;
    ERR (arr == NULL, "array is NULL\n") ;
    size_t len = arr->length ;
    json_dump_indent (depth) ; printf ("array, len: %lu\n", len) ;
    json_a el = arr->start ;
    for (size_t i = 0 ; i < len ; i++)
    {
        ERR (el == NULL, "el is NULL\n") ;
        json_dump_indent (depth) ;
        printf ("array el [%lu]:\n", i) ;
        json_val e = el->value ;
        ERR (e == NULL, "e is NULL\n") ;
        bool ok = json_dump_value (e, depth+1) ;
        ERR (!ok, "e failed\n") ;
        el = el->next ;
    }
    return (true) ;
}

//------------------------------------------------------------------------------
// json_dump_object: dump a json object
//------------------------------------------------------------------------------

// An element in a list of objects is the same as an element in an array,
// except that the element in a list of objects has a name, while the array
// elements do not have their own name.

bool json_dump_object (json_val v, int depth)
{
    ERR (v == NULL, "v null\n") ; ;
    ERR (v->type != json_type_object, "v not object\n") ;
    json_obj obj = (json_obj) v->payload ;
    ERR (obj == NULL, "object is NULL\n") ;
    size_t len = obj->length ;
    json_dump_indent (depth) ;
    printf ("object, len: %lu\n", len) ;
    json_o o = obj->start ;
    for (size_t i = 0 ; i < len ; i++)
    {
        ERR (o == NULL, "element NULL\n") ;
        json_str name = o->name ;
        ERR (name == NULL, "name NULL\n") ;
        json_dump_indent (depth) ;
        printf ("named element: [%lu]:%lu:[%s]\n", i,
            name->string_size, name->string) ;
        json_val e = o->value ;
        ERR (e == NULL, "e NULL\n") ;
        bool ok = json_dump_value (e, depth+1) ;
        ERR (!ok, "e failed\n") ;
        o = o->next ;
    }
    return (true) ;
}

//------------------------------------------------------------------------------
// main: read in a json string from a file and dump it out
//------------------------------------------------------------------------------

int main (void)
{
    // load in a json string from stdin
    size_t s = 256, k = 0 ;
    uint8_t *p = malloc (s) ;
    while (true)
    { 
        if (k == s)
        {
            s = 2*s ;
            p = realloc (p, s) ;
            assert (p != NULL) ;
        }
        int c = getchar ( ) ;
        if (c == EOF || c == '\0')
        {
            p [k] = '\0' ;
            break ;
        }
        else
        {
            p [k++] = (uint8_t) c ;
        }
    }

    // print it out
    printf ("\n=========================================== json_test:\n") ;
    printf ("s %lu k %lu\n", s, k) ;
    printf ("[%s]\n", p) ;
    printf ("entire json strlen %lu\n", strlen (p)) ;

    // parse it and free the string
    json_val root = json_parse (p, k) ;
    bool ok = json_dump_object (root, 0) ;
    assert (ok) ;
    free (p) ;
    printf ("\n#############################\n") ;

    // parse an LAGragh header
    json_obj obj = json_value_as_object (root) ;
    json_o o = obj->start ;
    json_a a = NULL ;
    json_arr arr = NULL ;
    json_str str = NULL ;

    LAGraph_Contents *Contents = NULL ;

    // OK (MATCH (o->name->string, "LAGraph")) ;
    if (MATCH (o->name->string, "LAGraph") && obj->length >= 3)
    {
        // LAGraph version: array of size 3
        // OK (json_dump_value (o->value, 0)) ;
        printf ("LAGraph: ") ;
        arr = json_value_as_array (o->value) ;

        // get an int array of size 3
        int lagraph_version [3] ;
        OK (get_int_array_3 (arr, lagraph_version)) ;
        printf ("(%d.%d.%d)\n", lagraph_version [0],
            lagraph_version [1], lagraph_version [2]) ;

        // GraphBLAS library name and version 
        o = o->next ;
        OK (MATCH (o->name->string, "GraphBLAS")) ;
        // OK (json_dump_value (o->value, 0)) ;
        arr = json_value_as_array (o->value) ;
        OK (arr->length == 2) ;
        a = arr->start ;
        // OK (json_dump_value (a->value, 0)) ;
        str = json_value_as_string (a->value) ;
        printf ("GraphBLAS: %s ", str->string) ;
        a = a->next ;
        arr = json_value_as_array (a->value) ;

        // get an array of size 3
        int graphblas_version [3] ;
        OK (get_int_array_3 (arr, graphblas_version)) ;
        printf ("(%d.%d.%d)\n", graphblas_version [0],
            graphblas_version [1], graphblas_version [2]) ;

        // matrix contents
        o = o->next ;
        printf ("contents: [%s]\n", o->name->string) ;
        OK (o->value->type == json_type_array) ;
        // OK (json_dump_value (o->value, 0)) ;

        // iterate over the contents
        arr = json_value_as_array (o->value) ;
        a = arr->start ;
        int len = arr->length ;
        // allocate an Contents array of size len to hold the contents
        Contents = malloc (len * sizeof (LAGraph_Contents)) ;
        OK (Contents != NULL) ;
        // printf ("size of item: %ld\n", sizeof (LAGraph_Contents)) ;

        int ncontents = 0 ;
        for (int i = 0 ; i < len && a != NULL ; i++)
        {
            if (a->value->type == json_type_null) break ;
            LAGraph_Contents *Item = &(Contents [i]) ;
            OK (get_contents (a, Item)) ;
            ncontents++ ;
            a = a->next ;
        }
        printf ("# of contents: %d\n", ncontents) ;

        // optional components
        o = o->next ;
        while (o != NULL)
        {
            printf ("other: [%s]\n", o->name->string) ;
            OK (json_dump_value (o->value, 0)) ;
            o = o->next ;
        }
    }

    free (root) ;
    printf ("\n=========================================== json_test: OK\n\n") ;
    fprintf (stderr, "test passed\n") ;
    if (Contents != NULL) free (Contents) ;
}

