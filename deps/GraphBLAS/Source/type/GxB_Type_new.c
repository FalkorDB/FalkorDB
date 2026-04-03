//------------------------------------------------------------------------------
// GxB_Type_new: create a new user-defined type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_Type_new is like GrB_Type_new, except that it gives the user application
// a mechanism for providing a unique name of the type and the C definition of
// the type.  Both are provided as null-terminated strings.

// When the name of the user type is known, it can be returned to the user
// application when querying the type of a GrB_Matrix, GrB_Vector, GrB_Scalar,
// or a serialized blob.

// If GrB_Type_new is used in SuiteSparse:GraphBLAS in its macro form, as
// GrB_Type_new (&t, sizeof (myctype)), then the type_name is extracted as the
// string "myctype".  This type_name can then be returnd by
// GxB_Matrix_type_name, GxB_deserialize_type_name, etc.

// This is not used for built-in types.  Those are created statically.

// Example:

//  GxB_Type_new (&MyQtype, sizeof (myquaternion), "myquaternion",
//      "typedef struct { float x [4][4] ; int color ; } myquaternion ;") ;

// The type_name and type_defn are optional and may by NULL, but they are
// required for the JIT.  If the type size is passed in as zero, it means the
// size is unknown; in this case, the type size is determined via the JIT.
// If the type size is zero but the JIT is disabled, of the two strings are not
// provided, then an error is returned (GrB_INVALID_VALUE). 

#include "GB.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GxB_Type_new
(
    GrB_Type *type,             // handle of user type to create
    size_t sizeof_type,         // size of the user type
    const char *type_name,      // name of the user type
    const char *type_defn       // typedef of the C type (any length)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (type) ;
    GB_BURBLE_START ("GxB_Type_new") ;

    GrB_Info info ;
    (*type) = NULL ;

    if (sizeof_type == 0 && (type_defn == NULL || type_name == NULL))
    { 
        // the JIT is required to determine size of the type, but this
        // requires two valid strings: the type name and the type definition
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // create the type
    //--------------------------------------------------------------------------

    // allocate the type
    size_t header_size ;
    GrB_Type t = GB_MALLOC_MEMORY (1, sizeof (struct GB_Type_opaque),
        &header_size) ;
    if (t == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    // initialize the type
    t->header_size = header_size ;
    t->user_name = NULL ;
    t->user_name_size = 0 ;
    t->size = sizeof_type ;
    t->code = GB_UDT_code ;         // user-defined type
    memset (t->name, 0, GxB_MAX_NAME_LEN) ;   // no name yet
    t->defn = NULL ;                // no definition yet
    t->defn_size = 0 ;
    t->print_function = NULL ;      // no function to print type

    //--------------------------------------------------------------------------
    // get the name
    //--------------------------------------------------------------------------

    if (type_name != NULL)
    {
        // copy the type_name into the working name
        strncpy (t->name, type_name, GxB_MAX_NAME_LEN-1) ;
    }

    // ensure t->name is null-terminated
    t->name [GxB_MAX_NAME_LEN-1] = '\0' ;

    // get the type name length and hash the name
    t->name_len = (int32_t) strlen (t->name) ;
    // type can be JIT'd only if it has a name and defn
    t->hash = GB_jitifyer_hash (t->name, t->name_len,
        (t->name_len > 0 && type_defn != NULL)) ;

    //--------------------------------------------------------------------------
    // get the typedef, if present
    //--------------------------------------------------------------------------

    if (type_defn != NULL)
    { 
        // determine the string length of the typedef
        size_t defn_len = strlen (type_defn) ;

        // allocate space for the typedef
        t->defn = GB_MALLOC_MEMORY (defn_len+1, sizeof (char),
            &(t->defn_size)) ;
        if (t->defn == NULL)
        { 
            // out of memory
            GB_FREE_MEMORY (&t, header_size) ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        // copy the typedef into the new type
        memcpy (t->defn, type_defn, defn_len+1) ;
    }

    // the type is valid, except perhaps for the typesize
    t->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // determine the type size via the JIT, if necessary
    //--------------------------------------------------------------------------

    if (sizeof_type == 0)
    { 
        info = GB_user_type_jit (&sizeof_type, t) ;
        if (info != GrB_SUCCESS)
        { 
            // unable to determine the type size
            GrB_Type_free (&t) ;
            // If the JIT fails, it returns GrB_NO_VALUE or GxB_JIT_ERROR,
            // Convert GrB_NO_VALUE to GrB_INVALID_VALUE (the size of the type
            // is 0 and cannot be determined by the JIT).
            return (info == GrB_NO_VALUE ? GrB_INVALID_VALUE : info) ;
        }
        t->size = sizeof_type ;
    }

    //--------------------------------------------------------------------------
    // typesize is limited on MS Visual Studio
    //--------------------------------------------------------------------------

    #if ( ! GB_HAS_VLA )
    {
        // Microsoft Visual Studio does not support VLAs allocating
        // automatically on the stack.  These arrays are used for scalar values
        // for a given type.  If VLA is not supported, user-defined types can
        // be no larger than GB_VLA_MAXSIZE.
        if (sizeof_type > GB_VLA_MAXSIZE)
        {
            GrB_Type_free (&t) ;
            return (GrB_INVALID_VALUE) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*type) = t ;
    ASSERT_TYPE_OK (t, "new user-defined type", GB0) ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

