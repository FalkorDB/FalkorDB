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

// The type is allocated in header arena determined by the current Context.

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
    int header_arena = GB_Context_header_arena ( ) ;
    return (GxB_Type_new_arena (type, sizeof_type, type_name, type_defn,
        header_arena)) ;
}

