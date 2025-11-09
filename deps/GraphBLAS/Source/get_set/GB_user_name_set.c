//------------------------------------------------------------------------------
// GB_user_name_set: set the user_name of an object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

GrB_Info GB_user_name_set
(
    // input/output
    char **object_user_name,        // user_name of the object
    size_t *object_user_name_size,  // user_name_size of the object
    // input
    const char *new_name,           // new name for the object
    const bool only_once            // if true, the name of the object can
                                    // only be set once
)
{ 

    if (only_once && (*object_user_name) != NULL)
    { 
        // types, operators, monoids, and semirings can have their GrB_NAME
        // set at most once
        return (GrB_ALREADY_SET) ;
    }

    // free the object user_name, if it already exists
    GB_FREE_MEMORY (object_user_name, (*object_user_name_size)) ;
    (*object_user_name_size) = 0 ;

    // get the length of the new name
    size_t len = strlen (new_name) ;
    if (len == 0)
    { 
        // no new name; leave the object unnamed
        return (GrB_SUCCESS) ;
    }

    // allocate the new name
    size_t user_name_size ;
    char *user_name = GB_MALLOC_MEMORY (len + 1, sizeof (char),
        &user_name_size) ;
    if (user_name == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    // set the new user_name
    strcpy (user_name, new_name) ;
    (*object_user_name) = user_name ;
    (*object_user_name_size) = user_name_size ;
    return (GrB_SUCCESS) ;
}

