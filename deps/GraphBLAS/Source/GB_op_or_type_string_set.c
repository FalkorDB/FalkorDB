//------------------------------------------------------------------------------
// GB_op_or_type_string_set: set the name or defn of a user-defined type or op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"
#include <ctype.h>
#include "GB_jitifyer.h"

GrB_Info GB_op_or_type_string_set
(
    // input:
    bool user_defined,
    bool jitable,
    char *value,
    int field,
    // output:
    char **user_name,
    size_t *user_name_size,
    char *name,
    int32_t *name_len,
    char **defn,
    size_t *defn_size,
    uint64_t *hash
) 
{

    //--------------------------------------------------------------------------
    // quick return for built-in types and operators
    //--------------------------------------------------------------------------

    if (!user_defined)
    { 
        // built-in type or operator
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // set the name or defn of a user-defined type or operator
    //--------------------------------------------------------------------------

    size_t len = strlen (value) ;
    bool compute_hash = false ;

    switch (field)
    {

        case GrB_NAME : 

            return (GB_user_name_set (user_name, user_name_size, value, true)) ;

        case GxB_JIT_C_NAME : 

            if (name [0] != '\0')
            { 
                return (GrB_ALREADY_SET) ;  // GxB_JIT_C_NAME already set
            }

            if (len == 0 || len >= GxB_MAX_NAME_LEN)
            { 
                // invalid name: the name cannot be empty, and the name cannot
                // exceed GxB_MAX_NAME_LEN-1 characters.
                return (GrB_INVALID_VALUE) ;
            }

            // set the name
            strncpy (name, value, GxB_MAX_NAME_LEN-1) ;
            name [GxB_MAX_NAME_LEN-1] = '\0' ;
            (*name_len) = (int32_t) len ;
            // compute the hash if the type defn has also been set
            compute_hash = ((*defn) != NULL) ;
            break ;

        case GxB_JIT_C_DEFINITION : 

            if ((*defn) != NULL)
            { 
                return (GrB_ALREADY_SET) ;  // GxB_JIT_C_DEFINITION already set
            }

            // allocate space for the definition
            (*defn) = GB_MALLOC (len+1, char, defn_size) ;
            if ((*defn) == NULL)
            { 
                // out of memory
                return (GrB_OUT_OF_MEMORY) ;
            }

            // copy the definition into the new operator
            memcpy ((*defn), value, len+1) ;
            // compute the hash if the type name has also been set
            compute_hash = (name [0] != '[') ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // compute the operator hash, if type->name and type->defn are now both set
    //--------------------------------------------------------------------------

    if (compute_hash)
    { 
        // the type name and defn have been set
        (*hash) = GB_jitifyer_hash (name, (*name_len), jitable) ;
    }

    return (GrB_SUCCESS) ;
}

