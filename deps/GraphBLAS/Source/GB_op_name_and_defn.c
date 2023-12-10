//------------------------------------------------------------------------------
// GB_op_name_and_defn: construct name and defn of a user-defined op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method constructs the name and defn of a user-defined op (unary,
// binary, or indexunary).  It also constructs the op->hash for the jit,
// which is never zero for this method.

#include "GB.h"
#include <ctype.h>
#include "GB_jitifyer.h"

GrB_Info GB_op_name_and_defn
(
    // output
    char *op_name,              // op->name of the GrB operator struct
    int32_t *op_name_len,       // op->name_len
    uint64_t *op_hash,          // op->hash
    char **op_defn,             // op->defn
    size_t *op_defn_size,       // op->defn_size
    // input
    const char *input_name,     // user-provided name, may be NULL
    const char *input_defn,     // user-provided name, may be NULL
    bool user_op,               // if true, a user-defined op
    bool jitable                // if true, the op can be JIT'd
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (op_name != NULL) ;
    ASSERT (op_defn != NULL) ;
    ASSERT (op_defn_size != NULL) ;
    (*op_defn) = NULL ;
    (*op_defn_size) = 0 ;

    //--------------------------------------------------------------------------
    // get the name of the operator
    //--------------------------------------------------------------------------

    // note: this can get a mangled name; see the BF methods in LAGraph

    memset (op_name, 0, GxB_MAX_NAME_LEN) ;
    if (input_name != NULL)
    {
        // copy the input_name
        strncpy (op_name, input_name, GxB_MAX_NAME_LEN-1) ;
    }

    // ensure op_name is null-terminated
    op_name [GxB_MAX_NAME_LEN-1] = '\0' ;

    // get the operator name length (zero if no name given)
    (*op_name_len) = (int32_t) strlen (op_name) ;

    // a user-defined op can only be JIT'd if it has a name and defn.
    // a new builtin op (created by GB_reduce_to_vector) can always be JIT'd.
    (*op_hash) = GB_jitifyer_hash (op_name, (*op_name_len),
        jitable && (!user_op || (*op_name_len) > 0)) ;

    //--------------------------------------------------------------------------
    // get the definition of the operator, if present
    //--------------------------------------------------------------------------

    char *defn = NULL ;
    size_t defn_size = 0 ;
    if (input_defn != NULL)
    { 
        // determine the string length of the definition
        size_t defn_len = strlen (input_defn) ;

        // allocate space for the definition
        defn = GB_MALLOC (defn_len+1, char, &defn_size) ;
        if (defn == NULL)
        { 
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }

        // copy the definition into the new operator
        memcpy (defn, input_defn, defn_len+1) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*op_defn) = defn ;
    (*op_defn_size) = defn_size ;
    return (GrB_SUCCESS) ;
}

