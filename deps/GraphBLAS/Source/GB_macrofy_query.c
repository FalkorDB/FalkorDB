//------------------------------------------------------------------------------
// GB_macrofy_query: construct GB_jit_query function for a kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_query
(
    FILE *fp,
    const bool builtin, // true if method is all builtin
    GrB_Monoid monoid,  // monoid for reduce or semiring; NULL otherwise
    GB_Operator op0,    // monoid op, select op, unary op, etc
    GB_Operator op1,    // binaryop for a semring
    GrB_Type type0,
    GrB_Type type1,
    GrB_Type type2,
    uint64_t hash       // hash code for the kernel
)
{

    //--------------------------------------------------------------------------
    // create the function header, and query the version
    //--------------------------------------------------------------------------

    fprintf (fp, 
        "GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;\n"
        "GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)\n"
        "{\n"
        "    (*hash) = 0x%016" PRIx64 " ;\n"
        "    v [0] = %d ; v [1] = %d ; v [2] = %d ;\n",
            hash,
            GxB_IMPLEMENTATION_MAJOR,
            GxB_IMPLEMENTATION_MINOR,
            GxB_IMPLEMENTATION_SUB) ;

    //--------------------------------------------------------------------------
    // query the operators
    //--------------------------------------------------------------------------

    // create the definition string for op0
    if (builtin || op0 == NULL || op0->defn == NULL)
    { 
        // op0 does not appear, or is builtin
        fprintf (fp, "    defn [0] = NULL ;\n") ;
    }
    else
    { 
        // op0 is user-defined
        fprintf (fp, "    defn [0] = GB_%s_USER_DEFN ;\n", op0->name) ;
    }

    // create the definition string for op1
    if (builtin || op1 == NULL || op1->defn == NULL)
    { 
        // op1 does not appear, or is builtin
        fprintf (fp, "    defn [1] = NULL ;\n") ;
    }
    else if (op0 == op1)
    { 
        // op1 is user-defined, but the same as op0
        fprintf (fp, "    defn [1] = defn [0] ;\n") ;
    }
    else
    { 
        // op1 is user-defined, and differs from op0
        fprintf (fp, "    defn [1] = GB_%s_USER_DEFN ;\n", op1->name) ;
    }

    //--------------------------------------------------------------------------
    // query the three types
    //--------------------------------------------------------------------------

    GrB_Type types [3] ;
    types [0] = type0 ;
    types [1] = type1 ;
    types [2] = type2 ;
    for (int k = 0 ; k <= 2 ; k++)
    {
        GrB_Type type = types [k] ;
        if (builtin || type == NULL || type->defn == NULL)
        { 
            // types [k] does not appear, or is builtin
            fprintf (fp, "    defn [%d] = NULL ;\n", k+2) ;
        }
        else
        { 
            // see if the type definition already appears
            bool is_unique = true ;
            for (int j = 0 ; j < k ; j++)
            {
                if (type == types [j])
                { 
                    is_unique = false ;
                    fprintf (fp, "    defn [%d] = defn [%d] ;\n", k+2, j+2) ;
                    break ;
                }
            }
            if (is_unique)
            { 
                // this type is unique, and user-defined
                fprintf (fp, "    defn [%d] = GB_%s_USER_DEFN ;\n", k+2,
                    type->name) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // query the monoid identity and terminal values
    //--------------------------------------------------------------------------

    if (monoid != NULL && monoid->hash != 0)
    { 
        // only create the query_monoid method if the monoid is not builtin
        bool has_terminal = (monoid->terminal != NULL) ;
        int zsize = (int) monoid->op->ztype->size ;
        int tsize = (has_terminal) ? zsize : 0 ;
        fprintf (fp,
            "    if (id_size != %d || term_size != %d) return (false) ;\n"
            "    GB_DECLARE_IDENTITY_CONST (zidentity) ;\n"
            "    if (id == NULL || memcmp (id, &zidentity, %d) != 0) "
                     "return (false) ;\n", zsize, tsize, zsize) ;
        if (has_terminal)
        { 
            fprintf (fp,
            "    GB_DECLARE_TERMINAL_CONST (zterminal) ;\n"
            "    if (term == NULL || memcmp (term, &zterminal, %d) != 0) "
                    "return (false) ;\n", tsize) ;
        }
    }
    fprintf (fp, "    return (true) ;\n}\n") ;
}

