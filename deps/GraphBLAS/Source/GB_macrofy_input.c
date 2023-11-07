//------------------------------------------------------------------------------
// GB_macrofy_input: construct a macro to load values from an input matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The macro, typically called GB_GETA or GB_GETB, also does typecasting.

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_input
(
    FILE *fp,
    // input:
    const char *aname,      // name of the scalar aij = ...
    const char *Amacro,     // name of the macro is GB_GET*(Amacro)
    const char *Aname,      // name of the input matrix (typically A or B)
    bool do_matrix_macros,  // if true, do the matrix macros
    GrB_Type a2type,        // type of aij after casting to x or y of f(x,y)
    GrB_Type atype,         // type of the input matrix
    int asparsity,          // sparsity format of the input matrix
    int acode,              // type code of the input (0 if pattern,
                            // 15 if A is NULL)
    int A_iso_code,         // 1 if A is iso
    int azombies            // 1 if A has zombies, 0 if A has no zombies;
                            // -1 if the macro should not be created.
)
{

    //--------------------------------------------------------------------------
    // construct the matrix status macros: pattern, iso, typename
    //--------------------------------------------------------------------------

    if (acode == 15)
    { 
        // quick return: no input matrix (for binary op with bind 1st or 2nd)
        return ;
    }

    int A_is_pattern = ((acode == 0) ? 1 : 0) || (atype == NULL) ;

    if (do_matrix_macros)
    {
        GB_macrofy_sparsity (fp, Aname, asparsity) ;
        GB_macrofy_nvals (fp, Aname, asparsity, A_iso_code) ;
        fprintf (fp, "#define GB_%s_ISO %d\n", Aname, A_iso_code) ;
        if (azombies >= 0)
        { 
            // if negative, do not create the macro at all.  Typically this
            // means A has no zombies.  For GrB_wait, using GB_select to
            // delete zombies, it means A always has zombies.
            fprintf (fp, "#define GB_A_HAS_ZOMBIES %d\n", azombies) ;
        }
        if (A_is_pattern)
        { 
            // values of A are not accessed
            fprintf (fp, "#define GB_%s_IS_PATTERN 1\n", Aname) ;
            GB_macrofy_type (fp, Aname, "_", "void") ;
        }
        else
        { 
            GB_macrofy_type (fp, Aname, "_", atype->name) ;
        }
    }

    //--------------------------------------------------------------------------
    // construct the macros to declare scalars and get values from the matrix
    //--------------------------------------------------------------------------

    if (A_is_pattern || a2type == NULL)
    {

        //----------------------------------------------------------------------
        // no need to access the values of A
        //----------------------------------------------------------------------

        if (do_matrix_macros)
        { 
            // aij is not needed as input to the operator
            GB_macrofy_type (fp, Aname, "2", "void") ;
        }

        fprintf (fp, "#define GB_DECLARE%s(%s)\n", Amacro, aname) ;
        fprintf (fp, "#define GB_GET%s(%s,%sx,p,iso)\n", Amacro, aname, Aname) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // construct the scalar/workspace declaration macros
        //----------------------------------------------------------------------

        // Declare a scalar or work array.  For example, suppose A has type
        // double, and the x input to an unflipped operator has type float.  To
        // declare a simple scalar or work array:

        //      GB_DECLAREA (aij) ;
        //      GB_DECLAREA (w [32]) ;

        // becomes:

        //      float aij ;
        //      float w [32] ;

        if (do_matrix_macros)
        { 
            GB_macrofy_type (fp, Aname, "2", a2type->name) ;
        }

        fprintf (fp, "#define GB_DECLARE%s(%s) %s %s\n",
            Amacro, aname, a2type->name, aname) ;

        //----------------------------------------------------------------------
        // construct the GB_GETA macro
        //----------------------------------------------------------------------

        // #define GB_GETA(a,Ax,p,iso) a = (a2type) Ax [iso ? 0 : p]
        // to load a value from the A matrix, and typecast it to the scalar a
        // of type a2type.  Note that the iso status is baked into the macro,
        // since the kernel will be jitified for that particular iso status.
        // If two cases are identical except for the iso status of an input
        // matrix, two different kernels will be constructed and compiled.

        // For example, to load the scalar aij (with a2type float, in the
        // example above), from the matrix A of type double:

        //      GB_GETA (aij,Ax,p,iso) ;

        // becomes:

        //      aij = ((float) (Ax [p])) ;

        // or, if A is iso:

        //      aij = ((float) (Ax [0])) ;

        #define SLEN 256
        char macro_name [SLEN+1], xargs [SLEN+1], xexpr [SLEN+1] ;
        snprintf (macro_name, SLEN, "GB_GET%s", Amacro) ;
        snprintf (xargs, SLEN, "%sx,p,iso", Aname) ;
        snprintf (xexpr, SLEN, A_iso_code ? "%sx [0]" : "%sx [p]", Aname) ;
        GB_macrofy_cast_input (fp, macro_name, aname, xargs, xexpr, a2type,
            atype) ;
    }
}
