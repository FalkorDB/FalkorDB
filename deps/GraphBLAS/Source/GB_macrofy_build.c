//------------------------------------------------------------------------------
// GB_macrofy_build: construct all macros for GB_build methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_build           // construct all macros for GB_build
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t build_code,        // unique encoding of the entire problem
    GrB_BinaryOp dup,           // dup binary operator to macrofy
    GrB_Type ttype,             // type of Tx
    GrB_Type stype              // type of Sx
)
{

    //--------------------------------------------------------------------------
    // extract the build_code
    //--------------------------------------------------------------------------

    // dup, z = f(x,y) (5 hex digits)
    int dup_ecode = GB_RSHIFT (build_code, 20, 8) ;
//  int zcode     = GB_RSHIFT (build_code, 16, 4) ;
//  int xcode     = GB_RSHIFT (build_code, 12, 4) ;
//  int ycode     = GB_RSHIFT (build_code,  8, 4) ;

    // types of S and T (2 hex digits)
//  int tcode     = GB_RSHIFT (build_code, 4, 4) ;
//  int scode     = GB_RSHIFT (build_code, 0, 4) ;

    //--------------------------------------------------------------------------
    // describe the operator
    //--------------------------------------------------------------------------

    ASSERT_BINARYOP_OK (dup, "dup for macrofy build", GB0) ;

    GrB_Type xtype = dup->xtype ;
    GrB_Type ytype = dup->ytype ;
    GrB_Type ztype = dup->ztype ;
    const char *xtype_name = xtype->name ;
    const char *ytype_name = ytype->name ;
    const char *ztype_name = ztype->name ;
    const char *ttype_name = ttype->name ;
    const char *stype_name = stype->name ;
    if (dup->hash == 0)
    { 
        // builtin operator
        fprintf (fp, "// op: (%s, %s)\n\n", dup->name, xtype_name) ;
    }
    else
    { 
        // user-defined operator, or created by GB_build
        fprintf (fp,
            "// op: %s%s, ztype: %s, xtype: %s, ytype: %s\n\n",
            (dup->opcode == GB_SECOND_binop_code) ? "2nd_" : "",
            dup->name, ztype_name, xtype_name, ytype_name) ;
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, stype, ttype, NULL, xtype, ytype, ztype) ;

    fprintf (fp, "// binary dup operator types:\n") ;
    GB_macrofy_type (fp, "Z", "_", ztype_name) ;
    GB_macrofy_type (fp, "X", "_", xtype_name) ;
    GB_macrofy_type (fp, "Y", "_", ytype_name) ;

    fprintf (fp, "\n// S and T data types:\n") ;
    GB_macrofy_type (fp, "T", "_", ttype_name) ;
    GB_macrofy_type (fp, "S", "_", stype_name) ;

    //--------------------------------------------------------------------------
    // construct macros for the binary operator
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// binary dup operator:\n") ;
    GB_macrofy_binop (fp, "GB_DUP", false, true, false, dup_ecode, false, dup,
        NULL, NULL) ;

    fprintf (fp, "\n// build copy/dup methods:\n") ;

    // no typecasting if all 5 types are the same
    bool nocasting = (ttype == stype) &&
        (ttype == xtype) && (ttype == ytype) && (ttype == ztype) ;

    if (nocasting)
    { 

        //----------------------------------------------------------------------
        // GB_BLD_COPY: Tx [p] = Sx [k]
        //----------------------------------------------------------------------

        fprintf (fp, "#define GB_BLD_COPY(Tx,p,Sx,k) Tx [p] = Sx [k]\n") ;

        //----------------------------------------------------------------------
        // GB_BLD_DUP:  Tx [p] += Sx [k]
        //----------------------------------------------------------------------

        fprintf (fp, "#define GB_BLD_DUP(Tx,p,Sx,k)") ;
        if (dup->opcode != GB_FIRST_binop_code)
        { 
            fprintf (fp, " GB_UPDATE (Tx [p], Sx [k])") ;
        }
        fprintf (fp, "\n") ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // GB_BLD_COPY: Tx [p] = (cast) Sx [k]
        //----------------------------------------------------------------------

        int nargs_s_to_t, nargs_s_to_y, nargs_t_to_x, nargs_z_to_t ;
        const char *cast_s_to_t =
            GB_macrofy_cast_expression (fp, ttype, stype, &nargs_s_to_t) ;
        const char *cast_s_to_y =
            GB_macrofy_cast_expression (fp, ytype, stype, &nargs_s_to_y) ;
        const char *cast_t_to_x =
            GB_macrofy_cast_expression (fp, xtype, ttype, &nargs_t_to_x) ;
        const char *cast_z_to_t =
            GB_macrofy_cast_expression (fp, ttype, ztype, &nargs_z_to_t) ;

        fprintf (fp, "#define GB_BLD_COPY(Tx,p,Sx,k)") ;
        if (cast_s_to_t == NULL)
        { 
            fprintf (fp, " Tx [p] = (%s) Sx [k]", ttype_name) ;
        }
        else if (nargs_s_to_t == 3)
        { 
            fprintf (fp, cast_s_to_t, " Tx [p]", "Sx [k]", "Sx [k]") ;
        }
        else
        { 
            fprintf (fp, cast_s_to_t, " Tx [p]", "Sx [k]") ;
        }
        fprintf (fp, "\n") ;

        //----------------------------------------------------------------------
        // GB_BLD_DUP:  Tx [p] += Sx [k], with typecasting
        //----------------------------------------------------------------------

        fprintf (fp, "#define GB_BLD_DUP(Tx,p,Sx,k) \\\n") ;

        // ytype y = (ytype) Sx [k] ;
        fprintf (fp, "    %s ", ytype_name) ;
        if (cast_s_to_y == NULL)
        { 
            fprintf (fp, "y = (%s) Sx [k]", ytype_name) ;
        }
        else if (nargs_s_to_y == 3)
        { 
            fprintf (fp, cast_s_to_y, "y", "Sx [k]", "Sx [k]") ;
        }
        else
        { 
            fprintf (fp, cast_s_to_y, "y", "Sx [k]") ;
        }
        fprintf (fp, " ; \\\n") ;

        // xtype x = (xtype) Tx [p] ;
        fprintf (fp, "    %s ", xtype_name) ;
        if (cast_t_to_x == NULL)
        { 
            fprintf (fp, "x = (%s) Tx [p]", xtype_name) ;
        }
        else if (nargs_t_to_x == 3)
        { 
            fprintf (fp, cast_t_to_x, "x", "Tx [p]", "Tx [p]") ;
        }
        else
        { 
            fprintf (fp, cast_t_to_x, "x", "Tx [p]") ;
        }
        fprintf (fp, " ; \\\n") ;

        // ztype z = dup (x,y) ;
        fprintf (fp, "    %s z ; \\\n", ztype_name) ;
        fprintf (fp, "    GB_DUP (z, x, y) ; \\\n") ;

        // Tx [p] = (ttype) z ;
        if (cast_z_to_t == NULL)
        { 
            fprintf (fp, "    Tx [p] = (%s) z", ttype_name) ;
        }
        else if (nargs_z_to_t == 3)
        { 
            fprintf (fp, cast_z_to_t, "    Tx [p]", "z", "z") ;
        }
        else
        { 
            fprintf (fp, cast_z_to_t, "    Tx [p]", "z") ;
        }
        fprintf (fp, " ;\n") ;
    }

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"GB_kernel_shared_definitions.h\"\n") ;
}

