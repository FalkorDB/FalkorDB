//------------------------------------------------------------------------------
// GB_macrofy_build: construct all macros for GB_build methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_build           // construct all macros for GB_build
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t method_code,       // unique encoding of the entire problem
    GrB_BinaryOp dup,           // dup binary operator to macrofy
    GrB_Type ttype,             // type of Tx
    GrB_Type stype              // type of Sx
)
{

    //--------------------------------------------------------------------------
    // extract the method_code
    //--------------------------------------------------------------------------

    // 32/64 bit (4 bits, 1 hex digit)
    int Ti_is_32  = GB_RSHIFT (method_code, 31, 1) ;
    int I_is_32   = GB_RSHIFT (method_code, 30, 1) ;
    int K_is_32   = GB_RSHIFT (method_code, 29, 1) ;
    int K_is_null = GB_RSHIFT (method_code, 28, 1) ;

    // dup, z = f(x,y) (6 hex digits)
    int no_dupl   = GB_RSHIFT (method_code, 27, 1) ;
//  int dup_code  = GB_RSHIFT (method_code, 20, 6) ;
//  int zcode     = GB_RSHIFT (method_code, 16, 4) ;
    int xcode     = GB_RSHIFT (method_code, 12, 4) ;
//  int ycode     = GB_RSHIFT (method_code,  8, 4) ;

    // types of S and T (2 hex digits)
//  int tcode     = GB_RSHIFT (method_code, 4, 4) ;
//  int scode     = GB_RSHIFT (method_code, 0, 4) ;

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
    GB_Opcode dup_opcode = dup->opcode ;
    if (xcode == GB_BOOL_code)  // && (ycode == GB_BOOL_code)
    { 
        // rename the operator
        dup_opcode = GB_boolean_rename (dup_opcode) ;
    }

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
            (dup_opcode == GB_SECOND_binop_code) ? "2nd_" : "",
            dup->name, ztype_name, xtype_name, ytype_name) ;
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, stype, ttype, NULL, xtype, ytype, ztype, NULL) ;

    fprintf (fp, "// binary dup operator types:\n") ;
    GB_macrofy_type (fp, "Z", "_", ztype_name) ;
    GB_macrofy_type (fp, "X", "_", xtype_name) ;
    GB_macrofy_type (fp, "Y", "_", ytype_name) ;

    fprintf (fp, "\n// Sx and Tx data types:\n") ;
    GB_macrofy_type (fp, "Tx", "_", ttype_name) ;
    GB_macrofy_type (fp, "Sx", "_", stype_name) ;

    //--------------------------------------------------------------------------
    // construct macros for the binary operator
    //--------------------------------------------------------------------------

    int dup_ecode ;
    GB_enumify_binop (&dup_ecode, dup_opcode, xcode, false, false) ;

    fprintf (fp, "\n// binary dup operator:\n") ;
    GB_macrofy_binop (fp, "GB_DUP", false, false, true, false, false,
        dup_ecode, false, dup, NULL, NULL, NULL) ;

    if (dup_opcode == GB_FIRST_binop_code)
    {
        fprintf (fp, "#define GB_DUP_IS_FIRST\n") ;
    }

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
        if (dup_opcode != GB_FIRST_binop_code)
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
    // 32/64 integer arrays
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// 32/64 integer types:\n") ;
    fprintf (fp, "#define GB_Ti_TYPE %s\n", Ti_is_32 ? "int32_t" : "int64_t") ;
    fprintf (fp, "#define GB_Ti_BITS %d\n", Ti_is_32 ? 32 : 64) ;
    fprintf (fp, "#define GB_I_TYPE  %s\n", I_is_32  ? "uint32_t":"uint64_t") ;
    fprintf (fp, "#define GB_K_TYPE  %s\n", K_is_32  ? "uint32_t":"uint64_t") ;
    fprintf (fp, "#define GB_K_WORK(k) %s\n", K_is_null ? "k" : "K_work [k]") ;
    fprintf (fp, "#define GB_K_IS_NULL %d\n", K_is_null) ;
    fprintf (fp, "#define GB_NO_DUPLICATES %d\n", no_dupl) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"include/GB_kernel_shared_definitions.h\"\n") ;
}

