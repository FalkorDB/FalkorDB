//------------------------------------------------------------------------------
// GB_macrofy_apply: construct all macros for apply methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_apply           // construct all macros for GrB_apply
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t method_code,
    // operator:
        const GB_Operator op,       // unary/index-unary to apply; not binaryop
    GrB_Type ctype,
    GrB_Type atype
)
{

    //--------------------------------------------------------------------------
    // extract the apply method_code
    //--------------------------------------------------------------------------

    // C and A properties (3 hex digits)
    bool Cp_is_32   = GB_RSHIFT (method_code, 44, 1) ;
    bool Ci_is_32   = GB_RSHIFT (method_code, 43, 1) ;
    bool Cj_is_32   = GB_RSHIFT (method_code, 42, 1) ;
    bool Ap_is_32   = GB_RSHIFT (method_code, 41, 1) ;
    bool Aj_is_32   = GB_RSHIFT (method_code, 40, 1) ;
    bool Ai_is_32   = GB_RSHIFT (method_code, 39, 1) ;
    int A_mat       = GB_RSHIFT (method_code, 38, 1) ;
    int A_zombies   = GB_RSHIFT (method_code, 37, 1) ;
    bool A_iso      = GB_RSHIFT (method_code, 36, 1) ;

    // C kind, i/j dependency and flipij (4 bits)
    int C_mat       = GB_RSHIFT (method_code, 35, 1) ;
    int i_dep       = GB_RSHIFT (method_code, 34, 1) ;
    int j_dep       = GB_RSHIFT (method_code, 33, 1) ;
    bool flipij     = GB_RSHIFT (method_code, 32, 1) ;

    // op, z = f(x,i,j,y) (5 hex digits)
    int unop_ecode  = GB_RSHIFT (method_code, 24, 8) ;
//  int zcode       = GB_RSHIFT (method_code, 20, 4) ;
    int xcode       = GB_RSHIFT (method_code, 16, 4) ;
    int ycode       = GB_RSHIFT (method_code, 12, 4) ;

    // types of C and A (2 hex digits)
//  int ccode       = GB_RSHIFT (method_code,  8, 4) ;
    int acode       = GB_RSHIFT (method_code,  4, 4) ;

    // sparsity structures of C and A (1 hex digit)
    int csparsity   = GB_RSHIFT (method_code,  2, 2) ;
    int asparsity   = GB_RSHIFT (method_code,  0, 2) ;

    //--------------------------------------------------------------------------
    // describe the operator
    //--------------------------------------------------------------------------

    GrB_Type xtype, ytype, ztype ;
    const char *xtype_name, *ytype_name, *ztype_name ;

    xtype = (xcode == 0) ? NULL : op->xtype ;
    ytype = (ycode == 0) ? NULL : op->ytype ;
    ztype = op->ztype ;
    xtype_name = (xtype == NULL) ? "void" : xtype->name ;
    ytype_name = (ytype == NULL) ? "void" : ytype->name ;
    ztype_name = ztype->name ;
    if (op->hash == 0)
    { 
        // builtin operator
        fprintf (fp, "// op: (%s%s, %s)\n\n",
            op->name, flipij ? " (flipped ij)" : "", xtype_name) ;
    }
    else
    { 
        // user-defined operator
        fprintf (fp,
            "// op: %s%s, ztype: %s, xtype: %s, ytype: %s\n\n",
            op->name, flipij ? " (flipped ij)" : "",
            ztype_name, xtype_name, ytype_name) ;
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, ctype, (acode == 0) ? NULL : atype, NULL,
        xtype, ytype, ztype, NULL) ;

    fprintf (fp, "// unary operator types:\n") ;
    GB_macrofy_type (fp, "Z", "_", ztype_name) ;
    GB_macrofy_type (fp, "X", "_", xtype_name) ;
    GB_macrofy_type (fp, "Y", "_", ytype_name) ;
    fprintf (fp, "#define GB_DECLAREZ(zwork) %s zwork\n", ztype_name) ;
    fprintf (fp, "#define GB_DECLAREX(xwork) %s xwork\n", xtype_name) ;
    fprintf (fp, "#define GB_DECLAREY(ywork) %s ywork\n", ytype_name) ;

    //--------------------------------------------------------------------------
    // construct macros for the unary operator
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// unary operator%s:\n", flipij ? " (flipped ij)" : "") ;
    GB_macrofy_unop (fp, "GB_UNARYOP", flipij, unop_ecode, op) ;

    int y_dep = (ytype != NULL) ? 1 : 0 ;
    fprintf (fp, "#define GB_DEPENDS_ON_X %d\n", (xtype != NULL) ? 1 : 0) ;
    fprintf (fp, "#define GB_DEPENDS_ON_Y %d\n", y_dep) ;
    fprintf (fp, "#define GB_DEPENDS_ON_I %d\n", i_dep) ;
    fprintf (fp, "#define GB_DEPENDS_ON_J %d\n", j_dep) ;

    bool no_typecast_of_A = (atype == xtype) || (xtype == NULL) ;

    // Cx [pC] = op (Ax [pA], i, j, y)
    char *pA = A_iso ? "0" : "pA" ;
    char *i = i_dep  ? "i" : " " ;
    char *j = j_dep  ? "j" : " " ;
    char *y = y_dep  ? "y" : " " ;
    fprintf (fp, "#define GB_UNOP(Cx,pC,Ax,pA,A_iso,i,j,y)") ;
    if (ctype == ztype && no_typecast_of_A)
    { 
        // no typecasting
        if (op->opcode == GB_IDENTITY_unop_code)
        { 
            // identity operator, no typecasting
            fprintf (fp, " Cx [pC] = Ax [%s]\n", pA) ;
        }
        else
        { 
            // any operator, no typecsting
            fprintf (fp, " GB_UNARYOP (Cx [pC], Ax [%s], %s, %s, %s)\n",
                pA, i, j, y) ;
        }
    }
    else if (ctype == ztype)
    { 
        // aij = (xtype) Ax [pC] must be typecast, but not z
        fprintf (fp, " \\\n"
            "{                                              \\\n"
            "    GB_DECLAREA (aij) ;                        \\\n"
            "    GB_GETA (aij, Ax, %s, ) ;                  \\\n"
            "    GB_UNARYOP (Cx [pC], aij, %s, %s, %s) ;    \\\n"
            "}\n", pA, i, j, y) ;
    }
    else if (no_typecast_of_A)
    { 
        // Cx [pC] = (ctype) z must be typecast, but not aij
        fprintf (fp, " \\\n"
            "{                                              \\\n"
            "    GB_DECLAREZ (z) ;                          \\\n"
            "    GB_UNARYOP (z, aij, Ax [%s], %s, %s, %s) ; \\\n"
            "    GB_PUTC (z, Cx, pC) ;                      \\\n"
            "}\n", pA, i, j, y) ;
    }
    else
    { 
        // both must be typecast
        fprintf (fp, " \\\n"
            "{                                      \\\n"
            "    GB_DECLAREA (aij) ;                \\\n"
            "    GB_GETA (aij, Ax, %s, ) ;          \\\n"
            "    GB_DECLAREZ (z) ;                  \\\n"
            "    GB_UNARYOP (z, aij, %s, %s, %s) ;  \\\n"
            "    GB_PUTC (z, Cx, pC) ;              \\\n"
            "}\n", pA, i, j, y) ;
    }

    //--------------------------------------------------------------------------
    // macros for the C array or matrix
    //--------------------------------------------------------------------------

    if (C_mat)
    { 
        // C = op(A) where C is a matrix
        GB_macrofy_output (fp, "c", "C", "C", ctype, ztype, csparsity, false,
            false, Cp_is_32, Cj_is_32, Ci_is_32) ;
    }
    else
    { 
        // Cx = op(A) where Cx is an array of type ztype
        ASSERT (ctype == ztype) ;
        fprintf (fp, "\n// C type:\n") ;
        GB_macrofy_type (fp, "C", "_", ctype->name) ;
        GB_macrofy_bits (fp, "C", Cp_is_32, Cj_is_32, Ci_is_32) ;
    }

    //--------------------------------------------------------------------------
    // construct the macros for A array or matrix
    //--------------------------------------------------------------------------

    if (A_mat)
    {
        // C or Cx = op(A) for a matrix A
        GB_macrofy_input (fp, "a", "A", "A", true, xtype,
            atype, asparsity, acode, A_iso, A_zombies,
            Ap_is_32, Aj_is_32, Ai_is_32) ;
    }
    else
    {
        // Cx = op(Ax) for arrays Cx and Ax (no typecast of A to xtype)
        ASSERT (no_typecast_of_A) ;
        fprintf (fp, "\n// A type:\n") ;
        GB_macrofy_type (fp, "A", "_", atype->name) ;
    }

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"include/GB_kernel_shared_definitions.h\"\n") ;
}

