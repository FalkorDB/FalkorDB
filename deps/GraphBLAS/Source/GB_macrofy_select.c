//------------------------------------------------------------------------------
// GB_macrofy_select: construct all macros for select methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_select          // construct all macros for GrB_select
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    // operator:
    const GrB_IndexUnaryOp op,
    GrB_Type atype
)
{

    //--------------------------------------------------------------------------
    // extract the select scode
    //--------------------------------------------------------------------------

    // iso of A and C (2 bits)
    bool C_iso      = GB_RSHIFT (scode, 37, 1) ;
    bool A_iso      = GB_RSHIFT (scode, 36, 1) ;

    // inplace, i/j dependency and flipij (4 bits)
//  int inplace     = GB_RSHIFT (scode, 35, 1) ;
    int i_dep       = GB_RSHIFT (scode, 34, 1) ;
    int j_dep       = GB_RSHIFT (scode, 33, 1) ;
    bool flipij     = GB_RSHIFT (scode, 32, 1) ;

    // op, z = f(x,i,j,y) (5 hex digits)
    int idxop_ecode = GB_RSHIFT (scode, 24, 8) ;
    int zcode       = GB_RSHIFT (scode, 20, 4) ;
    int xcode       = GB_RSHIFT (scode, 16, 4) ;
    int ycode       = GB_RSHIFT (scode, 12, 4) ;

    // types of C and A (2 hex digits)
//  int ccode       = GB_RSHIFT (scode,  8, 4) ;
    int acode       = GB_RSHIFT (scode,  4, 4) ;

    // sparsity structures of C and A (1 hex digit)
//  int csparsity   = GB_RSHIFT (scode,  2, 2) ;
    int asparsity   = GB_RSHIFT (scode,  0, 2) ;

    //--------------------------------------------------------------------------
    // describe the operator
    //--------------------------------------------------------------------------

    GrB_Type xtype, ytype, ztype ;
    const char *xtype_name, *ytype_name, *ztype_name ;

    xtype = (xcode == 0) ? NULL : op->xtype ;
    ytype = (ycode == 0) ? GrB_INT64 : op->ytype ;
    ztype = op->ztype ;
    xtype_name = (xtype == NULL) ? "GB_void" : xtype->name ;
    ytype_name = (ytype == NULL) ? "int64_t" : ytype->name ;
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

    GrB_Type ctype = atype ;    // this may change in the future

    GB_macrofy_typedefs (fp, ctype, atype, NULL,
        xtype, ytype, ztype) ;

    fprintf (fp, "// unary operator types:\n") ;
    GB_macrofy_type (fp, "Z", "_", ztype_name) ;
    GB_macrofy_type (fp, "X", "_", xtype_name) ;
    GB_macrofy_type (fp, "Y", "_", ytype_name) ;

    //--------------------------------------------------------------------------
    // construct macros for the unary operator
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// index unary operator%s:\n",
        flipij ? " (flipped ij)" : "") ;
    GB_macrofy_unop (fp, "GB_IDXUNOP", flipij, idxop_ecode, (GB_Operator) op) ;

    fprintf (fp, "#define GB_DEPENDS_ON_X %d\n", (xtype != NULL) ? 1 : 0) ;
    fprintf (fp, "#define GB_DEPENDS_ON_I %d\n", i_dep) ;
    fprintf (fp, "#define GB_DEPENDS_ON_J %d\n", j_dep) ;
    fprintf (fp, "#define GB_DEPENDS_ON_Y %d\n", (ycode == 0) ? 0 : 1) ;

    char *kind ;
    switch (op->opcode)
    {
        case GB_TRIL_idxunop_code       : kind = "TRIL"      ; break ;
        case GB_TRIU_idxunop_code       : kind = "TRIU"      ; break ;
        case GB_DIAG_idxunop_code       : kind = "DIAG"      ; break ;
        case GB_DIAGINDEX_idxunop_code  : 
        case GB_OFFDIAG_idxunop_code    : kind = "OFFDIAG"   ; break ;
        case GB_ROWINDEX_idxunop_code   : kind = "ROWINDEX"  ; break ;
        case GB_COLINDEX_idxunop_code   : kind = "COLINDEX"  ; break ;
        case GB_COLLE_idxunop_code      : kind = "COLLE"     ; break ;
        case GB_COLGT_idxunop_code      : kind = "COLGT"     ; break ;
        case GB_ROWLE_idxunop_code      : kind = "ROWLE"     ; break ;
        case GB_ROWGT_idxunop_code      : kind = "ROWGT"     ; break ;
        default                         : kind = "ENTRY"     ; break ;
    }
    fprintf (fp, "#define GB_%s_SELECTOR\n", kind) ;

    //--------------------------------------------------------------------------
    // construct the GB_TEST_VALUE_OF_ENTRY(keep,p) macro
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// test if A(i,j) is to be kept:\n") ;

    if (zcode == GB_BOOL_code)
    {
        // no typecasting of z to keep
        fprintf (fp, "#define GB_TEST_VALUE_OF_ENTRY(keep,p) \\\n"
                     "    bool keep ;                        \\\n") ;
        if (xcode == 0)
        { 
            // operator does not depend on x
            fprintf (fp, "    GB_IDXUNOP (keep, , i, j, y) ;\n") ;
        }
        else if (acode == xcode)
        { 
            // operator depends on x, but it has the same type as A
            fprintf (fp, "    GB_IDXUNOP (keep, Ax [%s], i, j, y) ;\n",
                A_iso ? "0" : "p") ;
        }
        else
        { 
            // must typecast from A to x
            fprintf (fp, "    GB_DECLAREA (x) ;                  \\\n"
                         "    GB_GETA (x, Ax, p, ) ;             \\\n"
                         "    GB_IDXUNOP (keep, x, i, j, y) ;\n") ;
        }
    }
    else
    {
        // must typecast z to keep
        int nargs ;
        const char *cast_z_to_bool = GB_macrofy_cast_expression (fp,
            GrB_BOOL, ztype, &nargs) ;
        fprintf (fp, "#define GB_TEST_VALUE_OF_ENTRY(keep,p) \\\n"
                     "    GB_Z_TYPE z ;                      \\\n") ;
        if (xcode == 0)
        { 
            // operator does not depend on x
            fprintf (fp, "    GB_IDXUNOP (z, , i, j, y) ; \\\n") ;
        }
        else if (acode == xcode)
        { 
            // operator depends on x, but it has the same type as A
            fprintf (fp, "    GB_IDXUNOP (z, Ax [%s], i, j, y) ; \\\n",
                A_iso ? "0" : "p") ;
        }
        else
        { 
            // must typecast from A to x
            fprintf (fp, "    GB_DECLAREA (x) ;                  \\\n"
                         "    GB_GETA (x, Ax, p, ) ;             \\\n"
                         "    GB_IDXUNOP (z, x, i, j, y) ;       \\\n") ;
        }
        ASSERT (cast_z_to_bool != NULL) ;
        if (nargs == 3)
        { 
            fprintf (fp, cast_z_to_bool, "    bool keep", "z", "z") ;
        }
        else
        { 
            fprintf (fp, cast_z_to_bool, "    bool keep", "z") ;
        }
    }

    //--------------------------------------------------------------------------
    // construct the GB_SELECT_ENTRY(Cx,pC,Ax,pA) macro
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// copy A(i,j) to C(i,j):\n"
        "#define GB_SELECT_ENTRY(Cx,pC,Ax,pA)") ;
    if (C_iso)
    { 
        // C is iso:  A is iso, or the operator is VALUEEQ
        fprintf (fp, "\n") ;
        fprintf (fp, "#define GB_ISO_SELECT 1\n") ;
    }
    else
    { 
        // C and A are both non-iso
        // this would need to typcase if A and C had different types
        ASSERT (!A_iso) ;
        fprintf (fp, " Cx [pC] = Ax [pA]\n") ;
    }

    //--------------------------------------------------------------------------
    // macros for the Cx array
    //--------------------------------------------------------------------------

    // Cx has the same type as A for now
    fprintf (fp, "\n// C type:\n") ;
    GB_macrofy_type (fp, "C", "_", ctype->name) ;

    //--------------------------------------------------------------------------
    // construct the macros for A
    //--------------------------------------------------------------------------

    GB_macrofy_input (fp, "a", "A", "A", true, xtype,
        atype, asparsity, acode, 0, -1) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"GB_select_shared_definitions.h\"\n") ;
}

