//------------------------------------------------------------------------------
// GB_macrofy_ewise: construct all macros for ewise methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_ewise           // construct all macros for GrB_eWise
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t method_code,
    uint64_t kcode,
    GrB_BinaryOp binaryop,      // binaryop to macrofy
    GrB_Type ctype,
    GrB_Type atype,             // NULL for apply bind1st
    GrB_Type btype              // NULL for apply bind2nd
)
{

    //--------------------------------------------------------------------------
    // extract the ewise method_code
    //--------------------------------------------------------------------------

    // C, M, A, B: 32/64 (3 hex digits)
    bool Cp_is_32   = GB_RSHIFT (method_code, 59, 1) ;
    bool Cj_is_32   = GB_RSHIFT (method_code, 58, 1) ;
    bool Ci_is_32   = GB_RSHIFT (method_code, 57, 1) ;

    bool Mp_is_32   = GB_RSHIFT (method_code, 56, 1) ;
    bool Mj_is_32   = GB_RSHIFT (method_code, 55, 1) ;
    bool Mi_is_32   = GB_RSHIFT (method_code, 54, 1) ;

    bool Ap_is_32   = GB_RSHIFT (method_code, 53, 1) ;
    bool Aj_is_32   = GB_RSHIFT (method_code, 52, 1) ;
    bool Ai_is_32   = GB_RSHIFT (method_code, 51, 1) ;

    bool Bp_is_32   = GB_RSHIFT (method_code, 50, 1) ;
    bool Bj_is_32   = GB_RSHIFT (method_code, 49, 1) ;
    bool Bi_is_32   = GB_RSHIFT (method_code, 48, 1) ;

    // C in, A, and B iso-valued (1 hex digit)
    bool C_in_iso   = GB_RSHIFT (method_code, 46, 1) ;
    bool A_iso      = GB_RSHIFT (method_code, 45, 1) ;
    bool B_iso      = GB_RSHIFT (method_code, 44, 1) ;

    // binary operator (5 hex digits)
    bool flipxy     = GB_RSHIFT (method_code, 43, 1) ;
    bool flipij     = GB_RSHIFT (method_code, 42, 1) ;
    #ifdef GB_DEBUG
    int binop_code  = GB_RSHIFT (method_code, 36, 6) ;
    #endif
//  int zcode       = GB_RSHIFT (method_code, 32, 4) ;
    int xcode       = GB_RSHIFT (method_code, 28, 4) ;
    int ycode       = GB_RSHIFT (method_code, 24, 4) ;

    // mask (1 hex digit)
    int mask_ecode  = GB_RSHIFT (method_code, 20, 4) ;

    // types of C, A, and B (3 hex digits)
    int ccode       = GB_RSHIFT (method_code, 16, 4) ;   // if 0: C is iso
    int acode       = GB_RSHIFT (method_code, 12, 4) ;   // if 0: A is pattern
    int bcode       = GB_RSHIFT (method_code,  8, 4) ;   // if 0: B is pattern

    bool C_iso = (ccode == 0) ;

    // formats of C, M, A, and B (2 hex digits)
    int csparsity   = GB_RSHIFT (method_code,  6, 2) ;
    int msparsity   = GB_RSHIFT (method_code,  4, 2) ;
    int asparsity   = GB_RSHIFT (method_code,  2, 2) ;
    int bsparsity   = GB_RSHIFT (method_code,  0, 2) ;

    //--------------------------------------------------------------------------
    // get the method
    //--------------------------------------------------------------------------

    bool is_eadd = (kcode == GB_JIT_KERNEL_ADD) ;
    bool is_kron = (kcode == GB_JIT_KERNEL_KRONER) ;

    //--------------------------------------------------------------------------
    // describe the operator
    //--------------------------------------------------------------------------

    GrB_Type xtype, ytype, ztype, theta_type ;
    const char *xtype_name, *ytype_name, *ztype_name, *theta_type_name ;
    ASSERT_BINARYOP_OK (binaryop, "binaryop to macrofy", GB0) ;

    GB_Opcode opcode ;
    if (C_iso)
    { 
        // values of C are not computed by the kernel
        opcode = GB_PAIR_binop_code ;
        xtype_name = "GB_void" ;
        ytype_name = "GB_void" ;
        ztype_name = "GB_void" ;
        theta_type_name = "void" ;
        xtype = NULL ;
        ytype = NULL ;
        ztype = NULL ;
        theta_type = NULL ;
        fprintf (fp, "// op: symbolic only (C is iso)\n\n") ;
    }
    else
    { 
        // general case
        opcode = binaryop->opcode ;
        if (xcode == GB_BOOL_code)  // && (ycode == GB_BOOL_code)
        { 
            // rename the operator
            opcode = GB_boolean_rename (opcode) ;
        }
        xtype = binaryop->xtype ;
        ytype = binaryop->ytype ;
        ztype = binaryop->ztype ;
        theta_type = binaryop->theta_type ;
        xtype_name = xtype->name ;
        ytype_name = ytype->name ;
        ztype_name = ztype->name ;
        theta_type_name = (theta_type == NULL) ? "void" : theta_type->name ;
        if (binaryop->hash == 0)
        { 
            // builtin operator
            fprintf (fp, "// op: (%s%s%s, %s)\n\n",
                binaryop->name,
                flipij ? " (flipped ij)" : "",
                flipxy ? " (flipped xy)" : "",
                xtype_name) ;
        }
        else
        { 
            // user-defined operator, or created by GB_wait
            fprintf (fp,
                "// op: %s%s%s%s, ztype: %s, xtype: %s, ytype: %s\n\n",
                (opcode == GB_SECOND_binop_code) ? "2nd_" : "",
                binaryop->name,
                flipij ? " (flipped ij)" : "",
                flipxy ? " (flipped xy)" : "",
                ztype_name, xtype_name, ytype_name) ;
        }
    }

    ASSERT (opcode == (binop_code + GB_USER_binop_code)) ;

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    if (!C_iso)
    { 
        GB_macrofy_typedefs (fp, ctype,
            (acode == 0 || acode == 15) ? NULL : atype,
            (bcode == 0 || bcode == 15) ? NULL : btype,
            xtype, ytype, ztype, theta_type) ;
    }

    fprintf (fp, "// binary operator types:\n") ;
    GB_macrofy_type (fp, "Z", "_", ztype_name) ;
    GB_macrofy_type (fp, "X", "_", xtype_name) ;
    GB_macrofy_type (fp, "Y", "_", ytype_name) ;
    if (GB_IS_INDEXBINARYOP_CODE (opcode))
    {
        GB_macrofy_type (fp, "THETA", "_", theta_type_name) ;
    }

    //--------------------------------------------------------------------------
    // construct macros for the binary operator
    //--------------------------------------------------------------------------

    int binop_ecode ;
    GB_enumify_binop (&binop_ecode, opcode, xcode, false, is_kron) ;

    fprintf (fp, "\n// binary operator%s%s:\n",
        flipij ? " (flipped ij)" : "",
        flipxy ? " (flipped xy)" : "") ;
    GB_macrofy_binop (fp, is_kron ? "GB_KRONOP" : "GB_BINOP",
        flipij, flipxy, false, true, is_kron,
        binop_ecode, C_iso, binaryop, NULL, NULL, NULL) ;

    if (opcode == GB_SECOND_binop_code)
    { 
        fprintf (fp, "#define GB_OP_IS_SECOND 1\n") ;
    }

    GB_macrofy_cast_copy (fp, "C", "A", (C_iso || !is_eadd) ? NULL : ctype,
            (acode == 0 || acode == 15) ? NULL : atype, A_iso) ;

    GB_macrofy_cast_copy (fp, "C", "B", (C_iso || !is_eadd) ? NULL : ctype,
            (bcode == 0 || bcode == 15) ? NULL : btype, B_iso) ;

    //--------------------------------------------------------------------------
    // macros for the C matrix
    //--------------------------------------------------------------------------

    GB_macrofy_output (fp, "c", "C", "C", ctype, ztype, csparsity, C_iso,
        C_in_iso, Cp_is_32, Cj_is_32, Ci_is_32) ;

    if (is_kron)
    { 
        fprintf (fp, "#define GB_KRONECKER_OP(Cx,p,a,ia,ja,b,ib,jb)") ;
        if (C_iso)
        { 
            fprintf (fp, "\n") ;
        }
        else
        { 
            ASSERT (ctype == ztype) ;
            fprintf (fp, " GB_KRONOP (Cx [p], a,ia,ja, b,ib,jb)\n") ;
        }
    }
    else
    {
        fprintf (fp, "#define GB_EWISEOP(Cx,p,aij,bij,i,j)") ;
        if (C_iso)
        { 
            fprintf (fp, "\n") ;
        }
        else if (ctype == ztype)
        { 
            fprintf (fp, " GB_BINOP (Cx [p], aij, bij, i, j)\n") ;
        }
        else
        { 
            fprintf (fp, " \\\n"
                "{                                      \\\n"
                "    GB_Z_TYPE z ;                      \\\n"
                "    GB_BINOP (z, aij, bij, i, j) ;     \\\n"
                "    GB_PUTC (z, Cx, p) ;               \\\n"
                "}\n") ;
        }
    }

    //--------------------------------------------------------------------------
    // construct the macros to access the mask (if any), and its name
    //--------------------------------------------------------------------------

    GB_macrofy_mask (fp, mask_ecode, "M", msparsity,
        Mp_is_32, Mj_is_32, Mi_is_32) ;

    //--------------------------------------------------------------------------
    // construct the macros for A and B
    //--------------------------------------------------------------------------

    // These methods create macros for defining the types of A and B, as well
    // as accessing the entries to provide inputs to the operator.  A and B
    // maybe be valued but not used for the operator.  For example, eWiseAdd
    // with the PAIR operator defines GB_DECLAREA, GB_GETA GB_DECLAREB, and
    // GB_GETB as empty, because the values of A and B are not needed for the
    // operator.  However, acode and bcode will not be 0, and GB_A_TYPE and
    // GB_B_TYPE will be defined, because the entries from A and B can bypass
    // the operator and be directly copied into C.

    // if flipxy false:  A is typecasted to x, and B is typecasted to y.
    // if flipxy true:   A is typecasted to y, and B is typecasted to x.

    if (xcode == 0)
    { 
        xtype = NULL ;
    }
    if (ycode == 0)
    { 
        ytype = NULL ;
    }

    GB_macrofy_input (fp, "a", "A", "A", true, flipxy ? ytype : xtype,
        atype, asparsity, acode, A_iso, -1, Ap_is_32, Aj_is_32, Ai_is_32) ;

    GB_macrofy_input (fp, "b", "B", "B", true, flipxy ? xtype : ytype,
        btype, bsparsity, bcode, B_iso, -1, Bp_is_32, Bj_is_32, Bi_is_32) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"include/GB_ewise_shared_definitions.h\"\n") ;
}

