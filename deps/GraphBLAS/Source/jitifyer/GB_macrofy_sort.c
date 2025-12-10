//------------------------------------------------------------------------------
// GB_macrofy_sort: construct all macros for sort methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_sort            // construct all macros for GxB_sort
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t method_code,
    GrB_BinaryOp binaryop,      // binaryop to macrofy
    GrB_Type ctype
)
{ 

    //--------------------------------------------------------------------------
    // extract the binaryop method_code
    //--------------------------------------------------------------------------

    // integers of C
    bool Cp_is_32   = GB_RSHIFT (method_code, 16, 1) ;
    bool Cj_is_32   = GB_RSHIFT (method_code, 15, 1) ;
    bool Ci_is_32   = GB_RSHIFT (method_code, 14, 1) ;

    // binary operator
//  int binop_code  = GB_RSHIFT (method_code,  8, 6) ;
    int xcode       = GB_RSHIFT (method_code,  4, 4) ;

    // type of C (1 hex digit)
    int ccode       = GB_RSHIFT (method_code,  0, 4) ; // 1 to 14, C is not iso

    //--------------------------------------------------------------------------
    // describe the operator
    //--------------------------------------------------------------------------

    ASSERT_BINARYOP_OK (binaryop, "binaryop to macrofy", GB0) ;

    GrB_Type xtype = binaryop->xtype ;
    const char *xtype_name = xtype->name ;

    fprintf (fp, "// comparator: (%s, %s)\n\n", binaryop->name, xtype_name) ;

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, ctype, xtype, NULL, NULL, NULL, NULL, NULL) ;

    fprintf (fp, "// comparator input type:\n") ;
    GB_macrofy_type (fp, "X", "_", xtype_name) ;

    //--------------------------------------------------------------------------
    // construct macros for the binary operator
    //--------------------------------------------------------------------------

    GB_Opcode opcode = binaryop->opcode ;
    if (xcode == GB_BOOL_code)  // && (ycode == GB_BOOL_code)
    { 
        // rename the operator
        opcode = GB_boolean_rename (opcode) ;
    }
    int binop_ecode ;
    GB_enumify_binop (&binop_ecode, opcode, xcode, false, false) ;

    fprintf (fp, "\n// binary operator:\n") ;
    GB_macrofy_binop (fp, "GB_BINOP", false, false, false, true, false,
        binop_ecode, false, binaryop, NULL, NULL, NULL) ;

    //--------------------------------------------------------------------------
    // macros for the C matrix
    //--------------------------------------------------------------------------

    GB_macrofy_input (fp, "c", "C", "C", true, xtype, ctype, 1, ccode,
        /* C_iso: */ false, -1, Cp_is_32, Cj_is_32, Ci_is_32) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"include/GB_kernel_shared_definitions.h\"\n") ;
}

