//------------------------------------------------------------------------------
// GB_macrofy_mxm: construct all macros for a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

//------------------------------------------------------------------------------
// GB_macrofy_mxm: create all macros for GrB_mxm
//------------------------------------------------------------------------------

void GB_macrofy_mxm         // construct all macros for GrB_mxm
(
    // output:
    FILE *fp,               // target file to write, already open
    // input:
    uint64_t method_code,
    GrB_Semiring semiring,  // the semiring to macrofy
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype
)
{

    //--------------------------------------------------------------------------
    // extract the semiring method_code
    //--------------------------------------------------------------------------

    // C, M, A, B: 32/64 (3 hex digits)
    bool Cp_is_32   = GB_RSHIFT (method_code, 63, 1) ;
    bool Cj_is_32   = GB_RSHIFT (method_code, 62, 1) ;
    bool Ci_is_32   = GB_RSHIFT (method_code, 61, 1) ;

    bool Mp_is_32   = GB_RSHIFT (method_code, 60, 1) ;
    bool Mj_is_32   = GB_RSHIFT (method_code, 59, 1) ;
    bool Mi_is_32   = GB_RSHIFT (method_code, 58, 1) ;

    bool Ap_is_32   = GB_RSHIFT (method_code, 57, 1) ;
    bool Aj_is_32   = GB_RSHIFT (method_code, 56, 1) ;
    bool Ai_is_32   = GB_RSHIFT (method_code, 55, 1) ;

    bool Bp_is_32   = GB_RSHIFT (method_code, 54, 1) ;
    bool Bj_is_32   = GB_RSHIFT (method_code, 53, 1) ;
    bool Bi_is_32   = GB_RSHIFT (method_code, 52, 1) ;

    // monoid (4 bits, 1 hex digit)
//  int add_code    = GB_RSHIFT (method_code, 48, 5) ;

    // C in, A, B iso-valued and flipxy (one hex digit)
    bool C_in_iso   = GB_RSHIFT (method_code, 47, 1) ;
    bool A_iso      = GB_RSHIFT (method_code, 46, 1) ;
    bool B_iso      = GB_RSHIFT (method_code, 45, 1) ;
    bool flipxy     = GB_RSHIFT (method_code, 44, 1) ;

    // multiplier (5 hex digits)
    // 2 bits unused here (42 and 43)
//  int mult_code   = GB_RSHIFT (method_code, 36, 6) ;
    int zcode       = GB_RSHIFT (method_code, 32, 4) ;    // if 0: C is iso
    int xcode       = GB_RSHIFT (method_code, 28, 4) ;    // if 0: ignored
    int ycode       = GB_RSHIFT (method_code, 24, 4) ;    // if 0: ignored

    // mask (one hex digit)
    int mask_ecode  = GB_RSHIFT (method_code, 20, 4) ;

    // types of C, A, and B (3 hex digits)
    int ccode       = GB_RSHIFT (method_code, 16, 4) ;   // if 0: C is iso
    int acode       = GB_RSHIFT (method_code, 12, 4) ;   // if 0: A is pattern
    int bcode       = GB_RSHIFT (method_code,  8, 4) ;   // if 0: B is pattern

    // formats of C, M, A, and B (2 hex digits)
    int csparsity   = GB_RSHIFT (method_code,  6, 2) ;
    int msparsity   = GB_RSHIFT (method_code,  4, 2) ;
    int asparsity   = GB_RSHIFT (method_code,  2, 2) ;
    int bsparsity   = GB_RSHIFT (method_code,  0, 2) ;

    //--------------------------------------------------------------------------
    // construct the semiring name
    //--------------------------------------------------------------------------

    GrB_Monoid monoid = semiring->add ;
    GrB_BinaryOp mult = semiring->multiply ;
    GrB_BinaryOp addop = monoid->op ;
    GB_Opcode mult_opcode = mult->opcode ;
    GB_Opcode add_opcode  = addop->opcode ;

    bool C_iso = (ccode == 0) ;

    if (C_iso)
    { 
        // C is iso; no operators are used
        add_opcode = GB_ANY_binop_code ;
        mult_opcode = GB_PAIR_binop_code ;
        xcode = 0 ;
        ycode = 0 ;
        zcode = 0 ;
        fprintf (fp, "// semiring: symbolic only (C is iso)\n") ;
    }
    else
    { 
        // general case

        fprintf (fp, "// semiring: (%s, %s%s, %s)\n",
            addop->name, mult->name, flipxy ? " (flipped)" : "",
            mult->xtype->name) ;
    }

    if (xcode == GB_BOOL_code)  // && (ycode == GB_BOOL_code)
    { 
        // rename the multiplicative operator
        mult_opcode = GB_boolean_rename (mult_opcode) ;
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GrB_Type xtype = (xcode == 0) ? NULL : mult->xtype ;
    GrB_Type ytype = (ycode == 0) ? NULL : mult->ytype ;
    GrB_Type ztype = (zcode == 0) ? NULL : mult->ztype ;

    if (!C_iso)
    { 
        GB_macrofy_typedefs (fp,
            (ccode == 0) ? NULL : ctype,
            (acode == 0) ? NULL : atype,
            (bcode == 0) ? NULL : btype,
            xtype, ytype, ztype, mult->theta_type) ;
    }

    //--------------------------------------------------------------------------
    // construct the monoid macros
    //--------------------------------------------------------------------------

    // turn off terminal condition for builtin monoids coupled with positional
    // multiply operators
    bool is_positional = GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (mult_opcode) ;

    fprintf (fp, "\n// monoid:\n") ;
    const char *u_expr, *g_expr ;
    GB_macrofy_type (fp, "Z", "_", (zcode == 0) ? "GB_void" : ztype->name) ;
    GB_macrofy_monoid (fp, C_iso, monoid, is_positional, &u_expr, &g_expr) ;

    //--------------------------------------------------------------------------
    // construct macros for the multiply operator
    //--------------------------------------------------------------------------

    int mult_ecode ;
    GB_enumify_binop (&mult_ecode, mult_opcode, xcode, true, false) ;

    fprintf (fp, "\n// multiplicative operator%s:\n",
        flipxy ? " (flipped)" : "") ;
    const char *f_expr ;
    GB_macrofy_type (fp, "X", "_", (xcode == 0) ? "GB_void" : xtype->name) ;
    GB_macrofy_type (fp, "Y", "_", (ycode == 0) ? "GB_void" : ytype->name) ;
    if (GB_IS_INDEXBINARYOP_CODE (mult_opcode))
    {
        char *theta_type_name = (mult->theta_type == NULL) ?
            "void" : mult->theta_type->name ;
        GB_macrofy_type (fp, "THETA", "_", theta_type_name) ;
    }
    GB_macrofy_binop (fp, "GB_MULT", false, flipxy, false, false, false,
        mult_ecode, C_iso, mult, &f_expr, NULL, NULL) ;

    //--------------------------------------------------------------------------
    // multiply-add operator
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// multiply-add operator:\n") ;
    bool is_bool   = (zcode == GB_BOOL_code) ;
    bool is_float  = (zcode == GB_FP32_code) ;
    bool is_double = (zcode == GB_FP64_code) ;
    bool is_first  = (mult_opcode == GB_FIRST_binop_code) ;
    bool is_second = (mult_opcode == GB_SECOND_binop_code) ;
    bool is_pair   = (mult_opcode == GB_PAIR_binop_code) ;
    bool is_user_monoid = (add_opcode == GB_USER_binop_code) ;

    if (C_iso)
    { 

        //----------------------------------------------------------------------
        // ANY_PAIR_BOOL semiring: nothing to do
        //----------------------------------------------------------------------

        fprintf (fp, "#define GB_MULTADD(z,x,y,i,k,j)\n") ;

    }
    else if (u_expr != NULL && f_expr != NULL && !is_user_monoid &&
        (is_float || is_double || is_bool || is_first || is_second || is_pair
            || is_positional))
    { 

        //----------------------------------------------------------------------
        // create a fused multiply-add operator
        //----------------------------------------------------------------------

        // Fusing operators can only be done if it avoids ANSI C integer
        // promotion rules.

        // float and double do not get promoted.
        // bool is OK since promotion of the result (0 or 1) to int is safe.
        // first and second are OK since no promotion occurs.
        // positional operators are OK too.

        // Since GB_MULT is not used, the fused GB_MULTADD must handle flipxy.

        if (g_expr == NULL)
        {
            // the CPU and GPU use the same macro
            GB_macrofy_multadd (fp, u_expr, f_expr, flipxy) ;
        }
        else
        {
            // the CPU uses u_expr, and GPU uses g_expr
            fprintf (fp, "#ifdef GB_CUDA_KERNEL\n") ;
            GB_macrofy_multadd (fp, g_expr, f_expr, flipxy) ;
            fprintf (fp, "#else\n") ;
            GB_macrofy_multadd (fp, u_expr, f_expr, flipxy) ;
            fprintf (fp, "#endif\n") ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // use a temporary variable for multiply-add
        //----------------------------------------------------------------------

        // All user-defined operators use this method. Built-in operators on
        // integers must use a temporary variable to avoid ANSI C integer
        // promotion.  Complex operators may use macros, so they use
        // temporaries as well.  GB_MULT handles flipxy.

        fprintf (fp,
            "#define GB_MULTADD(z,x,y,i,k,j)    \\\n"
            "{                                  \\\n"
            "   GB_Z_TYPE x_op_y ;              \\\n"
            "   GB_MULT (x_op_y, x,y,i,k,j) ;   \\\n"
            "   GB_UPDATE (z, x_op_y) ;         \\\n"
            "}\n") ;
    }

    //--------------------------------------------------------------------------
    // special case semirings
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// special cases:\n") ;

    if (C_iso)
    { 
        // ANY_PAIR_* (C is iso in this case, type is BOOL)
        fprintf (fp, "#define GB_IS_ANY_PAIR_SEMIRING 1\n") ;
    }
    else if (mult_opcode == GB_PAIR_binop_code)
    {

        //----------------------------------------------------------------------
        // ANY_PAIR, PLUS_PAIR, and related semirings
        //----------------------------------------------------------------------

        bool is_plus = (add_opcode == GB_PLUS_binop_code) ;
        if (is_plus && (zcode >= GB_INT8_code && zcode <= GB_FP64_code))
        { 

            // PLUS_PAIR_REAL semiring
            fprintf (fp, "#define GB_IS_PLUS_PAIR_REAL_SEMIRING 1\n") ;

            switch (zcode)
            {
                case GB_INT8_code    : 
                case GB_UINT8_code   : 
                    fprintf (fp, "#define GB_IS_PLUS_PAIR_8_SEMIRING 1\n") ;
                    break ;

                case GB_INT16_code   : 
                case GB_UINT16_code  : 
                    fprintf (fp, "#define GB_IS_PLUS_PAIR_16_SEMIRING 1\n") ;
                    break ;

                case GB_INT32_code   : 
                case GB_UINT32_code  : 
                    fprintf (fp, "#define GB_IS_PLUS_PAIR_32_SEMIRING 1\n") ;
                    break ;

                case GB_INT64_code   : 
                case GB_UINT64_code  : 
                case GB_FP32_code    : 
                case GB_FP64_code    : 
                    fprintf (fp, "#define GB_IS_PLUS_PAIR_BIG_SEMIRING 1\n") ;
                    break ;
                default:;
            }
        }
        else if (add_opcode == GB_LXOR_binop_code)
        { 
            // semiring is lxor_pair_bool
            fprintf (fp, "#define GB_IS_LXOR_PAIR_SEMIRING 1\n") ;
        }

    }
    else if (mult_opcode == GB_FIRSTJ_binop_code
          || mult_opcode == GB_FIRSTJ1_binop_code
          || mult_opcode == GB_SECONDI_binop_code
          || mult_opcode == GB_SECONDI1_binop_code)
    { 

        //----------------------------------------------------------------------
        // MIN_FIRSTJ and MAX_FIRSTJ
        //----------------------------------------------------------------------

        if (add_opcode == GB_MIN_binop_code)
        { 
            // semiring is min_firstj or min_firstj1
            fprintf (fp, "#define GB_IS_MIN_FIRSTJ_SEMIRING 1\n") ;
        }
        else if (add_opcode == GB_MAX_binop_code)
        { 
            // semiring is max_firstj or max_firstj1
            fprintf (fp, "#define GB_IS_MAX_FIRSTJ_SEMIRING 1\n") ;
        }

    }
    else if (add_opcode == GB_PLUS_binop_code &&
              mult_opcode == GB_TIMES_binop_code &&
            (zcode == GB_FP32_code || zcode == GB_FP64_code))
    { 

        //----------------------------------------------------------------------
        // semiring is PLUS_TIMES_FP32 or PLUS_TIMES_FP64
        //----------------------------------------------------------------------

        // future:: try AVX acceleration on more semirings
        fprintf (fp, "#define GB_SEMIRING_HAS_AVX_IMPLEMENTATION 1\n") ;
    }

    //--------------------------------------------------------------------------
    // special case multiply ops
    //--------------------------------------------------------------------------

    switch (mult_opcode)
    {
        case GB_PAIR_binop_code : 
            if (!is_user_monoid)
            { 
                fprintf (fp, "#define GB_IS_PAIR_MULTIPLIER 1\n") ;
                if (zcode == GB_FC32_code)
                { 
                    fprintf (fp, "#define GB_PAIR_ONE GxB_CMPLXF (1,0)\n") ;
                }
                else if (zcode == GB_FC64_code)
                { 
                    fprintf (fp, "#define GB_PAIR_ONE GxB_CMPLX (1,0)\n") ;
                }
            }
            break ;

        case GB_FIRSTI1_binop_code : 
            fprintf (fp, "#define GB_OFFSET 1\n") ;
        case GB_FIRSTI_binop_code : 
            fprintf (fp, "#define GB_IS_FIRSTI_MULTIPLIER 1\n") ;
            break ;

        case GB_FIRSTJ1_binop_code : 
        case GB_SECONDI1_binop_code : 
            fprintf (fp, "#define GB_OFFSET 1\n") ;
        case GB_FIRSTJ_binop_code : 
        case GB_SECONDI_binop_code : 
            fprintf (fp, "#define GB_IS_FIRSTJ_MULTIPLIER 1\n") ;
            break ;

        case GB_SECONDJ1_binop_code : 
            fprintf (fp, "\n#define GB_OFFSET 1\n") ;
        case GB_SECONDJ_binop_code : 
            fprintf (fp, "#define GB_IS_SECONDJ_MULTIPLIER 1\n") ;
            break ;

        default: ;
    }

    //--------------------------------------------------------------------------
    // macros for the C matrix
    //--------------------------------------------------------------------------

    GB_macrofy_output (fp, "c", "C", "C", ctype, ztype, csparsity, C_iso,
        C_in_iso, Cp_is_32, Cj_is_32, Ci_is_32) ;

    //--------------------------------------------------------------------------
    // construct the macros to access the mask (if any), and its name
    //--------------------------------------------------------------------------

    GB_macrofy_mask (fp, mask_ecode, "M", msparsity,
        Mp_is_32, Mj_is_32, Mi_is_32) ;

    //--------------------------------------------------------------------------
    // construct the macros for A and B
    //--------------------------------------------------------------------------

    // if flipxy false:  A is typecasted to x, and B is typecasted to y.
    // if flipxy true:   A is typecasted to y, and B is typecasted to x.

    GB_macrofy_input (fp, "a", "A", "A", true,
        flipxy ? ytype : xtype,
        atype, asparsity, acode, A_iso, -1, Ap_is_32, Aj_is_32, Ai_is_32) ;

    GB_macrofy_input (fp, "b", "B", "B", true,
        flipxy ? xtype : ytype,
        btype, bsparsity, bcode, B_iso, -1, Bp_is_32, Bj_is_32, Bi_is_32) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"include/GB_mxm_shared_definitions.h\"\n") ;
}

