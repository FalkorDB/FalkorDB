//------------------------------------------------------------------------------
// GB_macrofy_mxm: construct all macros for a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_mxm        // construct all macros for GrB_mxm
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    GrB_Semiring semiring,  // the semiring to macrofy
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype
)
{

    //--------------------------------------------------------------------------
    // extract the semiring scode
    //--------------------------------------------------------------------------

    // monoid (4 hex digits)
//  int unused      = GB_RSHIFT (scode, 63, 1) ;
    int add_ecode   = GB_RSHIFT (scode, 58, 5) ;
    int id_ecode    = GB_RSHIFT (scode, 53, 5) ;
    int term_ecode  = GB_RSHIFT (scode, 48, 5) ;
//  bool is_term    = (term_ecode < 30) ;

    // C in, A, B iso-valued and flipxy (one hex digit)
    bool C_in_iso   = GB_RSHIFT (scode, 47, 1) ;
    int A_iso_code  = GB_RSHIFT (scode, 46, 1) ;
    int B_iso_code  = GB_RSHIFT (scode, 45, 1) ;
    bool flipxy     = GB_RSHIFT (scode, 44, 1) ;

    // multiplier (5 hex digits)
    int mult_ecode  = GB_RSHIFT (scode, 36, 8) ;
    int zcode       = GB_RSHIFT (scode, 32, 4) ;    // if 0: C is iso
    int xcode       = GB_RSHIFT (scode, 28, 4) ;    // if 0: ignored
    int ycode       = GB_RSHIFT (scode, 24, 4) ;    // if 0: ignored

    // mask (one hex digit)
    int mask_ecode  = GB_RSHIFT (scode, 20, 4) ;

    // types of C, A, and B (3 hex digits)
    int ccode       = GB_RSHIFT (scode, 16, 4) ;   // if 0: C is iso
    int acode       = GB_RSHIFT (scode, 12, 4) ;   // if 0: A is pattern
    int bcode       = GB_RSHIFT (scode,  8, 4) ;   // if 0: B is pattern

    // formats of C, M, A, and B (2 hex digits)
    int csparsity   = GB_RSHIFT (scode,  6, 2) ;
    int msparsity   = GB_RSHIFT (scode,  4, 2) ;
    int asparsity   = GB_RSHIFT (scode,  2, 2) ;
    int bsparsity   = GB_RSHIFT (scode,  0, 2) ;

    //--------------------------------------------------------------------------
    // construct the semiring name
    //--------------------------------------------------------------------------

    GrB_Monoid monoid = semiring->add ;
    GrB_BinaryOp mult = semiring->multiply ;
    GrB_BinaryOp addop = monoid->op ;

    bool C_iso = (ccode == 0) ;

    if (C_iso)
    { 
        // C is iso; no operators are used
        fprintf (fp, "// semiring: symbolic only (C is iso)\n") ;
    }
    else
    { 
        // general case
        fprintf (fp, "// semiring: (%s, %s%s, %s)\n",
            addop->name, mult->name, flipxy ? " (flipped)" : "",
            mult->xtype->name) ;
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
            xtype, ytype, ztype) ;
    }

    //--------------------------------------------------------------------------
    // construct the monoid macros
    //--------------------------------------------------------------------------

    // turn off terminal condition for builtin monoids coupled with positional
    // multiply operators
    bool is_positional = GB_IS_BINARYOP_CODE_POSITIONAL (mult->opcode) ;

    fprintf (fp, "\n// monoid:\n") ;
    const char *u_expr ;
    GB_macrofy_type (fp, "Z", "_", (zcode == 0) ? "GB_void" : ztype->name) ;
    GB_macrofy_monoid (fp, add_ecode, id_ecode, term_ecode, C_iso, monoid,
        is_positional, &u_expr) ;

    //--------------------------------------------------------------------------
    // construct macros for the multiply operator
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// multiplicative operator%s:\n",
        flipxy ? " (flipped)" : "") ;
    const char *f_expr ;
    GB_macrofy_binop (fp, "GB_MULT", flipxy, false, false, mult_ecode, C_iso,
        mult, &f_expr, NULL) ;

    //--------------------------------------------------------------------------
    // multiply-add operator
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// multiply-add operator:\n") ;
    bool is_bool   = (zcode == GB_BOOL_code) ;
    bool is_float  = (zcode == GB_FP32_code) ;
    bool is_double = (zcode == GB_FP64_code) ;
    bool is_first  = (mult->opcode == GB_FIRST_binop_code) ;
    bool is_second = (mult->opcode == GB_SECOND_binop_code) ;
    bool is_pair   = (mult->opcode == GB_PAIR_binop_code) ;

    if (C_iso)
    { 

        //----------------------------------------------------------------------
        // ANY_PAIR_BOOL semiring: nothing to do
        //----------------------------------------------------------------------

        fprintf (fp, "#define GB_MULTADD(z,x,y,i,k,j)\n") ;

    }
    else if (u_expr != NULL && f_expr != NULL &&
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

        if (flipxy)
        { 
            fprintf (fp, "#define GB_MULTADD(z,y,x,j,k,i) ") ;
        }
        else
        { 
            fprintf (fp, "#define GB_MULTADD(z,x,y,i,k,j) ") ;
        }
        for (const char *p = u_expr ; (*p) != '\0' ; p++)
        {
            // all update operators have a single 'y'
            if ((*p) == 'y')
            { 
                // inject the multiply operator; all have the form "z = ..."
                fprintf (fp, "%s", f_expr + 4) ;
            }
            else
            { 
                // otherwise, print the update operator character
                fprintf (fp, "%c", (*p)) ;
            }
        }
        fprintf (fp, "\n") ;

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

    if (mult->opcode == GB_PAIR_binop_code)
    {

        //----------------------------------------------------------------------
        // ANY_PAIR, PLUS_PAIR, and related semirings
        //----------------------------------------------------------------------

        bool is_plus = (addop->opcode == GB_PLUS_binop_code) ;
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
        else if (C_iso)
        { 
            // ANY_PAIR_* (C is iso in this case, type is BOOL)
            fprintf (fp, "#define GB_IS_ANY_PAIR_SEMIRING 1\n") ;
        }
        else if (addop->opcode == GB_LXOR_binop_code)
        { 
            // semiring is lxor_pair_bool
            fprintf (fp, "#define GB_IS_LXOR_PAIR_SEMIRING 1\n") ;
        }

    }
    else if (mult->opcode == GB_FIRSTJ_binop_code
          || mult->opcode == GB_FIRSTJ1_binop_code
          || mult->opcode == GB_SECONDI_binop_code
          || mult->opcode == GB_SECONDI1_binop_code)
    { 

        //----------------------------------------------------------------------
        // MIN_FIRSTJ and MAX_FIRSTJ
        //----------------------------------------------------------------------

        if (addop->opcode == GB_MIN_binop_code)
        { 
            // semiring is min_firstj or min_firstj1
            fprintf (fp, "#define GB_IS_MIN_FIRSTJ_SEMIRING 1\n") ;
        }
        else if (addop->opcode == GB_MAX_binop_code)
        { 
            // semiring is max_firstj or max_firstj1
            fprintf (fp, "#define GB_IS_MAX_FIRSTJ_SEMIRING 1\n") ;
        }

    }
    else if (addop->opcode == GB_PLUS_binop_code &&
              mult->opcode == GB_TIMES_binop_code &&
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

    switch (mult->opcode)
    {
        case GB_PAIR_binop_code : 
            fprintf (fp, "#define GB_IS_PAIR_MULTIPLIER 1\n") ;
            if (zcode == GB_FC32_code)
            { 
                fprintf (fp, "#define GB_PAIR_ONE GxB_CMPLXF (1,0)\n") ;
            }
            else if (zcode == GB_FC64_code)
            { 
                fprintf (fp, "#define GB_PAIR_ONE GxB_CMPLX (1,0)\n") ;
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
        C_in_iso) ;

    //--------------------------------------------------------------------------
    // construct the macros to access the mask (if any), and its name
    //--------------------------------------------------------------------------

    GB_macrofy_mask (fp, mask_ecode, "M", msparsity) ;

    //--------------------------------------------------------------------------
    // construct the macros for A and B
    //--------------------------------------------------------------------------

    // if flipxy false:  A is typecasted to x, and B is typecasted to y.
    // if flipxy true:   A is typecasted to y, and B is typecasted to x.

    GB_macrofy_input (fp, "a", "A", "A", true,
        flipxy ? mult->ytype : mult->xtype,
        atype, asparsity, acode, A_iso_code, -1) ;

    GB_macrofy_input (fp, "b", "B", "B", true,
        flipxy ? mult->xtype : mult->ytype,
        btype, bsparsity, bcode, B_iso_code, -1) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"GB_mxm_shared_definitions.h\"\n") ;
}

