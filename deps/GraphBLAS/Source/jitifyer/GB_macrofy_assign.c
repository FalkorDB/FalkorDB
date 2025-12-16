//------------------------------------------------------------------------------
// GB_macrofy_assign: construct all macros for assign methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_assign          // construct all macros for GrB_assign
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t method_code,
    GrB_BinaryOp accum,         // accum operator to macrofy
    GrB_Type ctype,
    GrB_Type atype              // matrix or scalar type
)
{

    //--------------------------------------------------------------------------
    // extract the assign method_code
    //--------------------------------------------------------------------------

    // S, C, M, A, I, J integer types (4 hex digits)
    bool Sp_is_32   = GB_RSHIFT (method_code, 62, 1) ;
    bool Sj_is_32   = GB_RSHIFT (method_code, 61, 1) ;
    bool Si_is_32   = GB_RSHIFT (method_code, 60, 1) ;
    bool Sx_is_32   = GB_RSHIFT (method_code, 59, 1) ;

    bool Cp_is_32   = GB_RSHIFT (method_code, 58, 1) ;
    bool Cj_is_32   = GB_RSHIFT (method_code, 57, 1) ;
    bool Ci_is_32   = GB_RSHIFT (method_code, 56, 1) ;

    bool Mp_is_32   = GB_RSHIFT (method_code, 55, 1) ;
    bool Mj_is_32   = GB_RSHIFT (method_code, 54, 1) ;
    bool Mi_is_32   = GB_RSHIFT (method_code, 53, 1) ;

    bool Ap_is_32   = GB_RSHIFT (method_code, 52, 1) ;
    bool Aj_is_32   = GB_RSHIFT (method_code, 51, 1) ;
    bool Ai_is_32   = GB_RSHIFT (method_code, 50, 1) ;

    bool I_is_32    = GB_RSHIFT (method_code, 49, 1) ;
    bool J_is_32    = GB_RSHIFT (method_code, 48, 1) ;

    // C_replace, S present, scalar assign, A iso (1 hex digit)
    int C_replace   = GB_RSHIFT (method_code, 47, 1) ;
    int S_present   = GB_RSHIFT (method_code, 46, 1) ;
    bool s_assign   = GB_RSHIFT (method_code, 45, 1) ;
    bool A_iso      = GB_RSHIFT (method_code, 44, 1) ;

    // Ikind, Jkind (1 hex digit)
    int Ikind       = GB_RSHIFT (method_code, 42, 2) ;
    int Jkind       = GB_RSHIFT (method_code, 40, 2) ;

    // accum operator and assign_kind (5 hex digits)
    int assign_kind = GB_RSHIFT (method_code, 38, 2) ;
//  int accum_code  = GB_RSHIFT (method_code, 32, 6) ;
//  int zcode       = GB_RSHIFT (method_code, 28, 4) ;
    int xcode       = GB_RSHIFT (method_code, 24, 4) ;
//  int ycode       = GB_RSHIFT (method_code, 20, 4) ;

    // mask (one hex digit)
    int mask_ecode  = GB_RSHIFT (method_code, 16, 4) ;

    // types of C and A (or scalar type) (2 hex digits)
    int ccode       = GB_RSHIFT (method_code, 12, 4) ;   // if 0: C is iso
    int acode       = GB_RSHIFT (method_code,  8, 4) ;

    // sparsity structures of C, M, and A (2 hex digits),
    int csparsity   = GB_RSHIFT (method_code,  6, 2) ;
    int msparsity   = GB_RSHIFT (method_code,  4, 2) ;
    int ssparsity   = GB_RSHIFT (method_code,  2, 2) ;
    int asparsity   = GB_RSHIFT (method_code,  0, 2) ;

    //--------------------------------------------------------------------------
    // describe the assignment
    //--------------------------------------------------------------------------

    bool C_iso = (ccode == 0) ;

    #define SLEN 512
    char description [SLEN] ;
    bool Mask_comp = (mask_ecode % 2 == 1) ;
    bool Mask_struct = (mask_ecode <= 3) ;
    bool M_is_null = (mask_ecode == 0) ;
    int M_sparsity ;
    switch (msparsity)
    {
        default :
        case 0 : M_sparsity = GxB_HYPERSPARSE ; break ;
        case 1 : M_sparsity = GxB_SPARSE      ; break ;
        case 2 : M_sparsity = GxB_BITMAP      ; break ;
        case 3 : M_sparsity = GxB_FULL        ; break ;
    }

    switch (assign_kind)
    {
        case GB_ASSIGN     : fprintf (fp, "// assign: "     ) ; break ;
        case GB_SUBASSIGN  : fprintf (fp, "// subassign: "  ) ; break ;
        case GB_ROW_ASSIGN : fprintf (fp, "// row assign: " ) ; break ;
        case GB_COL_ASSIGN : fprintf (fp, "// col assign: " ) ; break ;
        default:;
    }
    GB_assign_describe (description, SLEN, C_replace, Ikind, Jkind,
        M_is_null, M_sparsity, Mask_comp, Mask_struct, accum, s_assign,
        assign_kind) ;
    fprintf (fp, "%s\n", description) ;

    fprintf (fp, "#define GB_ASSIGN_KIND ") ;
    switch (assign_kind)
    {
        case GB_ASSIGN     : fprintf (fp, "GB_ASSIGN\n"     ) ; break ;
        case GB_SUBASSIGN  : fprintf (fp, "GB_SUBASSIGN\n"  ) ; break ;
        case GB_ROW_ASSIGN : fprintf (fp, "GB_ROW_ASSIGN\n" ) ; break ;
        case GB_COL_ASSIGN : fprintf (fp, "GB_COL_ASSIGN\n" ) ; break ;
        default:;
    }

    fprintf (fp, "#define GB_SCALAR_ASSIGN %d\n", s_assign ? 1 : 0) ;

    fprintf (fp, "#define GB_I_KIND ") ;
    switch (Ikind)
    {
        case GB_ALL    : fprintf (fp, "GB_ALL\n"    ) ; break ;
        case GB_RANGE  : fprintf (fp, "GB_RANGE\n"  ) ; break ;
        case GB_STRIDE : fprintf (fp, "GB_STRIDE\n" ) ; break ;
        case GB_LIST   : fprintf (fp, "GB_LIST\n"   ) ; break ;
        default:;
    }

    fprintf (fp, "#define GB_J_KIND ") ;
    switch (Jkind)
    {
        case GB_ALL    : fprintf (fp, "GB_ALL\n"    ) ; break ;
        case GB_RANGE  : fprintf (fp, "GB_RANGE\n"  ) ; break ;
        case GB_STRIDE : fprintf (fp, "GB_STRIDE\n" ) ; break ;
        case GB_LIST   : fprintf (fp, "GB_LIST\n"   ) ; break ;
        default:;
    }

    fprintf (fp, "#define GB_I_TYPE uint%d_t\n", I_is_32 ? 32 : 64) ;
    fprintf (fp, "#define GB_J_TYPE uint%d_t\n", J_is_32 ? 32 : 64) ;
    fprintf (fp, "#define GB_I_IS_32 %d\n", I_is_32 ? 1 : 0) ;
    fprintf (fp, "#define GB_J_IS_32 %d\n", J_is_32 ? 1 : 0) ;

    fprintf (fp, "#define GB_C_REPLACE %d\n", C_replace) ;

    //--------------------------------------------------------------------------
    // describe the accum operator
    //--------------------------------------------------------------------------

    GrB_Type xtype, ytype, ztype ;
    const char *xtype_name, *ytype_name, *ztype_name ;

    fprintf (fp, "\n// accum: ") ;
    if (accum == NULL)
    { 
        // accum operator is not present
        xtype_name = "GB_void" ;
        ytype_name = "GB_void" ;
        ztype_name = "GB_void" ;
        xtype = NULL ;
        ytype = NULL ;
        ztype = NULL ;
        fprintf (fp, "not present\n\n") ;
    }
    else
    { 
        // accum operator is present
        xtype = accum->xtype ;
        ytype = accum->ytype ;
        ztype = accum->ztype ;
        xtype_name = xtype->name ;
        ytype_name = ytype->name ;
        ztype_name = ztype->name ;
        if (accum->hash == 0)
        { 
            // builtin operator
            fprintf (fp, "(%s, %s)\n\n", accum->name, xtype_name) ;
        }
        else
        { 
            // user-defined operator
            fprintf (fp,
                "%s, ztype: %s, xtype: %s, ytype: %s\n\n",
                accum->name, ztype_name, xtype_name, ytype_name) ;
        }
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, ctype, atype, NULL, xtype, ytype, ztype, NULL) ;

    if (accum != NULL)
    { 
        fprintf (fp, "// accum operator types:\n") ;
        GB_macrofy_type (fp, "Z", "_", ztype_name) ;
        GB_macrofy_type (fp, "X", "_", xtype_name) ;
        GB_macrofy_type (fp, "Y", "_", ytype_name) ;
        fprintf (fp, "#define GB_DECLAREZ(zwork) %s zwork\n", ztype_name) ;
        fprintf (fp, "#define GB_DECLAREX(xwork) %s xwork\n", xtype_name) ;
        fprintf (fp, "#define GB_DECLAREY(ywork) %s ywork\n", ytype_name) ;
    }

    //--------------------------------------------------------------------------
    // construct macros for the accum operator
    //--------------------------------------------------------------------------

    bool did_accum_scalar = false ;
    bool did_accum_aij = false ;
    bool need_copy_c_to_xwork = false ;

    if (accum != NULL)
    {
        fprintf (fp, "\n// accum operator:\n") ;

        GB_Opcode accum_opcode = accum->opcode ;
        if (xcode == GB_BOOL_code)  // && (ycode == GB_BOOL_code)
        { 
            // rename the operator
            accum_opcode = GB_boolean_rename (accum_opcode) ;
        }
        int accum_ecode ;
        GB_enumify_binop (&accum_ecode, accum_opcode, xcode, false, false) ;
        GB_macrofy_binop (fp, "GB_ACCUM_OP", false, false, true, false, false,
            accum_ecode, C_iso, accum, NULL, NULL, NULL) ;

        char *yname = "ywork" ;

        if (s_assign)
        {
            did_accum_scalar = true ;
            fprintf (fp, "#define GB_ACCUMULATE_scalar(Cx,pC,ywork,C_iso)") ;
            if (C_iso)
            { 
                fprintf (fp, "\n") ;
            }
            else
            { 
                fprintf (fp, " \\\n"
                    "{                                          \\\n") ;
            }
            // the scalar has already been typecasted into ywork
        }
        else
        {
            did_accum_aij = true ;
            fprintf (fp,
                "#define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork,C_iso)") ;
            if (C_iso)
            { 
                fprintf (fp, "\n") ;
            }
            else
            { 
                fprintf (fp, " \\\n"
                    "{                                          \\\n") ;
                // if A is iso, its iso value is already typecasted into ywork
                if (!A_iso)
                {
                    if (atype == ytype)
                    { 
                        // use Ax [pA] directly instead of ywork
                        yname = "Ax [pA]" ;
                    }
                    else
                    { 
                        // ywork = (ytype) Ax [pA]
                        fprintf (fp,
                        "    GB_DECLAREY (ywork) ;                  \\\n"
                        "    GB_GETA (ywork, Ax, pA, ) ;            \\\n") ;
                    }
                }
            }
        }

        if (!C_iso)
        {
            char *xname ;
            if (xtype == ctype)
            { 
                // use Cx [pC] directly
                xname = "Cx [pC]" ;
            }
            else
            { 
                // xwork = (xtype) Cx [pC]
                need_copy_c_to_xwork = true ;
                xname = "xwork" ;
                fprintf (fp,
                    "    GB_DECLAREX (xwork) ;                  \\\n"
                    "    GB_COPY_C_to_xwork (xwork, Cx, pC) ;   \\\n") ;
            }
            if (ztype == ctype)
            {
                // write directly in Cx [pC], no need for zwork
                if (xtype == ctype)
                { 
                    // use the update method: Cx [pC] += y
                    fprintf (fp,
                    "    GB_UPDATE (Cx [pC], %s) ;          \\\n"
                    "}\n", yname) ;
                }
                else
                { 
                    // Cx [pC] = f (x,y)
                    fprintf (fp,
                    "    GB_ACCUM_OP (Cx [pC], %s, %s) ;          \\\n"
                    "}\n", xname, yname) ;
                }
            }
            else
            { 
                // zwork = f (x,y)
                // Cx [pC] = (ctype) zwork
                fprintf (fp,
                "    GB_DECLAREZ (zwork) ;                  \\\n"
                "    GB_ACCUM_OP (zwork, %s, %s) ;          \\\n"
                "    GB_PUTC (zwork, Cx, pC) ;              \\\n"
                "}\n", xname, yname) ;
            }
        }
    }

    if (!did_accum_scalar)
    { 
        fprintf (fp, "#define GB_ACCUMULATE_scalar(Cx,pC,ywork,C_iso)"
            " /* unused */\n") ;
    }

    if (!did_accum_aij)
    { 
        fprintf (fp, "#define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork,C_iso)"
            " /* unused */\n") ;
    }

    //--------------------------------------------------------------------------
    // macros for the C matrix
    //--------------------------------------------------------------------------

    if (accum == NULL)
    { 
        // C(i,j) = (ctype) cwork, no typecasting
        GB_macrofy_output (fp, "cwork", "C", "C", ctype, ctype, csparsity,
            C_iso, C_iso, Cp_is_32, Cj_is_32, Ci_is_32) ;
    }
    else
    { 
        // C(i,j) = (ctype) zwork, with possible typecasting
        GB_macrofy_output (fp, "zwork", "C", "C", ctype, ztype, csparsity,
            C_iso, C_iso, Cp_is_32, Cj_is_32, Ci_is_32) ;
    }

    fprintf (fp, "#define GB_DECLAREC(cwork) %s cwork\n", ctype->name) ;

    if (s_assign)
    { 
        // cwork = (ctype) scalar
        GB_macrofy_cast_input (fp, "GB_COPY_scalar_to_cwork", "cwork",
            "scalar", "(*((GB_A_TYPE *) scalar))", ctype, atype) ;
        // C(i,j) = (ctype) scalar, already typecasted to cwork
        fprintf (fp, "#define GB_COPY_cwork_to_C(Cx,pC,cwork,C_iso)%s",
            C_iso ? "\n" : " Cx [pC] = cwork\n") ;
        // no copy of A(i,j) to cwork
        fprintf (fp, "#define GB_COPY_aij_to_cwork(cwork,Ax,pA,A_iso)"
            " /* unused */\n") ;
        // no copy of A(i,j) to C(i,j)
        fprintf (fp, "#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork,C_iso)"
            " /* unused */\n") ;
    }
    else
    {
        // C(i,j) = (ctype) A(i,j)
        GB_macrofy_cast_copy (fp, "C", "A", (C_iso) ? NULL : ctype, atype,
            A_iso) ;
        fprintf (fp, "#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork,C_iso)");
        if (C_iso)
        { 
            fprintf (fp, "\n");
        }
        else if (A_iso)
        { 
            // cwork = (ctype) Ax [0] already done
            fprintf (fp, " Cx [pC] = cwork\n") ;
        }
        else
        { 
            // general case
            fprintf (fp, " \\\n    GB_COPY_A_to_C (Cx, pC, Ax, pA, A_iso)\n") ;
        }
        // cwork = (ctype) A(i,j)
        GB_macrofy_cast_input (fp, "GB_COPY_aij_to_cwork", "cwork",
            "Ax,p,A_iso", A_iso ? "Ax [0]" : "Ax [p]", ctype, atype) ;
        // no copy of cwork to C
        fprintf (fp, "#define GB_COPY_cwork_to_C(Cx,pC,cwork,C_iso)"
            " /* unused */\n") ;
        // no copy of scalar to cwork
        fprintf (fp, "#define GB_COPY_scalar_to_cwork(cwork,scalar)"
            " /* unused */\n") ;
    }

    // xwork = (xtype) C(i,j), if needed
    if (need_copy_c_to_xwork)
    { 
        ASSERT (accum != NULL) ;
        ASSERT (!C_iso) ;
        ASSERT (xtype != ctype) ;
        GB_macrofy_cast_input (fp, "GB_COPY_C_to_xwork", "xwork",
            "Cx,p", "Cx [p]", xtype, ctype) ;
    }

    //--------------------------------------------------------------------------
    // construct the macros to access the mask (if any), and its name
    //--------------------------------------------------------------------------

    GB_macrofy_mask (fp, mask_ecode, "M", msparsity,
        Mp_is_32, Mj_is_32, Mi_is_32) ;

    //--------------------------------------------------------------------------
    // construct the macros for A or the scalar, including typecast to Y type
    //--------------------------------------------------------------------------

    bool did_scalar_to_ywork = false ;
    bool did_aij_to_ywork = false ;

    if (s_assign)
    { 
        // scalar assignment
        fprintf (fp, "\n// scalar:\n") ;
        GB_macrofy_type (fp, "A", "_", atype->name) ;
        if (accum != NULL)
        { 
            // accum is present
            // ywork = (ytype) scalar
            GB_macrofy_cast_input (fp, "GB_COPY_scalar_to_ywork", "ywork",
                "scalar", "(*((GB_A_TYPE *) scalar))", ytype, atype) ;
            did_scalar_to_ywork = true ;
        }
        GB_macrofy_sparsity (fp, "A", -1) ; // unused macros
        fprintf (fp, "#define GB_A_NVALS(e) int64_t e = 1 ; /* unused */\n") ;
        fprintf (fp, "#define GB_A_NHELD(e) int64_t e = 1 ; /* unused */\n") ;
        GB_macrofy_bits (fp, "A", false, false, false) ;
    }
    else
    {
        // matrix assignment
        GB_macrofy_input (fp, "a", "A", "A", true, ytype, atype, asparsity,
            acode, A_iso, -1, Ap_is_32, Aj_is_32, Ai_is_32) ;
        if (accum != NULL)
        { 
            // accum is present
            // ywork = (ytype) A(i,j)
            fprintf (fp, "#define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso) "
                "GB_GETA (ywork, Ax, pA, A_iso)\n") ;
            did_aij_to_ywork = true ;
        }
    }

    if (!did_scalar_to_ywork)
    {
        fprintf (fp, "#define GB_COPY_scalar_to_ywork(ywork,scalar)"
            " /* unused */\n") ;
    }

    if (!did_aij_to_ywork)
    {
        fprintf (fp, "#define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso)"
            " /* unused */\n") ;
    }

    //--------------------------------------------------------------------------
    // construct the macros for S
    //--------------------------------------------------------------------------

    if (S_present)
    {
        GB_macrofy_sparsity (fp, "S", ssparsity) ;
        fprintf (fp, "#define GB_S_CONSTRUCTED 1\n") ;
        GB_macrofy_bits (fp, "S", Sp_is_32, Sj_is_32, Si_is_32) ;
        fprintf (fp, "#define GB_Sx_BITS %d\n", Sx_is_32 ? 32 : 64) ;
        fprintf (fp, "#define GB_Sx_TYPE uint%d_t\n", Sx_is_32 ? 32 : 64) ;
    }
    else
    {
        fprintf (fp, "\n// S matrix: not constructed\n")  ;
        fprintf (fp, "#define GB_S_CONSTRUCTED 0\n") ;
    }

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"include/GB_assign_shared_definitions.h\"\n") ;
}

