//------------------------------------------------------------------------------
// GB_macrofy_masker: construct all macros for masker methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_masker          // construct all macros for GrB_eWise
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t method_code,
    GrB_Type rtype
)
{

    //--------------------------------------------------------------------------
    // extract the masker method_code
    //--------------------------------------------------------------------------

    // C and Z iso properties (1 hex digit)
    int C_iso       = GB_RSHIFT (method_code, 17, 1) ;
    int Z_iso       = GB_RSHIFT (method_code, 16, 1) ;

    // mask (1 hex digit)
    int mask_ecode  = GB_RSHIFT (method_code, 12, 4) ;

    // type of R (1 hex digit)
    int rcode       = GB_RSHIFT (method_code,  8, 4) ;

    // formats of R, M, C, and Z (2 hex digits)
    int rsparsity   = GB_RSHIFT (method_code,  6, 2) ;
    int msparsity   = GB_RSHIFT (method_code,  4, 2) ;
    int csparsity   = GB_RSHIFT (method_code,  2, 2) ;
    int zsparsity   = GB_RSHIFT (method_code,  0, 2) ;

    //--------------------------------------------------------------------------
    // construct the macros for the values of R
    //--------------------------------------------------------------------------

    fprintf (fp, "// masker: %s\n", (rtype == NULL) ? "" : rtype->name) ;
    if (rtype != NULL)
    { 

        //----------------------------------------------------------------------
        // construct the typedefs
        //----------------------------------------------------------------------

        GB_macrofy_typedefs (fp, rtype, NULL, NULL, NULL, NULL, NULL) ;
        GB_macrofy_type (fp, "R", "_", rtype->name) ;

        //----------------------------------------------------------------------
        // macros for copying entries into R
        //----------------------------------------------------------------------

        // copy a single value from C to R
        fprintf (fp, "#define GB_COPY_C_TO_R(Rx,pR,Cx,pC,C_iso,rsize) "
            "Rx [pR] = Cx [%s]\n", C_iso ? "0" : "pC") ;

        // copy a single value from Z to R
        fprintf (fp, "#define GB_COPY_Z_TO_R(Rx,pR,Zx,pZ,Z_iso,rsize) "
            "Rx [pR] = Zx [%s]\n", Z_iso ? "0" : "pZ") ;

        // copy a range of values from C to R
        fprintf (fp,
            "#define GB_COPY_C_TO_R_RANGE(Rx,pR,Cx,pC,C_iso,rsize,cjnz) \\\n"
            "{                                                          \\\n") ;
        if (C_iso)
        { 
            fprintf (fp,
            "    for (int64_t k = 0 ; k < cjnz ; k++)                   \\\n"
            "    {                                                      \\\n"
            "        Rx [pR+k] = Cx [0] ;                               \\\n"
            "    }                                                      \\\n") ;
        }
        else
        { 
            fprintf (fp,
            "    /* Rx [pR:pR+cjnz-1] = Cx [pC:pC+cjnz-1] */            \\\n"
            "    memcpy (Rx +(pR), Cx +(pC), (cjnz)*rsize) ;            \\\n") ;
        }
        fprintf (fp, "}\n") ;

        // copy a range of values from Z to R
        fprintf (fp,
            "#define GB_COPY_Z_TO_R_RANGE(Rx,pR,Zx,pZ,Z_iso,rsize,zjnz) \\\n"
            "{                                                          \\\n") ;
        if (Z_iso)
        { 
            fprintf (fp,
            "    for (int64_t k = 0 ; k < zjnz ; k++)                   \\\n"
            "    {                                                      \\\n"
            "        Rx [pR+k] = Zx [0] ;                               \\\n"
            "    }                                                      \\\n") ;
        }
        else
        { 
            fprintf (fp,
            "    /* Rx [pR:pR+zjnz-1] = Zx [pZ:pZ+zjnz-1] */            \\\n"
            "    memcpy (Rx +(pR), Zx +(pZ), (zjnz)*rsize) ;            \\\n") ;
        }
        fprintf (fp, "}\n") ;
    }

    GB_macrofy_sparsity (fp, "R", rsparsity) ;
    GB_macrofy_nvals (fp, "R", rsparsity, false) ;
    fprintf (fp, "#define GB_R_ISO 0\n") ;

    //--------------------------------------------------------------------------
    // construct the macros for C, M, and Z
    //--------------------------------------------------------------------------

    GB_macrofy_sparsity (fp, "C", csparsity) ;
    GB_macrofy_nvals (fp, "C", csparsity, C_iso) ;
    fprintf (fp, "#define GB_C_ISO %d\n", C_iso) ;

    GB_macrofy_mask (fp, mask_ecode, "M", msparsity) ;

    GB_macrofy_sparsity (fp, "Z", zsparsity) ;
    GB_macrofy_nvals (fp, "Z", zsparsity, Z_iso) ;
    fprintf (fp, "#define GB_Z_ISO %d\n", Z_iso) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"include/GB_masker_shared_definitions.h\"\n") ;
}

