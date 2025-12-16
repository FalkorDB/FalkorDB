//------------------------------------------------------------------------------
// GB_macrofy_reduce: construct all macros for a reduction to scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_reduce      // construct all macros for GrB_reduce to scalar
(
    FILE *fp,               // target file to write, already open
    // input:
    uint64_t rcode,         // encoded problem
    GrB_Monoid monoid,      // monoid to macrofy
    GrB_Type atype          // type of the A matrix to reduce
)
{ 

    //--------------------------------------------------------------------------
    // extract the reduction rcode
    //--------------------------------------------------------------------------

    // monoid
//  int cheese      = GB_RSHIFT (rcode, 16, 1) ;
//  int red_code    = GB_RSHIFT (rcode, 12, 4) ;

    // type of the monoid
    int zcode       = GB_RSHIFT (rcode, 8, 4) ;

    // type of A
    int acode       = GB_RSHIFT (rcode, 4, 4) ;

    // Ai: 32/64 bit
    bool Ai_is_32   = GB_RSHIFT (rcode, 3, 1) ;

    // zombies
    int azombies    = GB_RSHIFT (rcode, 2, 1) ;

    // format of A
    int asparsity   = GB_RSHIFT (rcode, 0, 2) ;

    //--------------------------------------------------------------------------
    // copyright, license, and describe monoid
    //--------------------------------------------------------------------------

    fprintf (fp, "// reduce: (%s, %s)\n",
        monoid->op->name, monoid->op->ztype->name) ;

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, NULL, atype, NULL, NULL, NULL, monoid->op->ztype,
        NULL) ;

    //--------------------------------------------------------------------------
    // construct the monoid macros
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// monoid:\n") ;
    GB_macrofy_type (fp, "Z", "_", monoid->op->ztype->name) ;
    GB_macrofy_monoid (fp, false, monoid, false, NULL, NULL) ;

    fprintf (fp, "#define GB_GETA_AND_UPDATE(z,Ax,p)") ;
    if (atype == monoid->op->ztype)
    { 
        // z += Ax [p], with no typecasting.  A is never iso.
        fprintf (fp, " GB_UPDATE (z, Ax [p])\n") ;
    }
    else
    { 
        // aij = (ztype) Ax [p] ; z += aij ; with typecasting.  A is never iso.
        fprintf (fp, " \\\n"
                     "{                             \\\n"
                     "    /* z += (ztype) Ax [p] */ \\\n"
                     "    GB_DECLAREA (aij) ;       \\\n"
                     "    GB_GETA (aij, Ax, p, ) ;  \\\n"
                     "    GB_UPDATE (z, aij) ;      \\\n"
                     "}\n"
                     ) ;
    }

    //--------------------------------------------------------------------------
    // construct the macros for A
    //--------------------------------------------------------------------------

    // iso reduction is handled by GB_reduce_to_scalar_iso, which takes
    // O(log(nvals(A))) for any monoid and uses the function pointer of the
    // monoid operator.  No JIT kernel is ever required to reduce an iso matrix
    // to a scalar, even for user-defined types and monoids.

    bool Ap_is_32 = false ; // OK: may be 32-bit but A->p is not accessed
    bool Aj_is_32 = false ; // OK: may be 32-bit but A->h is not accessed

    GB_macrofy_input (fp, "a", "A", "A", true, monoid->op->ztype,
        atype, asparsity, acode, /* A_iso: */ false, azombies,
        Ap_is_32, Aj_is_32, Ai_is_32) ;

    //--------------------------------------------------------------------------
    // reduction method
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// panel size for reduction:\n") ;
    int zsize = (int) monoid->op->ztype->size ;
    int panel = 1 ;

    GB_Opcode opcode = monoid->op->opcode ;

    if (opcode == GB_ANY_binop_code || azombies)
    { 
        // ANY monoid, or zombies: do not use panel reduction method
        panel = 1 ;
    }
    else if (zcode == GB_BOOL_code)
    { 
        // all boolean monoids, including user-defined
        panel = 8 ;
    }
    else
    { 

        switch (monoid->op->opcode)
        {

            // min and max
            case GB_MIN_binop_code : 
            case GB_MAX_binop_code : 
                panel = 16 ;
                break ;

            // plus, times, and all bitwise monoids
            case GB_PLUS_binop_code  : 
            case GB_TIMES_binop_code : 
            case GB_BOR_binop_code   : 
            case GB_BAND_binop_code  : 
            case GB_BXOR_binop_code  : 
            case GB_BXNOR_binop_code : 
                switch (zcode)
                {
                    // integer:
                    case GB_INT8_code    : 
                    case GB_UINT8_code   : 
                    case GB_INT16_code   : 
                    case GB_UINT16_code  : 
                    case GB_INT32_code   : 
                    case GB_UINT32_code  : panel = 64 ; break ;
                    case GB_INT64_code   : 
                    case GB_UINT64_code  : panel = 32 ; break ;

                    // floating point:
                    case GB_FP32_code    : panel = 64 ; break ;
                    case GB_FP64_code    : panel = 32 ; break ;
                    case GB_FC32_code    : panel = 32 ; break ;
                    case GB_FC64_code    : panel = 16 ; break ;
                    default:;
                }
                break ;

            default : 

                // all other monoids, including user-defined monoids
                if (zsize <= 16)
                { 
                    panel = 16 ;
                }
                else if (zsize <= 32)
                { 
                    panel = 8 ;
                }
                else
                { 
                    // type is large; do not use panel reduction method
                    panel = 1 ;
                }
        }
    }

    fprintf (fp, "#define GB_PANEL %d\n", panel) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"include/GB_monoid_shared_definitions.h\"\n") ;
}

