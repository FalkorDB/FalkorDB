//------------------------------------------------------------------------------
// GB_macrofy_mask: return string to define mask macros
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// User-defined types cannot be used as a valued mask.  They can be used
// as structural masks.

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_mask
(
    FILE *fp,               // file to write macros, assumed open already
    // input:
    int mask_ecode,         // enumified mask
    char *Mname,            // name of the mask
    int msparsity           // sparsity of the mask
)
{

    if (mask_ecode >= 2)
    { 
        GB_macrofy_sparsity (fp, Mname, msparsity) ;
        GB_macrofy_nvals (fp, Mname, msparsity, false) ;
    }

    switch (mask_ecode)
    {

        //----------------------------------------------------------------------
        // no mask
        //----------------------------------------------------------------------

        case 0 :    // mask not complemented
            fprintf (fp, "\n// %s matrix: none\n"
                "#define GB_M_TYPE void\n"
                "#define GB_MCAST(Mx,p,msize) 1\n"
                "#define GB_MASK_STRUCT 1\n"
                "#define GB_MASK_COMP   0\n"
                "#define GB_NO_MASK     1\n",
                Mname) ;
            break ;

        case 1 :    // mask complemented
            fprintf (fp, "\n// %s matrix: none (complemented):\n"
                "#define GB_M_TYPE void\n"
                "#define GB_MCAST(Mx,p,msize) 1\n"
                "#define GB_MASK_STRUCT 1\n"
                "#define GB_MASK_COMP   1\n"
                "#define GB_NO_MASK     1\n",
                Mname) ;
            break ;

        //----------------------------------------------------------------------
        // M is present, and structural (not valued)
        //----------------------------------------------------------------------

        case 2 : 
            // mask not complemented, type: structural
            fprintf (fp, "// structural mask:\n"
                "#define GB_M_TYPE void\n"
                "#define GB_MCAST(Mx,p,msize) 1\n"
                "#define GB_MASK_STRUCT 1\n"
                "#define GB_MASK_COMP   0\n"
                "#define GB_NO_MASK     0\n"
                ) ;
            if (msparsity == 1)
            { 
                // Mask is present, sparse, not complemented, and structural
                fprintf (fp,
                "#define GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED\n") ;
            }
            break ;

        case 3 :
            // mask complemented, type: structural
            fprintf (fp,
                "// structural mask (complemented):\n"
                "#define GB_M_TYPE void\n"
                "#define GB_MCAST(Mx,p,msize) 1\n"
                "#define GB_MASK_STRUCT 1\n"
                "#define GB_MASK_COMP   1\n"
                "#define GB_NO_MASK     0\n"
                ) ;
            break ;

        //----------------------------------------------------------------------
        // M is present, valued, size: 1 byte
        //----------------------------------------------------------------------

        case 4 :
            // mask not complemented, type: bool, int8, uint8
            fprintf (fp,
                "// valued mask (1 byte):\n"
                "#define GB_M_TYPE uint8_t\n"
                "#define GB_MCAST(Mx,p,msize) (Mx [p] != 0)\n"
                "#define GB_MASK_STRUCT 0\n"
                "#define GB_MASK_COMP   0\n"
                "#define GB_NO_MASK     0\n"
                ) ;
            break ;

        case 5 :
            // mask complemented, type: bool, int8, uint8
            fprintf (fp,
                "// valued mask (1 byte, complemented):\n"
                "#define GB_M_TYPE uint8_t\n"
                "#define GB_MCAST(Mx,p,msize) (Mx [p] != 0)\n"
                "#define GB_MASK_STRUCT 0\n"
                "#define GB_MASK_COMP   1\n"
                "#define GB_NO_MASK     0\n") ;
            break ;

        //----------------------------------------------------------------------
        // M is present, valued, size: 2 bytes
        //----------------------------------------------------------------------

        case 6 :
            // mask not complemented, type: int16, uint16
            fprintf (fp,
                "// valued mask (2 bytes):\n"
                "#define GB_M_TYPE uint16_t\n"
                "#define GB_MCAST(Mx,p,msize) (Mx [p] != 0)\n"
                "#define GB_MASK_STRUCT 0\n"
                "#define GB_MASK_COMP   0\n"
                "#define GB_NO_MASK     0\n") ;
            break ;

        case 7 :
            // mask complemented, type: int16, uint16
            fprintf (fp,
                "// valued mask (2 bytes, complemented):\n"
                "#define GB_M_TYPE uint16_t\n"
                "#define GB_MCAST(Mx,p,msize) (Mx [p] != 0)\n"
                "#define GB_MASK_STRUCT 0\n"
                "#define GB_MASK_COMP   1\n"
                "#define GB_NO_MASK     0\n") ;
            break ;

        //----------------------------------------------------------------------
        // M is present, valued, size: 4 bytes
        //----------------------------------------------------------------------

        case 8 :
            // mask not complemented, type: float, int32, uint32
            fprintf (fp,
                "// valued mask (4 bytes):\n"
                "#define GB_M_TYPE uint32_t\n"
                "#define GB_MCAST(Mx,p,msize) (Mx [p] != 0)\n"
                "#define GB_MASK_STRUCT 0\n"
                "#define GB_MASK_COMP   0\n"
                "#define GB_NO_MASK     0\n") ;
            break ;

        case 9 :
            // mask complemented, type: float, int32, uint32
            fprintf (fp,
                "// valued mask (4 bytes, complemented):\n"
                "#define GB_M_TYPE uint32_t\n"
                "#define GB_MCAST(Mx,p,msize) (Mx [p] != 0)\n"
                "#define GB_MASK_STRUCT 0\n"
                "#define GB_MASK_COMP   1\n"
                "#define GB_NO_MASK     0\n") ;
            break ;

        //----------------------------------------------------------------------
        // M is present, valued, size: 8 bytes
        //----------------------------------------------------------------------

        case 10 :
            // mask not complemented, type: double, float complex, int64, uint64
            fprintf (fp,
                "// valued mask (8 bytes):\n"
                "#define GB_M_TYPE uint64_t\n"
                "#define GB_MCAST(Mx,p,msize) (Mx [p] != 0)\n"
                "#define GB_MASK_STRUCT 0\n"
                "#define GB_MASK_COMP   0\n"
                "#define GB_NO_MASK     0\n") ;
            break ;

        case 11 :
            // mask complemented, type: double, float complex, int64, uint64
            fprintf (fp,
                "// valued mask (8 bytes, complemented):\n"
                "#define GB_M_TYPE uint64_t\n"
                "#define GB_MCAST(Mx,p,msize) (Mx [p] != 0)\n"
                "#define GB_MASK_STRUCT 0\n"
                "#define GB_MASK_COMP   1\n"
                "#define GB_NO_MASK     0\n") ;
            break ;

        //----------------------------------------------------------------------
        // M is present, valued, size: 16 bytes
        //----------------------------------------------------------------------

        case 12 :
            // mask not complemented, type: double complex
            fprintf (fp,
                "// valued mask (16 bytes):\n"
                "#define GB_M_TYPE uint64_t\n"
                "#define GB_MCAST(Mx,p,msize) "
                "(Mx [2*(p)] != 0 || Mx [2*(p)+1] != 0)\n"
                "#define GB_MASK_STRUCT 0\n"
                "#define GB_MASK_COMP   0\n"
                "#define GB_NO_MASK     0\n") ;
            break ;

        case 13 :
            // mask complemented, type: double complex
            fprintf (fp,
                "// valued mask (16 bytes, complemented):\n"
                "#define GB_M_TYPE uint64_t\n"
                "#define GB_MCAST(Mx,p,msize) "
                "(Mx [2*(p)] != 0 || Mx [2*(p)+1] != 0)\n"
                "#define GB_MASK_STRUCT 0\n"
                "#define GB_MASK_COMP   1\n"
                "#define GB_NO_MASK     0\n") ;
            break ;

        //----------------------------------------------------------------------
        // invalid case
        //----------------------------------------------------------------------

        default: ;
            fprintf (fp, "#error undefined mask behavior\n") ;
            break ;
    }

}

