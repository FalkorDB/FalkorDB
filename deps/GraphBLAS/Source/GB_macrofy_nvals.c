//------------------------------------------------------------------------------
// GB_macrofy_nvals: construct macros to return # of entries in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Constructs two macros:
//
//  GB_A_NVALS(e): # of entries in a matrix, same as e = GB_nnz (A)
//  GB_A_NHELD(e): # of entries held in the data structure, same as
//      e = GB_nnz_held (A)

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_nvals  
(
    FILE *fp,
    // input:
    const char *Aname,      // name of input matrix (typically A, B, C,..)
    int asparsity,          // sparsity format of the input matrix, -1 if NULL
    bool A_iso              // true if A is iso
)
{

    //--------------------------------------------------------------------------
    // GB_*_NVALS macro to compute # entries in a matrix
    //--------------------------------------------------------------------------

    switch (asparsity)
    {
        case 0 :    // hypersparse
        case 1 :    // sparse
        case 2 :    // bitmap
            fprintf (fp, "#define GB_%s_NVALS(e) int64_t e = %s->nvals\n",
                Aname, Aname) ;
            break ;

        case 3 :    // full
            if (A_iso)
            {
                // guard against integer overflow
                fprintf (fp, "#define GB_%s_NVALS(e) int64_t e = 0 ;"
                    " GB_INT64_MULT (e, %s->vlen, %s->vdim)\n",
                    Aname, Aname, Aname) ;
            }
            else
            {
                // if A is not iso, it's not possible for e to overflow
                fprintf (fp, "#define GB_%s_NVALS(e) "
                    "int64_t e = (%s->vlen * %s->vdim)\n",
                    Aname, Aname, Aname) ;
            }
            break ;

        default:
            // matrix is not present
            fprintf (fp, "#define GB_%s_NVALS(e) int64_t e = 0\n", Aname) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // GB_*_NHELD macro to compute # entries held a matrix data structure
    //--------------------------------------------------------------------------

    if (asparsity == 2)
    { 
        // bitmap case
        fprintf (fp, "#define GB_%s_NHELD(e) "
            "int64_t e = (%s->vlen * %s->vdim)\n", Aname, Aname, Aname) ;
    }
    else
    { 
        // all other matrix formats
        fprintf (fp, "#define GB_%s_NHELD(e) GB_%s_NVALS(e)\n", Aname, Aname) ;
    }
}

