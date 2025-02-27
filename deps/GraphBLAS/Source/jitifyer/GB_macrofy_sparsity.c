//------------------------------------------------------------------------------
// GB_macrofy_sparsity: define macro for the sparsity structure of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_sparsity    // construct macros for sparsity structure
(
    // input:
    FILE *fp,
    const char *matrix_name,    // "C", "M", "A", or "B"
    int sparsity
)
{ 

    fprintf (fp, "\n// %s matrix: ", matrix_name) ;

    switch (sparsity)
    {

        case 0 :    // hypersparse
            fprintf ( fp, "hypersparse\n"
                "#define GB_%s_IS_HYPER  1\n"
                "#define GB_%s_IS_SPARSE 0\n"
                "#define GB_%s_IS_BITMAP 0\n"
                "#define GB_%s_IS_FULL   0\n",
                matrix_name, matrix_name, matrix_name, matrix_name) ;
            fprintf (fp,
                "#define GBp_%s(%sp,k,vlen) %sp [k]\n"
                "#define GBh_%s(%sh,k)      %sh [k]\n"
                "#define GBi_%s(%si,p,vlen) %si [p]\n"
                "#define GBb_%s(%sb,p)      1\n",
                matrix_name, matrix_name, matrix_name, matrix_name,
                matrix_name, matrix_name, matrix_name, matrix_name,
                matrix_name, matrix_name, matrix_name) ;

            break ;

        case 1 :    // sparse
            fprintf ( fp,  "sparse\n"
                "#define GB_%s_IS_HYPER  0\n"
                "#define GB_%s_IS_SPARSE 1\n"
                "#define GB_%s_IS_BITMAP 0\n"
                "#define GB_%s_IS_FULL   0\n",
                matrix_name, matrix_name, matrix_name, matrix_name) ;
            fprintf (fp,
                "#define GBp_%s(%sp,k,vlen) %sp [k]\n"
                "#define GBh_%s(%sh,k)      (k)\n"
                "#define GBi_%s(%si,p,vlen) %si [p]\n"
                "#define GBb_%s(%sb,p)      1\n",
                matrix_name, matrix_name, matrix_name, matrix_name,
                matrix_name, matrix_name, matrix_name, matrix_name,
                matrix_name, matrix_name) ;

            break ;

        case 2 :    // bitmap
            fprintf ( fp,  "bitmap\n"
                "#define GB_%s_IS_HYPER  0\n"
                "#define GB_%s_IS_SPARSE 0\n"
                "#define GB_%s_IS_BITMAP 1\n"
                "#define GB_%s_IS_FULL   0\n",
                matrix_name, matrix_name, matrix_name, matrix_name) ;
            fprintf (fp,
                "#define GBp_%s(%sp,k,vlen) ((k) * (vlen))\n"
                "#define GBh_%s(%sh,k)      (k)\n"
                "#define GBi_%s(%si,p,vlen) ((p) %% (vlen))\n"
                "#define GBb_%s(%sb,p)      %sb [p]\n",
                matrix_name, matrix_name, matrix_name, matrix_name,
                matrix_name, matrix_name, matrix_name, matrix_name,
                matrix_name) ;

            break ;

        case 3 :    // full
            fprintf ( fp, "full\n"
                "#define GB_%s_IS_HYPER  0\n"
                "#define GB_%s_IS_SPARSE 0\n"
                "#define GB_%s_IS_BITMAP 0\n"
                "#define GB_%s_IS_FULL   1\n",
                matrix_name, matrix_name, matrix_name, matrix_name) ;
            fprintf (fp,
                "#define GBp_%s(%sp,k,vlen) ((k) * (vlen))\n"
                "#define GBh_%s(%sh,k)      (k)\n"
                "#define GBi_%s(%si,p,vlen) ((p) %% (vlen))\n"
                "#define GBb_%s(%sb,p)      1\n",
                matrix_name, matrix_name, matrix_name, matrix_name,
                matrix_name, matrix_name, matrix_name, matrix_name) ;

            break ;

        default :   // unused
            fprintf ( fp, "unused\n"
                "#define GB_%s_IS_HYPER  0\n"
                "#define GB_%s_IS_SPARSE 0\n"
                "#define GB_%s_IS_BITMAP 0\n"
                "#define GB_%s_IS_FULL   1\n",
                matrix_name, matrix_name, matrix_name, matrix_name) ;
            fprintf (fp,
                "#define GBp_%s(%sp,k,vlen) 0\n"
                "#define GBh_%s(%sh,k)      (k)\n"
                "#define GBi_%s(%si,p,vlen) 0\n"
                "#define GBb_%s(%sb,p)      1\n",
                matrix_name, matrix_name, matrix_name, matrix_name,
                matrix_name, matrix_name, matrix_name, matrix_name) ;

            break ;

    }
}

