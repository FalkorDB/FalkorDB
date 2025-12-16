//------------------------------------------------------------------------------
// GB_macrofy_bits: construct macros for 32/64 bit integers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_bits
(
    FILE *fp,
    // input:
    const char *Aname,      // name of the matrix
    int p_is_32,            // if true, Ap is 32-bit, else 64-bit
    int j_is_32,            // if true, Ah is 32-bit, else 64-bit
    int i_is_32             // if true, Ai is 32-bit, else 64-bit
)
{
    int pbits = p_is_32 ? 32 : 64 ;
    int jbits = j_is_32 ? 32 : 64 ;
    int ibits = i_is_32 ? 32 : 64 ;

    fprintf (fp, "#define GB_%sp_TYPE uint%d_t\n",       Aname, pbits) ;
    fprintf (fp, "#define GB_%sj_TYPE uint%d_t\n",       Aname, jbits) ;
    fprintf (fp, "#define GB_%sj_SIGNED_TYPE int%d_t\n", Aname, jbits) ;
    fprintf (fp, "#define GB_%si_TYPE uint%d_t\n",       Aname, ibits) ;
    fprintf (fp, "#define GB_%si_SIGNED_TYPE int%d_t\n", Aname, ibits) ;
    fprintf (fp, "#define GB_%sp_BITS %d\n",             Aname, pbits) ;
    fprintf (fp, "#define GB_%sj_BITS %d\n",             Aname, jbits) ;
    fprintf (fp, "#define GB_%si_BITS %d\n",             Aname, ibits) ;
}

