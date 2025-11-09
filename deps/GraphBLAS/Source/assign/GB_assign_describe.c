//------------------------------------------------------------------------------
// GB_assign_describe: construct a string that describes GrB_assign / subassign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

void GB_assign_describe
(
    // output
    char *str,                  // string of size slen
    int slen,
    // input
    const bool C_replace,       // descriptor for C
    const int Ikind,
    const int Jkind,
    const bool M_is_null,
    const int M_sparsity,
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present here
    const bool A_is_null,
    const int assign_kind       // row assign, col assign, assign, or subassign
)
{

    //--------------------------------------------------------------------------
    // construct the accum operator string
    //--------------------------------------------------------------------------

    str [0] = '\0' ;
    char *op_str ;
    if (accum == NULL)
    { 
        // no accum operator is present
        op_str = "" ;
    }
    else
    { 
        // use a simpler version of accum->name
        if (accum->opcode == GB_USER_binop_code) op_str = "op" ;
        else if (GB_STRING_MATCH (accum->name, "plus")) op_str = "+" ;
        else if (GB_STRING_MATCH (accum->name, "minus")) op_str = "-" ;
        else if (GB_STRING_MATCH (accum->name, "times")) op_str = "*" ;
        else if (GB_STRING_MATCH (accum->name, "div")) op_str = "/" ;
        else if (GB_STRING_MATCH (accum->name, "or")) op_str = "|" ;
        else if (GB_STRING_MATCH (accum->name, "and")) op_str = "&" ;
        else if (GB_STRING_MATCH (accum->name, "xor")) op_str = "^" ;
        else op_str = accum->name ;
    }

    //--------------------------------------------------------------------------
    // construct the Mask string
    //--------------------------------------------------------------------------

    #define GB_MASK_STRING_LEN 128
    const char *Mask ;
    char Mask_string [GB_MASK_STRING_LEN+1] ;
    if (M_is_null)
    {
        // M is not present
        if (Mask_comp)
        { 
            Mask = C_replace ? "<!,replace>" : "<!>" ;
        }
        else
        { 
            Mask = C_replace ? "<replace>" : "" ;
        }
    }
    else
    { 
        // M is present
        snprintf (Mask_string, GB_MASK_STRING_LEN, "<%sM%s%s%s>",
            (Mask_comp) ? "!" : "",
            (M_sparsity == GxB_BITMAP) ? ",bitmap"
                : ((M_sparsity == GxB_FULL) ? ",full" : ""),
            Mask_struct ? ",struct" : "",
            C_replace ? ",replace" : "") ;
        Mask = Mask_string ;
    }

    //--------------------------------------------------------------------------
    // construct the string for A or the scalar
    //--------------------------------------------------------------------------

    const char *S = (A_is_null) ? "scalar" : "A" ;

    //--------------------------------------------------------------------------
    // construct the string for (I,J)
    //--------------------------------------------------------------------------

    const char *Istr = (Ikind == GB_ALL) ? ":" : "I" ;
    const char *Jstr = (Jkind == GB_ALL) ? ":" : "J" ;

    //--------------------------------------------------------------------------
    // burble the final result
    //--------------------------------------------------------------------------

    switch (assign_kind)
    {
        case GB_ROW_ASSIGN : 
            // C(i,J) = A
            snprintf (str, slen, "C%s(i,%s) %s= A ", Mask, Jstr, op_str) ;
            break ;

        case GB_COL_ASSIGN : 
            // C(I,j) = A
            snprintf (str, slen, "C%s(%s,j) %s= A ", Mask, Istr, op_str) ;
            break ;

        case GB_ASSIGN : 
            // C<M>(I,J) = A
            if (Ikind == GB_ALL && Jkind == GB_ALL)
            { 
                // C<M> += A
                snprintf (str, slen, "C%s %s= %s ", Mask, op_str, S) ;
            }
            else
            { 
                // C<M>(I,J) = A
                snprintf (str, slen, "C%s(%s,%s) %s= %s ", Mask, Istr, Jstr,
                    op_str, S) ;
            }
            break ;

        case GB_SUBASSIGN : 
            // C(I,J)<M> = A
            if (Ikind == GB_ALL && Jkind == GB_ALL)
            { 
                // C<M> += A
                snprintf (str, slen, "C%s %s= %s ", Mask, op_str, S) ;
            }
            else
            { 
                // C(I,J)<M> = A
                snprintf (str, slen, "C(%s,%s)%s %s= %s ", Istr, Jstr, Mask,
                    op_str, S) ;
            }
            break ;

        default: ;
    }
}

