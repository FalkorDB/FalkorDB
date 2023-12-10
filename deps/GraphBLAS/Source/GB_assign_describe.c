//------------------------------------------------------------------------------
// GB_assign_describe: construct a string that describes GrB_assign / subassign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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
//  const GrB_Matrix M,
    const bool M_is_null,
    const int M_sparsity,
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present here
//  const GrB_Matrix A,         // input matrix, not transposed
    const bool A_is_null,
    const int assign_kind       // row assign, col assign, assign, or subassign
)
{

    //--------------------------------------------------------------------------
    // construct the accum operator string
    //--------------------------------------------------------------------------

    str [0] = '\0' ;
    char *Op ;
    if (accum == NULL)
    { 
        // no accum operator is present
        Op = "" ;
    }
    else
    { 
        // use a simpler version of accum->name
        if (accum->opcode == GB_USER_binop_code) Op = "op" ;
        else if (GB_STRING_MATCH (accum->name, "plus")) Op = "+" ;
        else if (GB_STRING_MATCH (accum->name, "minus")) Op = "-" ;
        else if (GB_STRING_MATCH (accum->name, "times")) Op = "*" ;
        else if (GB_STRING_MATCH (accum->name, "div")) Op = "/" ;
        else if (GB_STRING_MATCH (accum->name, "or")) Op = "|" ;
        else if (GB_STRING_MATCH (accum->name, "and")) Op = "&" ;
        else if (GB_STRING_MATCH (accum->name, "xor")) Op = "^" ;
        else Op = accum->name ;
    }

    //--------------------------------------------------------------------------
    // construct the Mask string
    //--------------------------------------------------------------------------

    #define GB_STRLEN 128
    const char *Mask ;
    char Mask_string [GB_STRLEN+1] ;
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
        snprintf (Mask_string, GB_STRLEN, "<%sM%s%s%s>",
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
            snprintf (str, slen, "C%s(i,%s) %s= A ", Mask, Jstr, Op) ;
            break ;

        case GB_COL_ASSIGN : 
            // C(I,j) = A
            snprintf (str, slen, "C%s(%s,j) %s= A ", Mask, Istr, Op) ;
            break ;

        case GB_ASSIGN : 
            // C<M>(I,J) = A
            if (Ikind == GB_ALL && Jkind == GB_ALL)
            { 
                // C<M> += A
                snprintf (str, slen, "C%s %s= %s ", Mask, Op, S) ;
            }
            else
            { 
                // C<M>(I,J) = A
                snprintf (str, slen, "C%s(%s,%s) %s= %s ", Mask, Istr, Jstr,
                    Op, S) ;
            }
            break ;

        case GB_SUBASSIGN : 
            // C(I,J)<M> = A
            if (Ikind == GB_ALL && Jkind == GB_ALL)
            { 
                // C<M> += A
                snprintf (str, slen, "C%s %s= %s ", Mask, Op, S) ;
            }
            else
            { 
                // C(I,J)<M> = A
                snprintf (str, slen, "C(%s,%s)%s %s= %s ", Istr, Jstr, Mask,
                    Op, S) ;
            }
            break ;

        default: ;
    }
}

