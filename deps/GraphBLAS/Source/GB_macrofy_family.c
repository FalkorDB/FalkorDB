//------------------------------------------------------------------------------
// GB_macrofy_family: construct all macros for all methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_family
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    GB_jit_family family,       // family to macrofy
    uint64_t scode,             // encoding of the specific problem
    GrB_Semiring semiring,      // semiring (for mxm family only)
    GrB_Monoid monoid,          // monoid (for reduce family only)
    GB_Operator op,             // unary/index_unary/binary op
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
)
{

    switch (family)
    {

        case GB_jit_apply_family  : 
            GB_macrofy_apply (fp, scode, op, type1, type2) ;
            break ;

        case GB_jit_assign_family : 
            GB_macrofy_assign (fp, scode, (GrB_BinaryOp) op, type1, type2) ;
            break ;

        case GB_jit_build_family  : 
            GB_macrofy_build (fp, scode, (GrB_BinaryOp) op, type1, type2) ;
            break ;

        case GB_jit_ewise_family  : 
            GB_macrofy_ewise (fp, scode, (GrB_BinaryOp) op, type1, type2,
                type3) ;
            break ;

        case GB_jit_mxm_family    : 
            GB_macrofy_mxm (fp, scode, semiring, type1, type2, type3) ;
            break ;

        case GB_jit_reduce_family : 
            GB_macrofy_reduce (fp, scode, monoid, type1) ;
            break ;

        case GB_jit_select_family : 
            GB_macrofy_select (fp, scode, (GrB_IndexUnaryOp) op, type1) ;
            break ;

        case GB_jit_user_op_family  : 
            GB_macrofy_user_op (fp, op) ;
            break ;

        case GB_jit_user_type_family  : 
            GB_macrofy_user_type (fp, type1) ;
            break ;

        default: ;
    }
}

