//------------------------------------------------------------------------------
// GB_macrofy_family: construct all macros for all methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_family
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    GB_jit_family family,       // family to macrofy
    uint64_t method_code,       // encoding of the specific problem
    uint64_t kcode,             // kernel code
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
            GB_macrofy_apply (fp, method_code, op, type1, type2) ;
            break ;

        case GB_jit_assign_family : 
            GB_macrofy_assign (fp, method_code, (GrB_BinaryOp) op, type1, type2) ;
            break ;

        case GB_jit_build_family  : 
            GB_macrofy_build (fp, method_code, (GrB_BinaryOp) op, type1, type2) ;
            break ;

        case GB_jit_ewise_family  : 
            GB_macrofy_ewise (fp, method_code, kcode, (GrB_BinaryOp) op, type1, type2,
                type3) ;
            break ;

        case GB_jit_mxm_family    : 
            GB_macrofy_mxm (fp, method_code, semiring, type1, type2, type3) ;
            break ;

        case GB_jit_reduce_family : 
            GB_macrofy_reduce (fp, method_code, monoid, type1) ;
            break ;

        case GB_jit_select_family : 
            GB_macrofy_select (fp, method_code, (GrB_IndexUnaryOp) op, type1) ;
            break ;

        case GB_jit_user_op_family : 
            GB_macrofy_user_op (fp, op) ;
            break ;

        case GB_jit_user_type_family : 
            GB_macrofy_user_type (fp, type1) ;
            break ;

        case GB_jit_masker_family : 
            GB_macrofy_masker (fp, method_code, type1) ;
            break ;

        case GB_jit_subref_family : 
            GB_macrofy_subref (fp, method_code, type1) ;
            break ;

        case GB_jit_sort_family : 
            GB_macrofy_sort (fp, method_code, op, type1) ;
            break ;

        default: ;
    }
}

