//------------------------------------------------------------------------------
// gb_string_and_type_to_binop_or_idxunop: get operator from a string and type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

// op_name: a built-in string defining the operator name:
//  1st, 2nd, any, pair (same as oneb), min, max, +, -, rminus, *, /, \
//  iseq, isne, isgt, islt, isge, isle,
//  ==, ~=, >, <, >=, <=,
//  ||, &&, xor, xnor
//  atan2, hypot, fmod, remainder, copysign, cmplx, pow, pow2

//  bitwise operators:
//      bitand, bitor, bitxor, bitxnor, bitget, bitset, bitclr, bitshift

// positional operators:
//      firsti0, firsti1, firstj0, firstj1, secondi0, secondi1, secondj0,
//      secondj1.  The default type is int64

// index unary operators:
//      tril, triu, diag, offdiag, diagindex, rowindex, rowle, rowgt,
//      colindex, colle, colgt

// The following synonyms are allowed for specifying these operators:
//
//      1st   first
//      2nd   second
//      pair  oneb
//      +     plus
//      -     minus
//      *     times
//      /     div
//      \     rdiv
//      ==    eq
//      ~=    ne
//      >     gt
//      <     lt
//      >=    ge
//      <=    le
//      ||    |     or  lor
//      &&    &     and land
//      xor   lxor
//      pow2  ldexp

GrB_BinaryOp gb_string_and_type_to_binop_or_idxunop
(
    const char *op_name,        // name of the operator, as a string
    const GrB_Type type,        // type of the x,y inputs to the operator
    const bool type_not_given,  // true if no type present in the string
    GrB_IndexUnaryOp *idxunop,          // idxunop from the string
    int64_t *ithunk                     // thunk for idxunop
)
{

    if (idxunop != NULL) (*idxunop) = NULL ;

    if (MATCH (op_name, "1st") || MATCH (op_name, "first"))
    { 

        if (type == GrB_BOOL  ) return (GrB_FIRST_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_FIRST_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_FIRST_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_FIRST_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_FIRST_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_FIRST_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_FIRST_UINT16) ;
        if (type == GrB_UINT32) return (GrB_FIRST_UINT32) ;
        if (type == GrB_UINT64) return (GrB_FIRST_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_FIRST_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_FIRST_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_FIRST_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_FIRST_FC64  ) ;

    }
    else if (MATCH (op_name, "2nd") || MATCH (op_name, "second"))
    { 

        if (type == GrB_BOOL  ) return (GrB_SECOND_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_SECOND_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_SECOND_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_SECOND_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_SECOND_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_SECOND_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_SECOND_UINT16) ;
        if (type == GrB_UINT32) return (GrB_SECOND_UINT32) ;
        if (type == GrB_UINT64) return (GrB_SECOND_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_SECOND_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_SECOND_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_SECOND_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_SECOND_FC64  ) ;

    }
    else if (MATCH (op_name, "any"))
    { 

        if (type == GrB_BOOL  ) return (GxB_ANY_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_ANY_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_ANY_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_ANY_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_ANY_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_ANY_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_ANY_UINT16) ;
        if (type == GrB_UINT32) return (GxB_ANY_UINT32) ;
        if (type == GrB_UINT64) return (GxB_ANY_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_ANY_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_ANY_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_ANY_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_ANY_FC64  ) ;

    }
    else if (MATCH (op_name, "pair") || MATCH (op_name, "oneb"))
    { 

        // GrB_ONEB is the new name for GxB_PAIR
        if (type == GrB_BOOL  ) return (GrB_ONEB_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_ONEB_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_ONEB_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_ONEB_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_ONEB_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_ONEB_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_ONEB_UINT16) ;
        if (type == GrB_UINT32) return (GrB_ONEB_UINT32) ;
        if (type == GrB_UINT64) return (GrB_ONEB_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_ONEB_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_ONEB_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_ONEB_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_ONEB_FC64  ) ;

    }
    else if (MATCH (op_name, "min"))
    { 

        if (type == GrB_BOOL  ) return (GrB_MIN_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_MIN_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_MIN_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_MIN_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_MIN_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_MIN_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_MIN_UINT16) ;
        if (type == GrB_UINT32) return (GrB_MIN_UINT32) ;
        if (type == GrB_UINT64) return (GrB_MIN_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_MIN_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_MIN_FP64  ) ;
        // no complex min

    }
    else if (MATCH (op_name, "max"))
    { 

        if (type == GrB_BOOL  ) return (GrB_MAX_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_MAX_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_MAX_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_MAX_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_MAX_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_MAX_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_MAX_UINT16) ;
        if (type == GrB_UINT32) return (GrB_MAX_UINT32) ;
        if (type == GrB_UINT64) return (GrB_MAX_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_MAX_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_MAX_FP64  ) ;
        // no complex max

    }
    else if (MATCH (op_name, "+") || MATCH (op_name, "plus"))
    { 

        if (type == GrB_BOOL  ) return (GrB_PLUS_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_PLUS_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_PLUS_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_PLUS_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_PLUS_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_PLUS_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_PLUS_UINT16) ;
        if (type == GrB_UINT32) return (GrB_PLUS_UINT32) ;
        if (type == GrB_UINT64) return (GrB_PLUS_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_PLUS_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_PLUS_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_PLUS_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_PLUS_FC64  ) ;

    }
    else if (MATCH (op_name, "-") || MATCH (op_name, "minus"))
    { 

        if (type == GrB_BOOL  ) return (GrB_MINUS_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_MINUS_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_MINUS_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_MINUS_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_MINUS_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_MINUS_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_MINUS_UINT16) ;
        if (type == GrB_UINT32) return (GrB_MINUS_UINT32) ;
        if (type == GrB_UINT64) return (GrB_MINUS_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_MINUS_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_MINUS_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_MINUS_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_MINUS_FC64  ) ;

    }
    else if (MATCH (op_name, "rminus"))
    { 

        if (type == GrB_BOOL  ) return (GxB_RMINUS_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_RMINUS_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_RMINUS_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_RMINUS_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_RMINUS_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_RMINUS_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_RMINUS_UINT16) ;
        if (type == GrB_UINT32) return (GxB_RMINUS_UINT32) ;
        if (type == GrB_UINT64) return (GxB_RMINUS_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_RMINUS_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_RMINUS_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_RMINUS_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_RMINUS_FC64  ) ;

    }
    else if (MATCH (op_name, "*") || MATCH (op_name, "times"))
    { 

        if (type == GrB_BOOL  ) return (GrB_TIMES_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_TIMES_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_TIMES_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_TIMES_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_TIMES_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_TIMES_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_TIMES_UINT16) ;
        if (type == GrB_UINT32) return (GrB_TIMES_UINT32) ;
        if (type == GrB_UINT64) return (GrB_TIMES_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_TIMES_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_TIMES_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_TIMES_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_TIMES_FC64  ) ;

    }
    else if (MATCH (op_name, "/") || MATCH (op_name, "div"))
    { 

        if (type == GrB_BOOL  ) return (GrB_DIV_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_DIV_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_DIV_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_DIV_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_DIV_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_DIV_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_DIV_UINT16) ;
        if (type == GrB_UINT32) return (GrB_DIV_UINT32) ;
        if (type == GrB_UINT64) return (GrB_DIV_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_DIV_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_DIV_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_DIV_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_DIV_FC64  ) ;

    }
    else if (MATCH (op_name, "\\") || MATCH (op_name, "rdiv"))
    { 

        if (type == GrB_BOOL  ) return (GxB_RDIV_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_RDIV_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_RDIV_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_RDIV_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_RDIV_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_RDIV_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_RDIV_UINT16) ;
        if (type == GrB_UINT32) return (GxB_RDIV_UINT32) ;
        if (type == GrB_UINT64) return (GxB_RDIV_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_RDIV_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_RDIV_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_RDIV_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_RDIV_FC64  ) ;

    }
    else if (MATCH (op_name, "iseq"))
    { 

        if (type == GrB_BOOL  ) return (GxB_ISEQ_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_ISEQ_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_ISEQ_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_ISEQ_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_ISEQ_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_ISEQ_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_ISEQ_UINT16) ;
        if (type == GrB_UINT32) return (GxB_ISEQ_UINT32) ;
        if (type == GrB_UINT64) return (GxB_ISEQ_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_ISEQ_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_ISEQ_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_ISEQ_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_ISEQ_FC64  ) ;

    }
    else if (MATCH (op_name, "isne"))
    { 

        if (type == GrB_BOOL  ) return (GxB_ISNE_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_ISNE_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_ISNE_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_ISNE_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_ISNE_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_ISNE_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_ISNE_UINT16) ;
        if (type == GrB_UINT32) return (GxB_ISNE_UINT32) ;
        if (type == GrB_UINT64) return (GxB_ISNE_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_ISNE_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_ISNE_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_ISNE_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_ISNE_FC64  ) ;

    }
    else if (MATCH (op_name, "isgt"))
    { 

        if (type == GrB_BOOL  ) return (GxB_ISGT_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_ISGT_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_ISGT_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_ISGT_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_ISGT_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_ISGT_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_ISGT_UINT16) ;
        if (type == GrB_UINT32) return (GxB_ISGT_UINT32) ;
        if (type == GrB_UINT64) return (GxB_ISGT_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_ISGT_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_ISGT_FP64  ) ;

    }
    else if (MATCH (op_name, "islt"))
    { 

        if (type == GrB_BOOL  ) return (GxB_ISLT_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_ISLT_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_ISLT_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_ISLT_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_ISLT_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_ISLT_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_ISLT_UINT16) ;
        if (type == GrB_UINT32) return (GxB_ISLT_UINT32) ;
        if (type == GrB_UINT64) return (GxB_ISLT_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_ISLT_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_ISLT_FP64  ) ;

    }
    else if (MATCH (op_name, "isge"))
    { 

        if (type == GrB_BOOL  ) return (GxB_ISGE_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_ISGE_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_ISGE_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_ISGE_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_ISGE_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_ISGE_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_ISGE_UINT16) ;
        if (type == GrB_UINT32) return (GxB_ISGE_UINT32) ;
        if (type == GrB_UINT64) return (GxB_ISGE_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_ISGE_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_ISGE_FP64  ) ;

    }
    else if (MATCH (op_name, "isle"))
    { 

        if (type == GrB_BOOL  ) return (GxB_ISLE_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_ISLE_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_ISLE_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_ISLE_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_ISLE_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_ISLE_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_ISLE_UINT16) ;
        if (type == GrB_UINT32) return (GxB_ISLE_UINT32) ;
        if (type == GrB_UINT64) return (GxB_ISLE_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_ISLE_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_ISLE_FP64  ) ;

    }
    else if (MATCH (op_name, "==") || MATCH (op_name, "eq"))
    { 

        if (type == GrB_BOOL  ) return (GrB_EQ_BOOL  ) ;    // == GrB_LXNOR
        if (type == GrB_INT8  ) return (GrB_EQ_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_EQ_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_EQ_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_EQ_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_EQ_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_EQ_UINT16) ;
        if (type == GrB_UINT32) return (GrB_EQ_UINT32) ;
        if (type == GrB_UINT64) return (GrB_EQ_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_EQ_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_EQ_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_EQ_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_EQ_FC64  ) ;

    }
    else if (MATCH (op_name, "~=") || MATCH (op_name, "ne"))
    { 

        if (type == GrB_BOOL  ) return (GrB_NE_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_NE_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_NE_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_NE_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_NE_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_NE_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_NE_UINT16) ;
        if (type == GrB_UINT32) return (GrB_NE_UINT32) ;
        if (type == GrB_UINT64) return (GrB_NE_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_NE_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_NE_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_NE_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_NE_FC64  ) ;

    }
    else if (MATCH (op_name, ">") || MATCH (op_name, "gt"))
    { 

        if (type == GrB_BOOL  ) return (GrB_GT_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_GT_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_GT_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_GT_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_GT_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_GT_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_GT_UINT16) ;
        if (type == GrB_UINT32) return (GrB_GT_UINT32) ;
        if (type == GrB_UINT64) return (GrB_GT_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_GT_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_GT_FP64  ) ;

    }
    else if (MATCH (op_name, "<") || MATCH (op_name, "lt"))
    { 

        if (type == GrB_BOOL  ) return (GrB_LT_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_LT_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_LT_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_LT_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_LT_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_LT_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_LT_UINT16) ;
        if (type == GrB_UINT32) return (GrB_LT_UINT32) ;
        if (type == GrB_UINT64) return (GrB_LT_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_LT_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_LT_FP64  ) ;

    }
    else if (MATCH (op_name, ">=") || MATCH (op_name, "ge"))
    { 

        if (type == GrB_BOOL  ) return (GrB_GE_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_GE_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_GE_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_GE_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_GE_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_GE_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_GE_UINT16) ;
        if (type == GrB_UINT32) return (GrB_GE_UINT32) ;
        if (type == GrB_UINT64) return (GrB_GE_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_GE_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_GE_FP64  ) ;

    }
    else if (MATCH (op_name, "<=") || MATCH (op_name, "le"))
    { 

        if (type == GrB_BOOL  ) return (GrB_LE_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_LE_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_LE_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_LE_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_LE_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_LE_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_LE_UINT16) ;
        if (type == GrB_UINT32) return (GrB_LE_UINT32) ;
        if (type == GrB_UINT64) return (GrB_LE_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_LE_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_LE_FP64  ) ;

    }
    else if (MATCH (op_name, "||") || MATCH (op_name, "|")  ||
             MATCH (op_name, "or") || MATCH (op_name, "lor"))
    { 

        if (type == GrB_BOOL  ) return (GrB_LOR       ) ;
        if (type == GrB_INT8  ) return (GxB_LOR_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_LOR_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_LOR_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_LOR_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_LOR_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_LOR_UINT16) ;
        if (type == GrB_UINT32) return (GxB_LOR_UINT32) ;
        if (type == GrB_UINT64) return (GxB_LOR_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_LOR_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_LOR_FP64  ) ;

    }
    else if (MATCH (op_name, "&&")  || MATCH (op_name, "&")   ||
             MATCH (op_name, "and") || MATCH (op_name, "land"))
    { 

        if (type == GrB_BOOL  ) return (GrB_LAND       ) ;
        if (type == GrB_INT8  ) return (GxB_LAND_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_LAND_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_LAND_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_LAND_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_LAND_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_LAND_UINT16) ;
        if (type == GrB_UINT32) return (GxB_LAND_UINT32) ;
        if (type == GrB_UINT64) return (GxB_LAND_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_LAND_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_LAND_FP64  ) ;

    }
    else if (MATCH (op_name, "xor") || MATCH (op_name, "lxor"))
    { 

        if (type == GrB_BOOL  ) return (GrB_LXOR       ) ;
        if (type == GrB_INT8  ) return (GxB_LXOR_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_LXOR_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_LXOR_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_LXOR_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_LXOR_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_LXOR_UINT16) ;
        if (type == GrB_UINT32) return (GxB_LXOR_UINT32) ;
        if (type == GrB_UINT64) return (GxB_LXOR_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_LXOR_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_LXOR_FP64  ) ;

    }
    else if (MATCH (op_name, "lxnor") || MATCH (op_name, "xnor"))
    { 

        if (type == GrB_BOOL  ) return (GrB_LXNOR ) ; // == GrB_EQ_BOOL

    }
    else if (MATCH (op_name, "atan2"))
    { 

        if (type == GrB_FP32  ) return (GxB_ATAN2_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_ATAN2_FP64  ) ;

    }
    else if (MATCH (op_name, "hypot"))
    { 

        if (type == GrB_FP32  ) return (GxB_HYPOT_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_HYPOT_FP64  ) ;

    }
    else if (MATCH (op_name, "fmod"))
    { 

        if (type == GrB_FP32  ) return (GxB_FMOD_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_FMOD_FP64  ) ;

    }
    else if (MATCH (op_name, "remainder"))
    { 

        if (type == GrB_FP32  ) return (GxB_REMAINDER_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_REMAINDER_FP64  ) ;

    }
    else if (MATCH (op_name, "copysign"))
    { 

        if (type == GrB_FP32  ) return (GxB_COPYSIGN_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_COPYSIGN_FP64  ) ;

    }
    else if (MATCH (op_name, "cmplx"))
    { 

        if (type == GrB_FP32  ) return (GxB_CMPLX_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_CMPLX_FP64  ) ;

    }
    else if (MATCH (op_name, "ldexp") || MATCH (op_name, "pow2"))
    { 

        if (type == GrB_FP32  ) return (GxB_LDEXP_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_LDEXP_FP64  ) ;

    }
    else if (MATCH (op_name, "pow"))
    { 

        if (type == GrB_BOOL  ) return (GxB_POW_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_POW_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_POW_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_POW_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_POW_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_POW_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_POW_UINT16) ;
        if (type == GrB_UINT32) return (GxB_POW_UINT32) ;
        if (type == GrB_UINT64) return (GxB_POW_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_POW_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_POW_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_POW_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_POW_FC64  ) ;

    }
    else if (MATCH (op_name, "bitor"))
    { 

        if (type == GrB_INT8  ) return (GrB_BOR_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_BOR_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_BOR_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_BOR_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_BOR_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_BOR_UINT16) ;
        if (type == GrB_UINT32) return (GrB_BOR_UINT32) ;
        if (type == GrB_UINT64) return (GrB_BOR_UINT64) ;

    }
    else if (MATCH (op_name, "bitand"))
    { 

        if (type == GrB_INT8  ) return (GrB_BAND_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_BAND_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_BAND_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_BAND_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_BAND_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_BAND_UINT16) ;
        if (type == GrB_UINT32) return (GrB_BAND_UINT32) ;
        if (type == GrB_UINT64) return (GrB_BAND_UINT64) ;

    }
    else if (MATCH (op_name, "bitxor"))
    { 

        if (type == GrB_INT8  ) return (GrB_BXOR_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_BXOR_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_BXOR_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_BXOR_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_BXOR_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_BXOR_UINT16) ;
        if (type == GrB_UINT32) return (GrB_BXOR_UINT32) ;
        if (type == GrB_UINT64) return (GrB_BXOR_UINT64) ;

    }
    else if (MATCH (op_name, "bitxnor"))
    { 

        if (type == GrB_INT8  ) return (GrB_BXNOR_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_BXNOR_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_BXNOR_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_BXNOR_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_BXNOR_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_BXNOR_UINT16) ;
        if (type == GrB_UINT32) return (GrB_BXNOR_UINT32) ;
        if (type == GrB_UINT64) return (GrB_BXNOR_UINT64) ;

    }
    else if (MATCH (op_name, "bitget"))
    { 

        if (type == GrB_INT8  ) return (GxB_BGET_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_BGET_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_BGET_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_BGET_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_BGET_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_BGET_UINT16) ;
        if (type == GrB_UINT32) return (GxB_BGET_UINT32) ;
        if (type == GrB_UINT64) return (GxB_BGET_UINT64) ;

    }
    else if (MATCH (op_name, "bitset"))
    { 

        if (type == GrB_INT8  ) return (GxB_BSET_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_BSET_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_BSET_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_BSET_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_BSET_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_BSET_UINT16) ;
        if (type == GrB_UINT32) return (GxB_BSET_UINT32) ;
        if (type == GrB_UINT64) return (GxB_BSET_UINT64) ;

    }
    else if (MATCH (op_name, "bitclr"))
    { 

        if (type == GrB_INT8  ) return (GxB_BCLR_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_BCLR_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_BCLR_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_BCLR_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_BCLR_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_BCLR_UINT16) ;
        if (type == GrB_UINT32) return (GxB_BCLR_UINT32) ;
        if (type == GrB_UINT64) return (GxB_BCLR_UINT64) ;

    }
    else if (MATCH (op_name, "bitshift"))
    { 

        if (type == GrB_INT8  ) return (GxB_BSHIFT_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_BSHIFT_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_BSHIFT_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_BSHIFT_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_BSHIFT_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_BSHIFT_UINT16) ;
        if (type == GrB_UINT32) return (GxB_BSHIFT_UINT32) ;
        if (type == GrB_UINT64) return (GxB_BSHIFT_UINT64) ;

    }
    else if (MATCH (op_name, "firsti0" ) || MATCH (op_name, "1sti0"))
    {

        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_FIRSTI_INT64 ) ;
        if (type == GrB_INT32) return (GxB_FIRSTI_INT32 ) ;

    }
    else if (MATCH (op_name, "firsti1" ) || MATCH (op_name, "1sti1") ||
             MATCH (op_name, "firsti"  ) || MATCH (op_name, "1sti"))
    {

        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_FIRSTI1_INT64 ) ;
        if (type == GrB_INT32) return (GxB_FIRSTI1_INT32 ) ;

    }
    else if (MATCH (op_name, "firstj0" ) || MATCH (op_name, "1stj0"))
    {

        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_FIRSTJ_INT64 ) ;
        if (type == GrB_INT32) return (GxB_FIRSTJ_INT32 ) ;

    }
    else if (MATCH (op_name, "firstj1" ) || MATCH (op_name, "1stj1") ||
             MATCH (op_name, "firstj"  ) || MATCH (op_name, "1stj"))
    {

        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_FIRSTJ1_INT64 ) ;
        if (type == GrB_INT32) return (GxB_FIRSTJ1_INT32 ) ;

    }
    else if (MATCH (op_name, "secondi0") || MATCH (op_name, "2ndi0"))
    {

        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_SECONDI_INT64 ) ;
        if (type == GrB_INT32) return (GxB_SECONDI_INT32 ) ;

    }
    else if (MATCH (op_name, "secondi1") || MATCH (op_name, "2ndi1") ||
             MATCH (op_name, "secondi" ) || MATCH (op_name, "2ndi"))
    {

        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_SECONDI1_INT64 ) ;
        if (type == GrB_INT32) return (GxB_SECONDI1_INT32 ) ;

    }
    else if (MATCH (op_name, "secondj0" ) || MATCH (op_name, "2ndj0"))
    {

        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_SECONDJ_INT64 ) ;
        if (type == GrB_INT32) return (GxB_SECONDJ_INT32 ) ;

    }
    else if (MATCH (op_name, "secondj1") || MATCH (op_name, "2ndj1") ||
             MATCH (op_name, "secondj" ) || MATCH (op_name, "2ndj"))
    {

        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_SECONDJ1_INT64 ) ;
        if (type == GrB_INT32) return (GxB_SECONDJ1_INT32 ) ;

    }
    else if (MATCH (op_name, "ignore") || MATCH (op_name, "ignore_dup"))
    {
        // valid for build only
        return (GxB_IGNORE_DUP) ;
    }

    //--------------------------------------------------------------------------
    // return an idxunop
    //--------------------------------------------------------------------------

    bool is32 = (type == GrB_INT32) ;

    if (idxunop != NULL)
    {

        CHECK_ERROR (ithunk == NULL, "thunk missing") ;

        if (MATCH (op_name, "tril"))
        { 
            (*idxunop) = GrB_TRIL ;
        }
        else if (MATCH (op_name, "triu"))
        { 
            (*idxunop) = GrB_TRIU ;
        }
        else if (MATCH (op_name, "diag"))
        {
            (*idxunop) = GrB_DIAG ;
        }
        else if (MATCH (op_name, "diagindex"))
        { 
            (*idxunop) = is32 ? GrB_DIAGINDEX_INT32 : GrB_DIAGINDEX_INT64 ;
        }
        else if (MATCH (op_name, "offdiag"))
        { 
            (*idxunop) = GrB_OFFDIAG ;
        }
        else if (MATCH (op_name, "rowindex"))
        { 
            (*idxunop) = is32 ? GrB_ROWINDEX_INT32 : GrB_ROWINDEX_INT64 ;
            (*ithunk)++ ;
        }
        else if (MATCH (op_name, "rowle"))
        { 
            (*idxunop) = GrB_ROWLE ;
            (*ithunk)-- ;
        }
        else if (MATCH (op_name, "rowgt"))
        { 
            (*idxunop) = GrB_ROWGT ;
            (*ithunk)-- ;
        }
        else if (MATCH (op_name, "colindex"))
        { 
            (*idxunop) = is32 ? GrB_COLINDEX_INT32 : GrB_COLINDEX_INT64 ;
            (*ithunk)++ ;
        }
        else if (MATCH (op_name, "colle"))
        { 
            (*idxunop) = GrB_COLLE ;
            (*ithunk)-- ;
        }
        else if (MATCH (op_name, "colgt"))
        { 
            (*idxunop) = GrB_COLGT ;
            (*ithunk)-- ;
        }

        if ((*idxunop) != NULL)
        {
            // this is not an error condition
            return (NULL) ;
        }
    }

    //--------------------------------------------------------------------------
    // unknown type or operator
    //--------------------------------------------------------------------------

    // the type can be NULL for positional operators, but no others

    CHECK_ERROR (type == NULL, "unknown type") ;
    ERROR2 ("unknown operator", op_name) ;
    return (NULL) ;
}

