//------------------------------------------------------------------------------
// GB_builtin.c: built-in types, functions, operators, and other externs
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file defines the predefined built-in types, descriptors, unary
// operators, index_unary operators, binary operators, monoids, and semirings.

#include "GB.h"
#include "math/GB_math.h"
#include "builtin/GB_builtin.h"

//------------------------------------------------------------------------------
// compiler flags
//------------------------------------------------------------------------------

#if GB_COMPILER_ICC || GB_COMPILER_ICX
    // disable icc warnings
    //  144:  initialize with incompatible pointer
    #pragma warning (disable: 144 )
#elif GB_COMPILER_GCC
    // disable gcc warnings
    #pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
#elif GB_COMPILER_MSC
    // disable MS Visual Studio warnings
    GB_PRAGMA (warning (disable : 4146 ))
#endif

//------------------------------------------------------------------------------
// built-in types
//------------------------------------------------------------------------------

#define GB_TYPEDEF(prefix,type,ctype,name)                                  \
    struct GB_Type_opaque GB_OPAQUE (type) =                                \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        sizeof (ctype),             /* sizeof the type */                   \
        GB_ ## type ## _code,       /* type code */                         \
        0, name,                    /* name_len and name */                 \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        NULL                        /* print function (user-defn types) */  \
    } ;                                                                     \
    GrB_Type prefix ## _ ## type = & GB_OPAQUE (type)

GB_TYPEDEF (GrB, BOOL  , bool      , "bool"      ) ;
GB_TYPEDEF (GrB, INT8  , int8_t    , "int8_t"    ) ;
GB_TYPEDEF (GrB, INT16 , int16_t   , "int16_t"   ) ;
GB_TYPEDEF (GrB, INT32 , int32_t   , "int32_t"   ) ;
GB_TYPEDEF (GrB, INT64 , int64_t   , "int64_t"   ) ;
GB_TYPEDEF (GrB, UINT8 , uint8_t   , "uint8_t"   ) ;
GB_TYPEDEF (GrB, UINT16, uint16_t  , "uint16_t"  ) ;
GB_TYPEDEF (GrB, UINT32, uint32_t  , "uint32_t"  ) ;
GB_TYPEDEF (GrB, UINT64, uint64_t  , "uint64_t"  ) ;
GB_TYPEDEF (GrB, FP32  , float     , "float"     ) ;
GB_TYPEDEF (GrB, FP64  , double    , "double"    ) ;
GB_TYPEDEF (GxB, FC32  , GxB_FC32_t, "GxB_FC32_t") ;
GB_TYPEDEF (GxB, FC64  , GxB_FC64_t, "GxB_FC64_t") ;

//------------------------------------------------------------------------------
// built-in descriptors
//------------------------------------------------------------------------------

#define o ((GrB_Desc_Value) GxB_DEFAULT)

#define GB_DESC(name,out,mask,in0,in1)                                      \
    struct GB_Descriptor_opaque GB_OPAQUE (desc_ ## name) =                 \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        "", 0,                      /* logger */                            \
        (GrB_Desc_Value) (out),                                             \
        (GrB_Desc_Value) (mask),                                            \
        (GrB_Desc_Value) (in0),                                             \
        (GrB_Desc_Value) (in1),                                             \
        o,                          /* default: axb */                      \
        0,                          /* default compression */               \
        0,                          /* no sort */                           \
        0,                          /* import */                            \
        0, 0, 0                     /* row_list, col_list, val_list */      \
    } ;                                                                     \
    GrB_Descriptor GRB (DESC_ ## name) = & GB_OPAQUE (desc_ ## name) ;

//       name     outp         structure     complement  in0       in1

// GrB_NULL     , o          , o             + o       , o       , o
GB_DESC (T1     , o          , o             + o       , o       , GrB_TRAN )
GB_DESC (T0     , o          , o             + o       , GrB_TRAN, o        )
GB_DESC (T0T1   , o          , o             + o       , GrB_TRAN, GrB_TRAN )

GB_DESC (C      , o          , o             + GrB_COMP, o       , o        )
GB_DESC (CT1    , o          , o             + GrB_COMP, o       , GrB_TRAN )
GB_DESC (CT0    , o          , o             + GrB_COMP, GrB_TRAN, o        )
GB_DESC (CT0T1  , o          , o             + GrB_COMP, GrB_TRAN, GrB_TRAN )

GB_DESC (S      , o          , GrB_STRUCTURE + o       , o       , o        )
GB_DESC (ST1    , o          , GrB_STRUCTURE + o       , o       , GrB_TRAN )
GB_DESC (ST0    , o          , GrB_STRUCTURE + o       , GrB_TRAN, o        )
GB_DESC (ST0T1  , o          , GrB_STRUCTURE + o       , GrB_TRAN, GrB_TRAN )

GB_DESC (SC     , o          , GrB_STRUCTURE + GrB_COMP, o       , o        )
GB_DESC (SCT1   , o          , GrB_STRUCTURE + GrB_COMP, o       , GrB_TRAN )
GB_DESC (SCT0   , o          , GrB_STRUCTURE + GrB_COMP, GrB_TRAN, o        )
GB_DESC (SCT0T1 , o          , GrB_STRUCTURE + GrB_COMP, GrB_TRAN, GrB_TRAN )

GB_DESC (R      , GrB_REPLACE, o             + o       , o       , o        )
GB_DESC (RT1    , GrB_REPLACE, o             + o       , o       , GrB_TRAN )
GB_DESC (RT0    , GrB_REPLACE, o             + o       , GrB_TRAN, o        )
GB_DESC (RT0T1  , GrB_REPLACE, o             + o       , GrB_TRAN, GrB_TRAN )

GB_DESC (RC     , GrB_REPLACE, o             + GrB_COMP, o       , o        )
GB_DESC (RCT1   , GrB_REPLACE, o             + GrB_COMP, o       , GrB_TRAN )
GB_DESC (RCT0   , GrB_REPLACE, o             + GrB_COMP, GrB_TRAN, o        )
GB_DESC (RCT0T1 , GrB_REPLACE, o             + GrB_COMP, GrB_TRAN, GrB_TRAN )

GB_DESC (RS     , GrB_REPLACE, GrB_STRUCTURE + o       , o       , o        )
GB_DESC (RST1   , GrB_REPLACE, GrB_STRUCTURE + o       , o       , GrB_TRAN )
GB_DESC (RST0   , GrB_REPLACE, GrB_STRUCTURE + o       , GrB_TRAN, o        )
GB_DESC (RST0T1 , GrB_REPLACE, GrB_STRUCTURE + o       , GrB_TRAN, GrB_TRAN )

GB_DESC (RSC    , GrB_REPLACE, GrB_STRUCTURE + GrB_COMP, o       , o        )
GB_DESC (RSCT1  , GrB_REPLACE, GrB_STRUCTURE + GrB_COMP, o       , GrB_TRAN )
GB_DESC (RSCT0  , GrB_REPLACE, GrB_STRUCTURE + GrB_COMP, GrB_TRAN, o        )
GB_DESC (RSCT0T1, GrB_REPLACE, GrB_STRUCTURE + GrB_COMP, GrB_TRAN, GrB_TRAN )

#undef o

//------------------------------------------------------------------------------
// GB_OP_NAME: construct the name of an operator
//------------------------------------------------------------------------------

#define GB_OP_NAME(op) GB_EVAL3 (op, _, GB_XTYPE)

//------------------------------------------------------------------------------
// helper macros to define unary operators
//------------------------------------------------------------------------------

#define GB_OP1zx(op,name,z_t,ztype,x_t,xtype)                               \
    extern void GB_FUNC_T (op, xtype) (z_t *z, const x_t *x) ;              \
    struct GB_UnaryOp_opaque GB_OPAQUE (GB_OP_NAME (op)) =                  \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (ztype),        /* ztype */                             \
        & GB_OPAQUE (xtype),        /* xtype */                             \
        NULL,                       /* ytype */                             \
        (GxB_unary_function) (& GB_FUNC_T (op, xtype)), NULL, NULL,         \
        name, 0,                    /* name and name_len */                 \
        GB_ ## op ## _unop_code,    /* opcode */                            \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        NULL, NULL, NULL, 0         /* theta_type, etc */                   \
    }

#define GRB_OP1z(op,name,z_t,ztype)                                         \
    GB_OP1zx (op, name, z_t, ztype, GB_TYPE, GB_XTYPE) ;                    \
    GrB_UnaryOp GRB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

#define GRB_OP1(op,name) GRB_OP1z (op, name, GB_TYPE, GB_XTYPE)

#define GXB_OP1z(op,name,z_t,ztype)                                         \
    GB_OP1zx (op, name, z_t, ztype, GB_TYPE, GB_XTYPE) ;                    \
    GrB_UnaryOp GXB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

#define GXB_OP1(op,name) GXB_OP1z (op, name, GB_TYPE, GB_XTYPE)

#define GXB_OP1_RENAME(op)                                                  \
    GrB_UnaryOp GXB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

//------------------------------------------------------------------------------
// helper macros to define binary operators
//------------------------------------------------------------------------------

#define GB_OP2zxy(op,name,z_t,ztype,x_t,xtype,y_t,ytype)                    \
    extern void GB_FUNC_T(op,xtype) (z_t *z, const x_t *x, const y_t *y) ;  \
    struct GB_BinaryOp_opaque GB_OPAQUE (GB_OP_NAME (op)) =                 \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (ztype),        /* ztype */                             \
        & GB_OPAQUE (xtype),                                                \
        & GB_OPAQUE (ytype),                                                \
        NULL, NULL, (GxB_binary_function) (& GB_FUNC_T (op, xtype)),        \
        name, 0,                    /* name and name_len */                 \
        GB_ ## op ## _binop_code,   /* opcode */                            \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        NULL, NULL, NULL, 0         /* theta_type, etc */                   \
    }

#define GRB_OP2z(op,name,z_t,ztype)                                          \
    GB_OP2zxy (op, name, z_t, ztype, GB_TYPE, GB_XTYPE, GB_TYPE, GB_XTYPE) ; \
    GrB_BinaryOp GRB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

#define GRB_OP2(op,name) GRB_OP2z (op, name, GB_TYPE, GB_XTYPE)

#define GXB_OP2z(op,name,z_t,ztype)                                          \
    GB_OP2zxy (op, name, z_t, ztype, GB_TYPE, GB_XTYPE, GB_TYPE, GB_XTYPE) ; \
    GrB_BinaryOp GXB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

#define GXB_OP2(op,name) GXB_OP2z (op, name, GB_TYPE, GB_XTYPE)

#define GXB_OP2shift(op,name) \
    GB_OP2zxy (op, name, GB_TYPE, GB_XTYPE, GB_TYPE, GB_XTYPE, int8_t, INT8) ; \
    GrB_BinaryOp GXB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

//------------------------------------------------------------------------------
// positional unary operators
//------------------------------------------------------------------------------

// The function pointer inside a positional unary operator cannot be called
// directly, since it does not depend on the values of its two arguments.  The
// operator can only be implemented via its opcode.

// helper macros to define positional unary operators
#define GXB_OP1_POS(op,name,type)                                           \
    struct GB_UnaryOp_opaque GB_OPAQUE (op ## _ ## type) =                  \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (type),         /* ztype */                             \
        & GB_OPAQUE (type),         /* xtype */                             \
        NULL,                       /* ytype */                             \
        NULL, NULL, NULL,           /* no function pointer */               \
        name, 0,                    /* name and name_len */                 \
        GB_ ## op ## _unop_code,    /* opcode */                            \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        NULL, NULL, NULL, 0         /* theta_type, etc */                   \
    } ;                                                                     \
    GrB_UnaryOp GXB (op ## _ ## type) = & GB_OPAQUE (op ## _ ## type)

GXB_OP1_POS (POSITIONI , "positioni" , INT32) ;
GXB_OP1_POS (POSITIONI , "positioni" , INT64) ;
GXB_OP1_POS (POSITIONI1, "positioni1", INT32) ;
GXB_OP1_POS (POSITIONI1, "positioni1", INT64) ;
GXB_OP1_POS (POSITIONJ , "positionj" , INT32) ;
GXB_OP1_POS (POSITIONJ , "positionj" , INT64) ;
GXB_OP1_POS (POSITIONJ1, "positionj1", INT32) ;
GXB_OP1_POS (POSITIONJ1, "positionj1", INT64) ;

//------------------------------------------------------------------------------
// built-in binary operators based on an internal index-binary op
//------------------------------------------------------------------------------

// This macro creates the FIRSTI, SECONDI, and related GrB_BinaryOps.  They
// are built as if they came from a built-in index binary op, but the
// corresponding GxB_IndexBinaryOp is not actually defined.  Instead, it is
// entirely encapsulated inside these GrB_BinaryOps.  None of these ops use
// their theta value; the offset of +1 for FIRSTI1 is built into the operator
// itself as z=ix+1; it is not computed as z = (ix)+theta with theta = 1.

// helper macros to define binary operators based on an index-binary op
#define GXB_OP2_POS(op,name)                                                \
    extern void GB_FUNC_T(op,GB_XTYPE) (GB_TYPE *z,                         \
        const void *x, uint64_t ix, uint64_t jx,                            \
        const void *y, uint64_t iy, uint64_t jy,                            \
        const void *theta_parameter) ;                                      \
    GB_TYPE GB_OPAQUE (GB_EVAL3 (op, GB_XTYPE, _theta)) = 0 ;               \
    struct GB_BinaryOp_opaque GB_OPAQUE (GB_OP_NAME (op)) =                 \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (GB_XTYPE),     /* ztype */                             \
        & GB_OPAQUE (GB_XTYPE),     /* xtype */                             \
        & GB_OPAQUE (GB_XTYPE),     /* ytype */                             \
        NULL, NULL, NULL,           /* no function pointer */               \
        name, 0,                    /* name and name_len */                 \
        GB_ ## op ## _binop_code,   /* opcode */                            \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        & GB_OPAQUE (GB_XTYPE),     /* theta_type */                        \
        (GxB_index_binary_function) (& GB_FUNC_T (op, GB_XTYPE)), /* func */\
        & GB_OPAQUE (GB_EVAL3 (op, GB_XTYPE, _theta)),     /* theta = 0 */  \
        0                           /* theta_size */                        \
    } ;                                                                     \
    GrB_BinaryOp GXB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

//------------------------------------------------------------------------------
// built-in index_unary operators
//------------------------------------------------------------------------------

// IndexUnaryOps that depend on i,j,y but not A(i,j), and result has
// the same type as the scalar y: ROWINDEX, COLINDEX, DIAGINDEX
#define GRB_IDXOP_POSITIONAL(op,name)                                       \
    extern void GB_FUNC_T(op,GB_XTYPE) (GB_TYPE *z, const void *unused,     \
        uint64_t i, uint64_t j, const GB_TYPE *y) ;                         \
    struct GB_IndexUnaryOp_opaque GB_OPAQUE (GB_OP_NAME (op)) =             \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (GB_XTYPE),     /* ztype */                             \
        NULL,                       /* xtype */                             \
        & GB_OPAQUE (GB_XTYPE),     /* ytype */                             \
        NULL, (GxB_index_unary_function) (& GB_FUNC_T (op, GB_XTYPE)), NULL,\
        name, 0,                    /* name and name_len */                 \
        GB_ ## op ## _idxunop_code, /* opcode */                            \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        NULL, NULL, NULL, 0         /* theta_type, etc */                   \
    } ;                                                                     \
    GrB_IndexUnaryOp GRB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

// GxB_IndexUnaryOps that depend on i,j,y but not A(i,j), and result has
// the same type as the scalar y: FLIPDIAGINDEX
#define GXB_IDXOP_POSITIONAL(op,name)                                       \
    extern void GB_FUNC_T(op,GB_XTYPE) (GB_TYPE *z, const void *unused,     \
        uint64_t i, uint64_t j, const GB_TYPE *y) ;                         \
    struct GB_IndexUnaryOp_opaque GB_OPAQUE (GB_OP_NAME (op)) =             \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (GB_XTYPE),     /* ztype */                             \
        NULL,                       /* xtype */                             \
        & GB_OPAQUE (GB_XTYPE),     /* ytype */                             \
        NULL, (GxB_index_unary_function) (& GB_FUNC_T (op, GB_XTYPE)), NULL,\
        name, 0,                    /* name and name_len */                 \
        GB_ ## op ## _idxunop_code, /* opcode */                            \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        NULL, NULL, NULL, 0         /* theta_type, etc */                   \
    } ;                                                                     \
    GrB_IndexUnaryOp GXB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

// IndexUnaryOps that depend on i,j, and y but not A(i,j), and result is
// bool: TRIL, TRIU, DIAG, OFFDIAG, COLLE, COLGT, ROWLE, ROWGT.
// No suffix on the GrB name.
#define GRB_IDXOP_POSITIONAL_BOOL(op,name)                                  \
    extern void GB_FUNC_T(op,GB_XTYPE) (bool *z, const void *unused,        \
        uint64_t i, uint64_t j, const GB_TYPE *y) ;                         \
    struct GB_IndexUnaryOp_opaque GB_OPAQUE (GB_OP_NAME (op)) =             \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (BOOL),         /* ztype */                             \
        NULL,                       /* xtype */                             \
        & GB_OPAQUE (GB_XTYPE),     /* ytype */                             \
        NULL, (GxB_index_unary_function) (& GB_FUNC_T (op, GB_XTYPE)), NULL,\
        name, 0,                    /* name and name_len */                 \
        GB_ ## op ## _idxunop_code, /* opcode */                            \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        NULL, NULL, NULL, 0         /* theta_type, etc */                   \
    } ;                                                                     \
    GrB_IndexUnaryOp GRB (op) = & GB_OPAQUE (GB_OP_NAME (op))

// GrB_IndexUnaryOps that depend on A(i,j), and result is bool: VALUE* ops
#define GRB_IDXOP_VALUE(op,name)                                            \
    extern void GB_FUNC_T(op,GB_XTYPE) (bool *z, const GB_TYPE *x,          \
        uint64_t i_unused, uint64_t j_unused, const GB_TYPE *y) ;           \
    struct GB_IndexUnaryOp_opaque GB_OPAQUE (GB_OP_NAME (op)) =             \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (BOOL),         /* ztype */                             \
        & GB_OPAQUE (GB_XTYPE),     /* xtype */                             \
        & GB_OPAQUE (GB_XTYPE),     /* ytype */                             \
        NULL, (GxB_index_unary_function) (& GB_FUNC_T (op, GB_XTYPE)), NULL,\
        name, 0,                    /* name and name_len */                 \
        GB_ ## op ## _idxunop_code, /* opcode */                            \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        NULL, NULL, NULL, 0         /* theta_type, etc */                   \
    } ;                                                                     \
    GrB_IndexUnaryOp GRB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

// GxB* IndexUnaryOps that depend on A(i,j), result is bool: VALUE* complex ops
#define GXB_IDXOP_VALUE(op,name)                                            \
    extern void GB_FUNC_T(op,GB_XTYPE) (bool *z, const GB_TYPE *x,          \
        uint64_t i_unused, uint64_t j_unused, const GB_TYPE *y) ;           \
    struct GB_IndexUnaryOp_opaque GB_OPAQUE (GB_OP_NAME (op)) =             \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (BOOL),         /* ztype */                             \
        & GB_OPAQUE (GB_XTYPE),     /* xtype */                             \
        & GB_OPAQUE (GB_XTYPE),     /* ytype */                             \
        NULL, (GxB_index_unary_function) (& GB_FUNC_T (op, GB_XTYPE)), NULL,\
        name, 0,                    /* name and name_len */                 \
        GB_ ## op ## _idxunop_code, /* opcode */                            \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        NULL, NULL, NULL, 0         /* theta_type, etc */                   \
    } ;                                                                     \
    GrB_IndexUnaryOp GXB (GB_OP_NAME (op)) = & GB_OPAQUE (GB_OP_NAME (op))

//------------------------------------------------------------------------------
// built-in select operators (DEPRECATED: do not use in any code)
//------------------------------------------------------------------------------

#define GXB_SEL(op,name)                                                    \
    struct GB_SelectOp_opaque GB_OPAQUE (op) =                              \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (BOOL),         /* ztype */                             \
        NULL,                       /* xtype */                             \
        NULL,                       /* thunk type */                        \
        NULL, NULL, NULL,           /* no function pointer */               \
        name, 0,                    /* name and name_len */                 \
        GB_ ## op ## _selop_code,   /* opcode */                            \
        NULL, 0, 0,                 /* defn, alloc, hash */                 \
        NULL, NULL, NULL, 0         /* theta_type, etc */                   \
    } ;                                                                     \
    GxB_SelectOp GXB (op) = & GB_OPAQUE (op)

GXB_SEL (TRIL     , "tril"    ) ;
GXB_SEL (TRIU     , "triu"    ) ;
GXB_SEL (DIAG     , "diag"    ) ;
GXB_SEL (OFFDIAG  , "offdiag" ) ;

GXB_SEL (NONZERO  , "nonzero" ) ;
GXB_SEL (EQ_ZERO  , "eq_zero" ) ;
GXB_SEL (GT_ZERO  , "gt_zero" ) ;
GXB_SEL (GE_ZERO  , "ge_zero" ) ;
GXB_SEL (LT_ZERO  , "lt_zero" ) ;
GXB_SEL (LE_ZERO  , "le_zero" ) ;

GXB_SEL (NE_THUNK , "ne_thunk") ;
GXB_SEL (EQ_THUNK , "eq_thunk") ;
GXB_SEL (GT_THUNK , "gt_thunk") ;
GXB_SEL (GE_THUNK , "ge_thunk") ;
GXB_SEL (LT_THUNK , "lt_thunk") ;
GXB_SEL (LE_THUNK , "le_thunk") ;

//------------------------------------------------------------------------------
// define all built-in operators
//------------------------------------------------------------------------------

#define GB_TYPE             bool
#define GB_XTYPE            BOOL
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             int8_t
#define GB_XTYPE            INT8
#define GB_SIGNED_INT
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             uint8_t
#define GB_XTYPE            UINT8
#define GB_UNSIGNED_INT
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             int16_t
#define GB_XTYPE            INT16
#define GB_SIGNED_INT
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             uint16_t
#define GB_XTYPE            UINT16
#define GB_UNSIGNED_INT
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             int32_t
#define GB_XTYPE            INT32
#define GB_SIGNED_INT
#define GB_SIGNED_INDEX
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             uint32_t
#define GB_XTYPE            UINT32
#define GB_UNSIGNED_INT
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             int64_t
#define GB_XTYPE            INT64
#define GB_SIGNED_INT
#define GB_SIGNED_INDEX
#define GB_SIGNED_INDEX64
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             uint64_t
#define GB_XTYPE            UINT64
#define GB_UNSIGNED_INT
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             float
#define GB_XTYPE            FP32
#define GB_FLOAT
#define GB_FLOATING_POINT
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             double
#define GB_XTYPE            FP64
#define GB_DOUBLE
#define GB_FLOATING_POINT
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             GxB_FC32_t
#define GB_XTYPE            FC32
#define GB_FLOAT_COMPLEX
#define GB_COMPLEX
#define GB_FLOATING_POINT
#include "builtin/factory/GB_builtin_template.c"

#define GB_TYPE             GxB_FC64_t
#define GB_XTYPE            FC64
#define GB_DOUBLE_COMPLEX
#define GB_COMPLEX
#define GB_FLOATING_POINT
#include "builtin/factory/GB_builtin_template.c"

//------------------------------------------------------------------------------
// special cases for functions and operators
//------------------------------------------------------------------------------

// 5 special cases:
// purely boolean operators: these do not have _BOOL in their name
// They are not created by the templates above.
GrB_UnaryOp  GrB_LNOT  = & GB_OPAQUE (LNOT_BOOL) ;
GrB_BinaryOp GrB_LOR   = & GB_OPAQUE (LOR_BOOL) ;
GrB_BinaryOp GrB_LAND  = & GB_OPAQUE (LAND_BOOL) ;
GrB_BinaryOp GrB_LXOR  = & GB_OPAQUE (LXOR_BOOL) ;
GrB_BinaryOp GrB_LXNOR = & GB_OPAQUE (EQ_BOOL) ;

// Special case for GrB_Matrix_build and GrB_Vector_build:  this operator is
// not valid to use in any other methods.
struct GB_BinaryOp_opaque GB_OPAQUE (IGNORE_DUP) =
{
    GB_MAGIC2, 0,               // magic and header_size
    NULL, 0,                    // no user_name for GrB_get/GrB_set
    NULL,                       // ztype
    NULL,                       // xtype
    NULL,                       // ytype
    NULL, NULL, NULL,           // no function pointer
    "ignore_dup", 0,            // name and name_len
    GB_NOP_code,                // opcode
    NULL, 0, 0,                 // defn, alloc, and hash
    NULL, NULL, NULL, 0         // theta_type, etc
} ;
GrB_BinaryOp GxB_IGNORE_DUP = & GB_OPAQUE (IGNORE_DUP) ;

// GrB_ONEB_T: identical to GxB_PAIR_T
GrB_BinaryOp GrB_ONEB_BOOL   = & GB_OPAQUE (PAIR_BOOL) ;
GrB_BinaryOp GrB_ONEB_INT8   = & GB_OPAQUE (PAIR_INT8) ;
GrB_BinaryOp GrB_ONEB_INT16  = & GB_OPAQUE (PAIR_INT16) ;
GrB_BinaryOp GrB_ONEB_INT32  = & GB_OPAQUE (PAIR_INT32) ;
GrB_BinaryOp GrB_ONEB_INT64  = & GB_OPAQUE (PAIR_INT64) ;
GrB_BinaryOp GrB_ONEB_UINT8  = & GB_OPAQUE (PAIR_UINT8) ;
GrB_BinaryOp GrB_ONEB_UINT16 = & GB_OPAQUE (PAIR_UINT16) ;
GrB_BinaryOp GrB_ONEB_UINT32 = & GB_OPAQUE (PAIR_UINT32) ;
GrB_BinaryOp GrB_ONEB_UINT64 = & GB_OPAQUE (PAIR_UINT64) ;
GrB_BinaryOp GrB_ONEB_FP32   = & GB_OPAQUE (PAIR_FP32) ;
GrB_BinaryOp GrB_ONEB_FP64   = & GB_OPAQUE (PAIR_FP64) ;
GrB_BinaryOp GxB_ONEB_FC32   = & GB_OPAQUE (PAIR_FC32) ;
GrB_BinaryOp GxB_ONEB_FC64   = & GB_OPAQUE (PAIR_FC64) ;

// nonzombie function for generic case
extern void GB_nonzombie_func (bool *z, const void *x,
    int64_t i, uint64_t j, const void *y) ;

// GxB_NONZOMBIE: internal use only
struct GB_IndexUnaryOp_opaque GB_OPAQUE (NONZOMBIE) =
{
    GB_MAGIC, 0,                // magic and header_size
    NULL, 0,                    // no user_name for GrB_get/GrB_set
    & GB_OPAQUE (BOOL),         // ztype
    NULL,                       // xtype
    & GB_OPAQUE (INT64),        // ytype
    NULL, (GxB_index_unary_function) &GB_nonzombie_func, NULL,
    "nonzombie", 0,             // name and name_len
    GB_NONZOMBIE_idxunop_code,  // opcode
    NULL, 0, 0,                 // defn, alloc, hash
    NULL, NULL, NULL, 0         // theta_type, etc
} ;
GrB_IndexUnaryOp GxB_NONZOMBIE = & GB_OPAQUE (NONZOMBIE) ;

//------------------------------------------------------------------------------
// GrB_ALL
//------------------------------------------------------------------------------

// The GrB_ALL pointer is never dereferenced.  It is passed in as an argument
// to indicate that all indices are to be used, as in the colon in C = A(:,j).

uint64_t GB_OPAQUE (ALL) = 0 ;
const uint64_t *GrB_ALL = & GB_OPAQUE (ALL) ;

// the default hyper_switch is defined in GB_defaults.h
const double GxB_HYPER_DEFAULT = GB_HYPER_SWITCH_DEFAULT ;

// set GxB_HYPER_SWITCH to either of these to ensure matrix is always, or never,
// stored in hypersparse format, respectively.
const double GxB_ALWAYS_HYPER = GB_ALWAYS_HYPER ;
const double GxB_NEVER_HYPER  = GB_NEVER_HYPER ;
const int GxB_FORMAT_DEFAULT = GxB_BY_ROW ;

//------------------------------------------------------------------------------
// predefined built-in monoids
//------------------------------------------------------------------------------

#if defined (GxB_HAVE_COMPLEX_MSVC)
#define GB_FC32_ONE  {1.0f, 0.0f}
#define GB_FC64_ONE  {1.0 , 0.0 }
#define GB_FC32_ZERO {0.0f, 0.0f}
#define GB_FC64_ZERO {0.0 , 0.0 }
#else
#define GB_FC32_ONE  GxB_CMPLXF (1,0)
#define GB_FC64_ONE  GxB_CMPLX  (1,0)
#define GB_FC32_ZERO GxB_CMPLXF (0,0)
#define GB_FC64_ZERO GxB_CMPLX  (0,0)
#endif

// helper macro to define built-in monoids (no terminal value)
#define GB_MONOID_DEF(op,ztype,identity)                                    \
    ztype GB_OPAQUE (GB_EVAL2 (identity_, op)) = identity ;                 \
    struct GB_Monoid_opaque GB_OPAQUE (GB_EVAL2 (op, _MONOID)) =            \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (op),           /* additive operator */                 \
        & GB_OPAQUE (GB_EVAL2 (identity_, op)),     /* identity */          \
        NULL,                                       /* terminal */          \
        0, 0, 0                     /* alloc id & term, hash */             \
    } ;                                                                     \
    GrB_Monoid GXB (GB_EVAL2 (op, _MONOID)) =                               \
        & GB_OPAQUE (GB_EVAL2 (op, _MONOID)) ;

// helper macro to define built-in monoids (with terminal value)
#define GB_MONOID_DEFT(op,ztype,identity,terminal)                          \
    ztype GB_OPAQUE (GB_EVAL2 (identity_, op)) = identity ;                 \
    ztype GB_OPAQUE (GB_EVAL2 (terminal_, op)) = terminal ;                 \
    struct GB_Monoid_opaque GB_OPAQUE (GB_EVAL2 (op, _MONOID)) =            \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (op),           /* additive operator */                 \
        & GB_OPAQUE (GB_EVAL2 (identity_, op)),     /* identity */          \
        & GB_OPAQUE (GB_EVAL2 (terminal_, op)),     /* terminal */          \
        0, 0, 0                     /* alloc id & term, hash */             \
    } ;                                                                     \
    GrB_Monoid GXB (GB_EVAL2 (op, _MONOID)) =                               \
        & GB_OPAQUE (GB_EVAL2 (op, _MONOID)) ;

// macro to construct GrB_* monoids in the updated specification
#define GB_MONOID_GRB(op,type)                                              \
GrB_Monoid GRB (GB_EVAL3 (op, _MONOID_, type)) =                            \
    & GB_OPAQUE (GB_EVAL4 (op, _, type, _MONOID)) ;

// MIN monoids:
GB_MONOID_DEFT ( MIN_INT8     , int8_t    , INT8_MAX    , INT8_MIN  )
GB_MONOID_DEFT ( MIN_INT16    , int16_t   , INT16_MAX   , INT16_MIN )
GB_MONOID_DEFT ( MIN_INT32    , int32_t   , INT32_MAX   , INT32_MIN )
GB_MONOID_DEFT ( MIN_INT64    , int64_t   , INT64_MAX   , INT64_MIN )
GB_MONOID_DEFT ( MIN_UINT8    , uint8_t   , UINT8_MAX   , 0         )
GB_MONOID_DEFT ( MIN_UINT16   , uint16_t  , UINT16_MAX  , 0         )
GB_MONOID_DEFT ( MIN_UINT32   , uint32_t  , UINT32_MAX  , 0         )
GB_MONOID_DEFT ( MIN_UINT64   , uint64_t  , UINT64_MAX  , 0         )
GB_MONOID_DEF  ( MIN_FP32     , float     , INFINITY    )
GB_MONOID_DEF  ( MIN_FP64     , double    , INFINITY    )

GB_MONOID_GRB  ( MIN, INT8     )
GB_MONOID_GRB  ( MIN, INT16    )
GB_MONOID_GRB  ( MIN, INT32    )
GB_MONOID_GRB  ( MIN, INT64    )
GB_MONOID_GRB  ( MIN, UINT8    )
GB_MONOID_GRB  ( MIN, UINT16   )
GB_MONOID_GRB  ( MIN, UINT32   )
GB_MONOID_GRB  ( MIN, UINT64   )
GB_MONOID_GRB  ( MIN, FP32     )
GB_MONOID_GRB  ( MIN, FP64     )

// MAX monoids:
GB_MONOID_DEFT ( MAX_INT8     , int8_t    , INT8_MIN    , INT8_MAX  )
GB_MONOID_DEFT ( MAX_INT16    , int16_t   , INT16_MIN   , INT16_MAX )
GB_MONOID_DEFT ( MAX_INT32    , int32_t   , INT32_MIN   , INT32_MAX )
GB_MONOID_DEFT ( MAX_INT64    , int64_t   , INT64_MIN   , INT64_MAX )
GB_MONOID_DEFT ( MAX_UINT8    , uint8_t   , 0           , UINT8_MAX )
GB_MONOID_DEFT ( MAX_UINT16   , uint16_t  , 0           , UINT16_MAX)
GB_MONOID_DEFT ( MAX_UINT32   , uint32_t  , 0           , UINT32_MAX)
GB_MONOID_DEFT ( MAX_UINT64   , uint64_t  , 0           , UINT64_MAX)
GB_MONOID_DEF  ( MAX_FP32     , float     , -INFINITY   )
GB_MONOID_DEF  ( MAX_FP64     , double    , -INFINITY   )

GB_MONOID_GRB  ( MAX, INT8     )
GB_MONOID_GRB  ( MAX, INT16    )
GB_MONOID_GRB  ( MAX, INT32    )
GB_MONOID_GRB  ( MAX, INT64    )
GB_MONOID_GRB  ( MAX, UINT8    )
GB_MONOID_GRB  ( MAX, UINT16   )
GB_MONOID_GRB  ( MAX, UINT32   )
GB_MONOID_GRB  ( MAX, UINT64   )
GB_MONOID_GRB  ( MAX, FP32     )
GB_MONOID_GRB  ( MAX, FP64     )

// PLUS monoids:
GB_MONOID_DEF  ( PLUS_INT8    , int8_t    , 0           )
GB_MONOID_DEF  ( PLUS_INT16   , int16_t   , 0           )
GB_MONOID_DEF  ( PLUS_INT32   , int32_t   , 0           )
GB_MONOID_DEF  ( PLUS_INT64   , int64_t   , 0           )
GB_MONOID_DEF  ( PLUS_UINT8   , uint8_t   , 0           )
GB_MONOID_DEF  ( PLUS_UINT16  , uint16_t  , 0           )
GB_MONOID_DEF  ( PLUS_UINT32  , uint32_t  , 0           )
GB_MONOID_DEF  ( PLUS_UINT64  , uint64_t  , 0           )
GB_MONOID_DEF  ( PLUS_FP32    , float     , 0           )
GB_MONOID_DEF  ( PLUS_FP64    , double    , 0           )
GB_MONOID_DEF  ( PLUS_FC32    , GxB_FC32_t, GB_FC32_ZERO)
GB_MONOID_DEF  ( PLUS_FC64    , GxB_FC64_t, GB_FC64_ZERO)

GB_MONOID_GRB  ( PLUS, INT8     )
GB_MONOID_GRB  ( PLUS, INT16    )
GB_MONOID_GRB  ( PLUS, INT32    )
GB_MONOID_GRB  ( PLUS, INT64    )
GB_MONOID_GRB  ( PLUS, UINT8    )
GB_MONOID_GRB  ( PLUS, UINT16   )
GB_MONOID_GRB  ( PLUS, UINT32   )
GB_MONOID_GRB  ( PLUS, UINT64   )
GB_MONOID_GRB  ( PLUS, FP32     )
GB_MONOID_GRB  ( PLUS, FP64     )

// TIMES monoids:
GB_MONOID_DEFT ( TIMES_INT8   , int8_t    , 1           , 0)
GB_MONOID_DEFT ( TIMES_INT16  , int16_t   , 1           , 0)
GB_MONOID_DEFT ( TIMES_INT32  , int32_t   , 1           , 0)
GB_MONOID_DEFT ( TIMES_INT64  , int64_t   , 1           , 0)
GB_MONOID_DEFT ( TIMES_UINT8  , uint8_t   , 1           , 0)
GB_MONOID_DEFT ( TIMES_UINT16 , uint16_t  , 1           , 0)
GB_MONOID_DEFT ( TIMES_UINT32 , uint32_t  , 1           , 0)
GB_MONOID_DEFT ( TIMES_UINT64 , uint64_t  , 1           , 0)
GB_MONOID_DEF  ( TIMES_FP32   , float     , 1           )
GB_MONOID_DEF  ( TIMES_FP64   , double    , 1           )
GB_MONOID_DEF  ( TIMES_FC32   , GxB_FC32_t, GB_FC32_ONE )
GB_MONOID_DEF  ( TIMES_FC64   , GxB_FC64_t, GB_FC64_ONE )

GB_MONOID_GRB  ( TIMES, INT8     )
GB_MONOID_GRB  ( TIMES, INT16    )
GB_MONOID_GRB  ( TIMES, INT32    )
GB_MONOID_GRB  ( TIMES, INT64    )
GB_MONOID_GRB  ( TIMES, UINT8    )
GB_MONOID_GRB  ( TIMES, UINT16   )
GB_MONOID_GRB  ( TIMES, UINT32   )
GB_MONOID_GRB  ( TIMES, UINT64   )
GB_MONOID_GRB  ( TIMES, FP32     )
GB_MONOID_GRB  ( TIMES, FP64     )

// ANY monoids:
GB_MONOID_DEFT ( ANY_INT8     , int8_t    , 0           , 0)
GB_MONOID_DEFT ( ANY_INT16    , int16_t   , 0           , 0)
GB_MONOID_DEFT ( ANY_INT32    , int32_t   , 0           , 0)
GB_MONOID_DEFT ( ANY_INT64    , int64_t   , 0           , 0)
GB_MONOID_DEFT ( ANY_UINT8    , uint8_t   , 0           , 0)
GB_MONOID_DEFT ( ANY_UINT16   , uint16_t  , 0           , 0)
GB_MONOID_DEFT ( ANY_UINT32   , uint32_t  , 0           , 0)
GB_MONOID_DEFT ( ANY_UINT64   , uint64_t  , 0           , 0)
GB_MONOID_DEFT ( ANY_FP32     , float     , 0           , 0)
GB_MONOID_DEFT ( ANY_FP64     , double    , 0           , 0)
GB_MONOID_DEFT ( ANY_FC32     , GxB_FC32_t, GB_FC32_ZERO, GB_FC32_ZERO)
GB_MONOID_DEFT ( ANY_FC64     , GxB_FC64_t, GB_FC64_ZERO, GB_FC64_ZERO)

// Boolean monoids:
GB_MONOID_DEFT ( ANY_BOOL     , bool      , false       , false)
GB_MONOID_DEFT ( LOR_BOOL     , bool      , false       , true )
GB_MONOID_DEFT ( LAND_BOOL    , bool      , true        , false)
GB_MONOID_DEF  ( LXOR_BOOL    , bool      , false       )
GB_MONOID_DEF  ( EQ_BOOL      , bool      , true        )
// GrB_LXNOR_BOOL_MONIOD is the same as GrB_EQ_BOOL_MONIOD:
GrB_Monoid GxB_LXNOR_BOOL_MONOID = & GB_OPAQUE (EQ_BOOL_MONOID) ;

GB_MONOID_GRB  ( LOR  , BOOL     )
GB_MONOID_GRB  ( LAND , BOOL     )
GB_MONOID_GRB  ( LXOR , BOOL     )
GrB_Monoid GrB_LXNOR_MONOID_BOOL = & GB_OPAQUE (EQ_BOOL_MONOID) ;

// BOR monoids (bitwise or):
GB_MONOID_DEFT ( BOR_UINT8    , uint8_t   , 0, 0xFF               )
GB_MONOID_DEFT ( BOR_UINT16   , uint16_t  , 0, 0xFFFF             )
GB_MONOID_DEFT ( BOR_UINT32   , uint32_t  , 0, 0xFFFFFFFF         )
GB_MONOID_DEFT ( BOR_UINT64   , uint64_t  , 0, 0xFFFFFFFFFFFFFFFF )

// BAND monoids (bitwise and):
GB_MONOID_DEFT ( BAND_UINT8   , uint8_t   , 0xFF              , 0 )
GB_MONOID_DEFT ( BAND_UINT16  , uint16_t  , 0xFFFF            , 0 )
GB_MONOID_DEFT ( BAND_UINT32  , uint32_t  , 0xFFFFFFFF        , 0 )
GB_MONOID_DEFT ( BAND_UINT64  , uint64_t  , 0xFFFFFFFFFFFFFFFF, 0 )

// BXOR monoids (bitwise xor):
GB_MONOID_DEF  ( BXOR_UINT8   , uint8_t   , 0)
GB_MONOID_DEF  ( BXOR_UINT16  , uint16_t  , 0)
GB_MONOID_DEF  ( BXOR_UINT32  , uint32_t  , 0)
GB_MONOID_DEF  ( BXOR_UINT64  , uint64_t  , 0)

// BXNOR monoids (bitwise xnor):
GB_MONOID_DEF  ( BXNOR_UINT8  , uint8_t   , 0xFF               )
GB_MONOID_DEF  ( BXNOR_UINT16 , uint16_t  , 0xFFFF             )
GB_MONOID_DEF  ( BXNOR_UINT32 , uint32_t  , 0xFFFFFFFF         )
GB_MONOID_DEF  ( BXNOR_UINT64 , uint64_t  , 0xFFFFFFFFFFFFFFFF )

//------------------------------------------------------------------------------
// predefined built-in semirings
//------------------------------------------------------------------------------

#define GB_SEMIRING_NAME(add,mult) \
    GB_EVAL5 (add, _, mult, _, GB_XTYPE)

// helper macro to define semirings: all x,y,z types the same
#define GXB_SEMIRING(add,mult)                                              \
    struct GB_Semiring_opaque GB_OPAQUE (GB_SEMIRING_NAME(add, mult)) =     \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (GB_EVAL4 (add, _, GB_XTYPE, _MONOID)),                 \
        & GB_OPAQUE (GB_EVAL3 (mult, _, GB_XTYPE)),                         \
        NULL, 0, 0, 0               /* name, name_len, name_size, hash */   \
    } ;                                                                     \
    GrB_Semiring GXB (GB_SEMIRING_NAME (add, mult)) =                       \
         & GB_OPAQUE (GB_SEMIRING_NAME (add, mult)) ;

// helper macro to define semirings: x,y types the same, z boolean
#define GB_SEMIRING_COMPARE_DEFINE(add,mult)                                \
    struct GB_Semiring_opaque GB_OPAQUE (GB_SEMIRING_NAME(add, mult)) =     \
    {                                                                       \
        GB_MAGIC, 0,                /* magic and header_size */             \
        NULL, 0,                    /* no user_name for GrB_get/GrB_set */  \
        & GB_OPAQUE (GB_EVAL2 (add, _BOOL_MONOID)),                         \
        & GB_OPAQUE (GB_EVAL3 (mult, _, GB_XTYPE)),                         \
        NULL, 0, 0, 0               /* name, name_len, name_size, hash */   \
    } ;

#define GXB_SEMIRING_COMPARE(add,mult)                                      \
    GB_SEMIRING_COMPARE_DEFINE (add, mult)                                  \
    GrB_Semiring GXB (GB_SEMIRING_NAME (add, mult)) =                       \
         & GB_OPAQUE (GB_SEMIRING_NAME (add, mult)) ;

#define GB_XTYPE    BOOL
#define GB_BOOLEAN
#include "builtin/factory/GB_semiring_template.c"

#define GB_XTYPE    INT8
#include "builtin/factory/GB_semiring_template.c"

#define GB_XTYPE    UINT8
#define GB_UNSIGNED_INT
#include "builtin/factory/GB_semiring_template.c"

#define GB_XTYPE    INT16
#include "builtin/factory/GB_semiring_template.c"

#define GB_XTYPE    UINT16
#define GB_UNSIGNED_INT
#include "builtin/factory/GB_semiring_template.c"

#define GB_XTYPE    INT32
#define GB_POSITIONAL
#include "builtin/factory/GB_semiring_template.c"

#define GB_UNSIGNED_INT
#define GB_XTYPE    UINT32
#include "builtin/factory/GB_semiring_template.c"

#define GB_POSITIONAL
#define GB_XTYPE    INT64
#include "builtin/factory/GB_semiring_template.c"

#define GB_UNSIGNED_INT
#define GB_XTYPE    UINT64
#include "builtin/factory/GB_semiring_template.c"

#define GB_XTYPE    FP32
#include "builtin/factory/GB_semiring_template.c"

#define GB_XTYPE    FP64
#include "builtin/factory/GB_semiring_template.c"

#define GB_COMPLEX
#define GB_XTYPE    FC32
#include "builtin/factory/GB_semiring_template.c"

#define GB_COMPLEX
#define GB_XTYPE    FC64
#include "builtin/factory/GB_semiring_template.c"

//------------------------------------------------------------------------------
// 124 predefined built-in semirings in the v1.3 C API
//------------------------------------------------------------------------------

// These predefined semirings have been added to the spec, as of v1.3.
// They are identical to the GxB* semirings of the same name, except for
// GrB_LXNOR_LOR_SEMIRING_BOOL, which is identical to GxB_EQ_LOR_BOOL.

#define GRB_SEMIRING(add,mult,xtype)                                \
GrB_Semiring GRB (GB_EVAL5 (add, _, mult, _SEMIRING_, xtype)) =     \
    & GB_OPAQUE (GB_EVAL5 (add, _, mult, _, xtype)) ;

    //--------------------------------------------------------------------------
    // 4 boolean semirings
    //--------------------------------------------------------------------------

    GRB_SEMIRING (LOR, LAND, BOOL)
    GRB_SEMIRING (LAND, LOR, BOOL)
    GRB_SEMIRING (LXOR, LAND, BOOL)
    GrB_Semiring GRB (LXNOR_LOR_SEMIRING_BOOL) = & GB_OPAQUE (EQ_LOR_BOOL) ;

    //--------------------------------------------------------------------------
    // 20 semirings with PLUS monoids
    //--------------------------------------------------------------------------

    GRB_SEMIRING (PLUS, TIMES, INT8)
    GRB_SEMIRING (PLUS, TIMES, INT16)
    GRB_SEMIRING (PLUS, TIMES, INT32)
    GRB_SEMIRING (PLUS, TIMES, INT64)
    GRB_SEMIRING (PLUS, TIMES, UINT8)
    GRB_SEMIRING (PLUS, TIMES, UINT16)
    GRB_SEMIRING (PLUS, TIMES, UINT32)
    GRB_SEMIRING (PLUS, TIMES, UINT64)
    GRB_SEMIRING (PLUS, TIMES, FP32)
    GRB_SEMIRING (PLUS, TIMES, FP64)

    GRB_SEMIRING (PLUS, MIN, INT8)
    GRB_SEMIRING (PLUS, MIN, INT16)
    GRB_SEMIRING (PLUS, MIN, INT32)
    GRB_SEMIRING (PLUS, MIN, INT64)
    GRB_SEMIRING (PLUS, MIN, UINT8)
    GRB_SEMIRING (PLUS, MIN, UINT16)
    GRB_SEMIRING (PLUS, MIN, UINT32)
    GRB_SEMIRING (PLUS, MIN, UINT64)
    GRB_SEMIRING (PLUS, MIN, FP32)
    GRB_SEMIRING (PLUS, MIN, FP64)

    //--------------------------------------------------------------------------
    // 50 semirings with MIN monoids
    //--------------------------------------------------------------------------

    GRB_SEMIRING (MIN, PLUS, INT8)
    GRB_SEMIRING (MIN, PLUS, INT16)
    GRB_SEMIRING (MIN, PLUS, INT32)
    GRB_SEMIRING (MIN, PLUS, INT64)
    GRB_SEMIRING (MIN, PLUS, UINT8)
    GRB_SEMIRING (MIN, PLUS, UINT16)
    GRB_SEMIRING (MIN, PLUS, UINT32)
    GRB_SEMIRING (MIN, PLUS, UINT64)
    GRB_SEMIRING (MIN, PLUS, FP32)
    GRB_SEMIRING (MIN, PLUS, FP64)

    GRB_SEMIRING (MIN, TIMES, INT8)
    GRB_SEMIRING (MIN, TIMES, INT16)
    GRB_SEMIRING (MIN, TIMES, INT32)
    GRB_SEMIRING (MIN, TIMES, INT64)
    GRB_SEMIRING (MIN, TIMES, UINT8)
    GRB_SEMIRING (MIN, TIMES, UINT16)
    GRB_SEMIRING (MIN, TIMES, UINT32)
    GRB_SEMIRING (MIN, TIMES, UINT64)
    GRB_SEMIRING (MIN, TIMES, FP32)
    GRB_SEMIRING (MIN, TIMES, FP64)

    GRB_SEMIRING (MIN, FIRST, INT8)
    GRB_SEMIRING (MIN, FIRST, INT16)
    GRB_SEMIRING (MIN, FIRST, INT32)
    GRB_SEMIRING (MIN, FIRST, INT64)
    GRB_SEMIRING (MIN, FIRST, UINT8)
    GRB_SEMIRING (MIN, FIRST, UINT16)
    GRB_SEMIRING (MIN, FIRST, UINT32)
    GRB_SEMIRING (MIN, FIRST, UINT64)
    GRB_SEMIRING (MIN, FIRST, FP32)
    GRB_SEMIRING (MIN, FIRST, FP64)

    GRB_SEMIRING (MIN, SECOND, INT8)
    GRB_SEMIRING (MIN, SECOND, INT16)
    GRB_SEMIRING (MIN, SECOND, INT32)
    GRB_SEMIRING (MIN, SECOND, INT64)
    GRB_SEMIRING (MIN, SECOND, UINT8)
    GRB_SEMIRING (MIN, SECOND, UINT16)
    GRB_SEMIRING (MIN, SECOND, UINT32)
    GRB_SEMIRING (MIN, SECOND, UINT64)
    GRB_SEMIRING (MIN, SECOND, FP32)
    GRB_SEMIRING (MIN, SECOND, FP64)

    GRB_SEMIRING (MIN, MAX, INT8)
    GRB_SEMIRING (MIN, MAX, INT16)
    GRB_SEMIRING (MIN, MAX, INT32)
    GRB_SEMIRING (MIN, MAX, INT64)
    GRB_SEMIRING (MIN, MAX, UINT8)
    GRB_SEMIRING (MIN, MAX, UINT16)
    GRB_SEMIRING (MIN, MAX, UINT32)
    GRB_SEMIRING (MIN, MAX, UINT64)
    GRB_SEMIRING (MIN, MAX, FP32)
    GRB_SEMIRING (MIN, MAX, FP64)

    //--------------------------------------------------------------------------
    // 50 semirings with MAX monoids
    //--------------------------------------------------------------------------

    GRB_SEMIRING (MAX, PLUS, INT8)
    GRB_SEMIRING (MAX, PLUS, INT16)
    GRB_SEMIRING (MAX, PLUS, INT32)
    GRB_SEMIRING (MAX, PLUS, INT64)
    GRB_SEMIRING (MAX, PLUS, UINT8)
    GRB_SEMIRING (MAX, PLUS, UINT16)
    GRB_SEMIRING (MAX, PLUS, UINT32)
    GRB_SEMIRING (MAX, PLUS, UINT64)
    GRB_SEMIRING (MAX, PLUS, FP32)
    GRB_SEMIRING (MAX, PLUS, FP64)

    GRB_SEMIRING (MAX, TIMES, INT8)
    GRB_SEMIRING (MAX, TIMES, INT16)
    GRB_SEMIRING (MAX, TIMES, INT32)
    GRB_SEMIRING (MAX, TIMES, INT64)
    GRB_SEMIRING (MAX, TIMES, UINT8)
    GRB_SEMIRING (MAX, TIMES, UINT16)
    GRB_SEMIRING (MAX, TIMES, UINT32)
    GRB_SEMIRING (MAX, TIMES, UINT64)
    GRB_SEMIRING (MAX, TIMES, FP32)
    GRB_SEMIRING (MAX, TIMES, FP64)

    GRB_SEMIRING (MAX, FIRST, INT8)
    GRB_SEMIRING (MAX, FIRST, INT16)
    GRB_SEMIRING (MAX, FIRST, INT32)
    GRB_SEMIRING (MAX, FIRST, INT64)
    GRB_SEMIRING (MAX, FIRST, UINT8)
    GRB_SEMIRING (MAX, FIRST, UINT16)
    GRB_SEMIRING (MAX, FIRST, UINT32)
    GRB_SEMIRING (MAX, FIRST, UINT64)
    GRB_SEMIRING (MAX, FIRST, FP32)
    GRB_SEMIRING (MAX, FIRST, FP64)

    GRB_SEMIRING (MAX, SECOND, INT8)
    GRB_SEMIRING (MAX, SECOND, INT16)
    GRB_SEMIRING (MAX, SECOND, INT32)
    GRB_SEMIRING (MAX, SECOND, INT64)
    GRB_SEMIRING (MAX, SECOND, UINT8)
    GRB_SEMIRING (MAX, SECOND, UINT16)
    GRB_SEMIRING (MAX, SECOND, UINT32)
    GRB_SEMIRING (MAX, SECOND, UINT64)
    GRB_SEMIRING (MAX, SECOND, FP32)
    GRB_SEMIRING (MAX, SECOND, FP64)

    GRB_SEMIRING (MAX, MIN, INT8)
    GRB_SEMIRING (MAX, MIN, INT16)
    GRB_SEMIRING (MAX, MIN, INT32)
    GRB_SEMIRING (MAX, MIN, INT64)
    GRB_SEMIRING (MAX, MIN, UINT8)
    GRB_SEMIRING (MAX, MIN, UINT16)
    GRB_SEMIRING (MAX, MIN, UINT32)
    GRB_SEMIRING (MAX, MIN, UINT64)
    GRB_SEMIRING (MAX, MIN, FP32)
    GRB_SEMIRING (MAX, MIN, FP64)

//------------------------------------------------------------------------------
// GxB_CONTEXT_WORLD
//------------------------------------------------------------------------------

struct GB_Context_opaque GB_OPAQUE (CONTEXT_WORLD) =
{
    GB_MAGIC,                       // magic: initialized
    0,                              // header_size: statically allocated
    NULL, 0,                        // no user_name for GrB_get/GrB_set
    // revised by GxB_Context_get/set:
    (double) GB_CHUNK_DEFAULT,      // chunk
    1,                              // nthreads_max
    0,                              // int32_t ngpus
    // uint8_t gpu_ids [GB_MAX_NGPUS]:
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
} ;

GxB_Context GxB_CONTEXT_WORLD = & GB_OPAQUE (CONTEXT_WORLD) ;

