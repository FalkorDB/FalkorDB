//------------------------------------------------------------------------------
// GB_mex_test28: test GrB_get and GrB_set (binary ops)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test28"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

#define GETOP(op,opname)                                                \
{                                                                       \
    size_t siz1, siz2, siz3 ;                                           \
    OK (GrB_BinaryOp_get_String (op, name, GrB_NAME)) ;                 \
    CHECK (MATCH (name, opname)) ;                                      \
    OK (GrB_BinaryOp_get_String (op, cname, GxB_JIT_C_NAME)) ;      \
    printf ("%s: %s\n", name, cname) ; \
    OK (GrB_BinaryOp_get_SIZE (op, &size, GrB_NAME)) ;                  \
    CHECK (size == strlen (name) + 1) ;                                 \
    GrB_Info info2, info3 ;                                             \
    info2 = GrB_BinaryOp_get_SIZE (op, &siz1, GrB_INP0_TYPE_STRING) ;  \
    info3 = GrB_BinaryOp_get_String (op, name, GrB_INP0_TYPE_STRING) ; \
    CHECK (info2 == info3) ;                                            \
    CHECK (siz1 == strlen (name) + 1) ;                                 \
    if (info2 == GrB_NO_VALUE) { CHECK (siz1 == 1) ; }                  \
    info2 = GrB_BinaryOp_get_SIZE (op, &siz2, GrB_INP1_TYPE_STRING) ;  \
    info3 = GrB_BinaryOp_get_String (op, name, GrB_INP1_TYPE_STRING) ; \
    CHECK (info2 == info3) ;                                            \
    CHECK (siz2 == strlen (name) + 1) ;                                 \
    if (info2 == GrB_NO_VALUE) { CHECK (siz2 == 1) ; }                  \
    info2 = GrB_BinaryOp_get_SIZE (op, &siz3, GrB_OUTP_TYPE_STRING) ;  \
    info3 = GrB_BinaryOp_get_String (op, name, GrB_OUTP_TYPE_STRING) ; \
    CHECK (info2 == info3) ;                                            \
    CHECK (siz3 == strlen (name) + 1) ;                                 \
    if (info2 == GrB_NO_VALUE) { CHECK (siz3 == 1) ; }                  \
}

#define GETNAME(op)                                         \
{                                                           \
    GETOP (op, #op) ;                                       \
/*  OK (GxB_BinaryOp_fprint (op, "binop", 3, NULL)) ;   */  \
}

#define GETNAM2(op,alias)                                   \
{                                                           \
    GETOP (op,alias) ;                                      \
/*  OK (GxB_BinaryOp_fprint (op, "binop", 3, NULL)) ; */    \
}

void myfunc (float *z, const float *x, const float *y) ;
void myfunc (float *z, const float *x, const float *y) { (*z) = (*x)+(*y) ; }
#define MYFUNC_DEFN \
"void myfunc (float *z, const float *x, const float *y) { (*z) = (*x)+(*y) ; }"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info, expected ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Scalar s = NULL, s_fp64 = NULL, s_int32 = NULL, s_fp32 = NULL ;
    GrB_BinaryOp binop = NULL ;
    uint8_t stuff [256] ;
    void *nothing = stuff ;
    size_t size ;
    char name [256] ;
    char cname [256] ;
    char defn [2048] ;
    int32_t code, i ;
    float fvalue ;
    double dvalue ;

    OK (GrB_Scalar_new (&s_fp64, GrB_FP64)) ;
    OK (GrB_Scalar_new (&s_fp32, GrB_FP32)) ;
    OK (GrB_Scalar_new (&s_int32, GrB_INT32)) ;

    //--------------------------------------------------------------------------
    // GrB_BinaryOp get name
    //--------------------------------------------------------------------------

    GETNAME (GrB_FIRST_BOOL) ;
    GETNAME (GrB_FIRST_INT8) ;
    GETNAME (GrB_FIRST_INT16) ;
    GETNAME (GrB_FIRST_INT32) ;
    GETNAME (GrB_FIRST_INT64) ;
    GETNAME (GrB_FIRST_UINT8) ;
    GETNAME (GrB_FIRST_UINT16) ;
    GETNAME (GrB_FIRST_UINT32) ;
    GETNAME (GrB_FIRST_UINT64) ;
    GETNAME (GrB_FIRST_FP32) ;
    GETNAME (GrB_FIRST_FP64) ;
    GETNAME (GxB_FIRST_FC32) ;
    GETNAME (GxB_FIRST_FC64) ;

    GETNAME (GrB_SECOND_BOOL) ;
    GETNAME (GrB_SECOND_INT8) ;
    GETNAME (GrB_SECOND_INT16) ;
    GETNAME (GrB_SECOND_INT32) ;
    GETNAME (GrB_SECOND_INT64) ;
    GETNAME (GrB_SECOND_UINT8) ;
    GETNAME (GrB_SECOND_UINT16) ;
    GETNAME (GrB_SECOND_UINT32) ;
    GETNAME (GrB_SECOND_UINT64) ;
    GETNAME (GrB_SECOND_FP32) ;
    GETNAME (GrB_SECOND_FP64) ;
    GETNAME (GxB_SECOND_FC32) ;
    GETNAME (GxB_SECOND_FC64) ;

    GETNAME (GrB_ONEB_BOOL) ;
    GETNAME (GrB_ONEB_INT8) ;
    GETNAME (GrB_ONEB_INT16) ;
    GETNAME (GrB_ONEB_INT32) ;
    GETNAME (GrB_ONEB_INT64) ;
    GETNAME (GrB_ONEB_UINT8) ;
    GETNAME (GrB_ONEB_UINT16) ;
    GETNAME (GrB_ONEB_UINT32) ;
    GETNAME (GrB_ONEB_UINT64) ;
    GETNAME (GrB_ONEB_FP32) ;
    GETNAME (GrB_ONEB_FP64) ;
    GETNAME (GxB_ONEB_FC32) ;
    GETNAME (GxB_ONEB_FC64) ;

    GETNAME (GxB_POW_BOOL) ;
    GETNAME (GxB_POW_INT8) ;
    GETNAME (GxB_POW_INT16) ;
    GETNAME (GxB_POW_INT32) ;
    GETNAME (GxB_POW_INT64) ;
    GETNAME (GxB_POW_UINT8) ;
    GETNAME (GxB_POW_UINT16) ;
    GETNAME (GxB_POW_UINT32) ;
    GETNAME (GxB_POW_UINT64) ;
    GETNAME (GxB_POW_FP32) ;
    GETNAME (GxB_POW_FP64) ;
    GETNAME (GxB_POW_FC32) ;
    GETNAME (GxB_POW_FC64) ;

    GETNAME (GrB_PLUS_BOOL) ;
    GETNAME (GrB_PLUS_INT8) ;
    GETNAME (GrB_PLUS_INT16) ;
    GETNAME (GrB_PLUS_INT32) ;
    GETNAME (GrB_PLUS_INT64) ;
    GETNAME (GrB_PLUS_UINT8) ;
    GETNAME (GrB_PLUS_UINT16) ;
    GETNAME (GrB_PLUS_UINT32) ;
    GETNAME (GrB_PLUS_UINT64) ;
    GETNAME (GrB_PLUS_FP32) ;
    GETNAME (GrB_PLUS_FP64) ;
    GETNAME (GxB_PLUS_FC32) ;
    GETNAME (GxB_PLUS_FC64) ;

    GETNAME (GrB_MINUS_BOOL) ;
    GETNAME (GrB_MINUS_INT8) ;
    GETNAME (GrB_MINUS_INT16) ;
    GETNAME (GrB_MINUS_INT32) ;
    GETNAME (GrB_MINUS_INT64) ;
    GETNAME (GrB_MINUS_UINT8) ;
    GETNAME (GrB_MINUS_UINT16) ;
    GETNAME (GrB_MINUS_UINT32) ;
    GETNAME (GrB_MINUS_UINT64) ;
    GETNAME (GrB_MINUS_FP32) ;
    GETNAME (GrB_MINUS_FP64) ;
    GETNAME (GxB_MINUS_FC32) ;
    GETNAME (GxB_MINUS_FC64) ;

    GETNAME (GrB_TIMES_BOOL) ;
    GETNAME (GrB_TIMES_INT8) ;
    GETNAME (GrB_TIMES_INT16) ;
    GETNAME (GrB_TIMES_INT32) ;
    GETNAME (GrB_TIMES_INT64) ;
    GETNAME (GrB_TIMES_UINT8) ;
    GETNAME (GrB_TIMES_UINT16) ;
    GETNAME (GrB_TIMES_UINT32) ;
    GETNAME (GrB_TIMES_UINT64) ;
    GETNAME (GrB_TIMES_FP32) ;
    GETNAME (GrB_TIMES_FP64) ;
    GETNAME (GxB_TIMES_FC32) ;
    GETNAME (GxB_TIMES_FC64) ;

    GETNAME (GrB_DIV_BOOL) ;
    GETNAME (GrB_DIV_INT8) ;
    GETNAME (GrB_DIV_INT16) ;
    GETNAME (GrB_DIV_INT32) ;
    GETNAME (GrB_DIV_INT64) ;
    GETNAME (GrB_DIV_UINT8) ;
    GETNAME (GrB_DIV_UINT16) ;
    GETNAME (GrB_DIV_UINT32) ;
    GETNAME (GrB_DIV_UINT64) ;
    GETNAME (GrB_DIV_FP32) ;
    GETNAME (GrB_DIV_FP64) ;
    GETNAME (GxB_DIV_FC32) ;
    GETNAME (GxB_DIV_FC64) ;

    GETNAME (GxB_RMINUS_BOOL) ;
    GETNAME (GxB_RMINUS_INT8) ;
    GETNAME (GxB_RMINUS_INT16) ;
    GETNAME (GxB_RMINUS_INT32) ;
    GETNAME (GxB_RMINUS_INT64) ;
    GETNAME (GxB_RMINUS_UINT8) ;
    GETNAME (GxB_RMINUS_UINT16) ;
    GETNAME (GxB_RMINUS_UINT32) ;
    GETNAME (GxB_RMINUS_UINT64) ;
    GETNAME (GxB_RMINUS_FP32) ;
    GETNAME (GxB_RMINUS_FP64) ;
    GETNAME (GxB_RMINUS_FC32) ;
    GETNAME (GxB_RMINUS_FC64) ;

    GETNAME (GxB_RDIV_BOOL) ;
    GETNAME (GxB_RDIV_INT8) ;
    GETNAME (GxB_RDIV_INT16) ;
    GETNAME (GxB_RDIV_INT32) ;
    GETNAME (GxB_RDIV_INT64) ;
    GETNAME (GxB_RDIV_UINT8) ;
    GETNAME (GxB_RDIV_UINT16) ;
    GETNAME (GxB_RDIV_UINT32) ;
    GETNAME (GxB_RDIV_UINT64) ;
    GETNAME (GxB_RDIV_FP32) ;
    GETNAME (GxB_RDIV_FP64) ;
    GETNAME (GxB_RDIV_FC32) ;
    GETNAME (GxB_RDIV_FC64) ;

    GETNAM2 (GxB_PAIR_BOOL,     "GrB_ONEB_BOOL") ;
    GETNAM2 (GxB_PAIR_INT8,     "GrB_ONEB_INT8") ;
    GETNAM2 (GxB_PAIR_INT16,    "GrB_ONEB_INT16") ;
    GETNAM2 (GxB_PAIR_INT32,    "GrB_ONEB_INT32") ;
    GETNAM2 (GxB_PAIR_INT64,    "GrB_ONEB_INT64") ;
    GETNAM2 (GxB_PAIR_UINT8,    "GrB_ONEB_UINT8") ;
    GETNAM2 (GxB_PAIR_UINT16,   "GrB_ONEB_UINT16") ;
    GETNAM2 (GxB_PAIR_UINT32,   "GrB_ONEB_UINT32") ;
    GETNAM2 (GxB_PAIR_UINT64,   "GrB_ONEB_UINT64") ;
    GETNAM2 (GxB_PAIR_FP32,     "GrB_ONEB_FP32") ;
    GETNAM2 (GxB_PAIR_FP64,     "GrB_ONEB_FP64") ;
    GETNAM2 (GxB_PAIR_FC32,     "GxB_ONEB_FC32") ;
    GETNAM2 (GxB_PAIR_FC64,     "GxB_ONEB_FC64") ;

    GETNAME (GxB_ANY_BOOL) ;
    GETNAME (GxB_ANY_INT8) ;
    GETNAME (GxB_ANY_INT16) ;
    GETNAME (GxB_ANY_INT32) ;
    GETNAME (GxB_ANY_INT64) ;
    GETNAME (GxB_ANY_UINT8) ;
    GETNAME (GxB_ANY_UINT16) ;
    GETNAME (GxB_ANY_UINT32) ;
    GETNAME (GxB_ANY_UINT64) ;
    GETNAME (GxB_ANY_FP32) ;
    GETNAME (GxB_ANY_FP64) ;
    GETNAME (GxB_ANY_FC32) ;
    GETNAME (GxB_ANY_FC64) ;

    GETNAME (GxB_ISEQ_BOOL) ;
    GETNAME (GxB_ISEQ_INT8) ;
    GETNAME (GxB_ISEQ_INT16) ;
    GETNAME (GxB_ISEQ_INT32) ;
    GETNAME (GxB_ISEQ_INT64) ;
    GETNAME (GxB_ISEQ_UINT8) ;
    GETNAME (GxB_ISEQ_UINT16) ;
    GETNAME (GxB_ISEQ_UINT32) ;
    GETNAME (GxB_ISEQ_UINT64) ;
    GETNAME (GxB_ISEQ_FP32) ;
    GETNAME (GxB_ISEQ_FP64) ;
    GETNAME (GxB_ISEQ_FC32) ;
    GETNAME (GxB_ISEQ_FC64) ;

    GETNAME (GxB_ISNE_BOOL) ;
    GETNAME (GxB_ISNE_INT8) ;
    GETNAME (GxB_ISNE_INT16) ;
    GETNAME (GxB_ISNE_INT32) ;
    GETNAME (GxB_ISNE_INT64) ;
    GETNAME (GxB_ISNE_UINT8) ;
    GETNAME (GxB_ISNE_UINT16) ;
    GETNAME (GxB_ISNE_UINT32) ;
    GETNAME (GxB_ISNE_UINT64) ;
    GETNAME (GxB_ISNE_FP32) ;
    GETNAME (GxB_ISNE_FP64) ;
    GETNAME (GxB_ISNE_FC32) ;
    GETNAME (GxB_ISNE_FC64) ;

    GETNAME (GxB_ISGT_BOOL) ;
    GETNAME (GxB_ISGT_INT8) ;
    GETNAME (GxB_ISGT_INT16) ;
    GETNAME (GxB_ISGT_INT32) ;
    GETNAME (GxB_ISGT_INT64) ;
    GETNAME (GxB_ISGT_UINT8) ;
    GETNAME (GxB_ISGT_UINT16) ;
    GETNAME (GxB_ISGT_UINT32) ;
    GETNAME (GxB_ISGT_UINT64) ;
    GETNAME (GxB_ISGT_FP32) ;
    GETNAME (GxB_ISGT_FP64) ;

    GETNAME (GxB_ISLT_BOOL) ;
    GETNAME (GxB_ISLT_INT8) ;
    GETNAME (GxB_ISLT_INT16) ;
    GETNAME (GxB_ISLT_INT32) ;
    GETNAME (GxB_ISLT_INT64) ;
    GETNAME (GxB_ISLT_UINT8) ;
    GETNAME (GxB_ISLT_UINT16) ;
    GETNAME (GxB_ISLT_UINT32) ;
    GETNAME (GxB_ISLT_UINT64) ;
    GETNAME (GxB_ISLT_FP32) ;
    GETNAME (GxB_ISLT_FP64) ;

    GETNAME (GxB_ISGE_BOOL) ;
    GETNAME (GxB_ISGE_INT8) ;
    GETNAME (GxB_ISGE_INT16) ;
    GETNAME (GxB_ISGE_INT32) ;
    GETNAME (GxB_ISGE_INT64) ;
    GETNAME (GxB_ISGE_UINT8) ;
    GETNAME (GxB_ISGE_UINT16) ;
    GETNAME (GxB_ISGE_UINT32) ;
    GETNAME (GxB_ISGE_UINT64) ;
    GETNAME (GxB_ISGE_FP32) ;
    GETNAME (GxB_ISGE_FP64) ;

    GETNAME (GxB_ISLE_BOOL) ;
    GETNAME (GxB_ISLE_INT8) ;
    GETNAME (GxB_ISLE_INT16) ;
    GETNAME (GxB_ISLE_INT32) ;
    GETNAME (GxB_ISLE_INT64) ;
    GETNAME (GxB_ISLE_UINT8) ;
    GETNAME (GxB_ISLE_UINT16) ;
    GETNAME (GxB_ISLE_UINT32) ;
    GETNAME (GxB_ISLE_UINT64) ;
    GETNAME (GxB_ISLE_FP32) ;
    GETNAME (GxB_ISLE_FP64) ;

    GETNAME (GrB_MIN_BOOL) ;
    GETNAME (GrB_MIN_INT8) ;
    GETNAME (GrB_MIN_INT16) ;
    GETNAME (GrB_MIN_INT32) ;
    GETNAME (GrB_MIN_INT64) ;
    GETNAME (GrB_MIN_UINT8) ;
    GETNAME (GrB_MIN_UINT16) ;
    GETNAME (GrB_MIN_UINT32) ;
    GETNAME (GrB_MIN_UINT64) ;
    GETNAME (GrB_MIN_FP32) ;
    GETNAME (GrB_MIN_FP64) ;

    GETNAME (GrB_MAX_BOOL) ;
    GETNAME (GrB_MAX_INT8) ;
    GETNAME (GrB_MAX_INT16) ;
    GETNAME (GrB_MAX_INT32) ;
    GETNAME (GrB_MAX_INT64) ;
    GETNAME (GrB_MAX_UINT8) ;
    GETNAME (GrB_MAX_UINT16) ;
    GETNAME (GrB_MAX_UINT32) ;
    GETNAME (GrB_MAX_UINT64) ;
    GETNAME (GrB_MAX_FP32) ;
    GETNAME (GrB_MAX_FP64) ;

    GETNAME (GrB_LOR) ;
    GETNAM2 (GxB_LOR_BOOL,      "GrB_LOR") ;
    GETNAME (GxB_LOR_INT8) ;
    GETNAME (GxB_LOR_INT16) ;
    GETNAME (GxB_LOR_INT32) ;
    GETNAME (GxB_LOR_INT64) ;
    GETNAME (GxB_LOR_UINT8) ;
    GETNAME (GxB_LOR_UINT16) ;
    GETNAME (GxB_LOR_UINT32) ;
    GETNAME (GxB_LOR_UINT64) ;
    GETNAME (GxB_LOR_FP32) ;
    GETNAME (GxB_LOR_FP64) ;

    GETNAME (GrB_LAND) ;
    GETNAM2 (GxB_LAND_BOOL,      "GrB_LAND") ;
    GETNAME (GxB_LAND_INT8) ;
    GETNAME (GxB_LAND_INT16) ;
    GETNAME (GxB_LAND_INT32) ;
    GETNAME (GxB_LAND_INT64) ;
    GETNAME (GxB_LAND_UINT8) ;
    GETNAME (GxB_LAND_UINT16) ;
    GETNAME (GxB_LAND_UINT32) ;
    GETNAME (GxB_LAND_UINT64) ;
    GETNAME (GxB_LAND_FP32) ;
    GETNAME (GxB_LAND_FP64) ;

    GETNAME (GrB_LXOR) ;
    GETNAM2 (GxB_LXOR_BOOL,      "GrB_LXOR") ;
    GETNAME (GxB_LXOR_INT8) ;
    GETNAME (GxB_LXOR_INT16) ;
    GETNAME (GxB_LXOR_INT32) ;
    GETNAME (GxB_LXOR_INT64) ;
    GETNAME (GxB_LXOR_UINT8) ;
    GETNAME (GxB_LXOR_UINT16) ;
    GETNAME (GxB_LXOR_UINT32) ;
    GETNAME (GxB_LXOR_UINT64) ;
    GETNAME (GxB_LXOR_FP32) ;
    GETNAME (GxB_LXOR_FP64) ;

    GETNAME (GrB_LXNOR) ;

    GETNAME (GxB_ATAN2_FP32) ;
    GETNAME (GxB_ATAN2_FP64) ;

    GETNAME (GxB_HYPOT_FP32) ;
    GETNAME (GxB_HYPOT_FP64) ;

    GETNAME (GxB_FMOD_FP32) ;
    GETNAME (GxB_FMOD_FP64) ;

    GETNAME (GxB_REMAINDER_FP32) ;
    GETNAME (GxB_REMAINDER_FP64) ;

    GETNAME (GxB_LDEXP_FP32) ;
    GETNAME (GxB_LDEXP_FP64) ;

    GETNAME (GxB_COPYSIGN_FP32) ;
    GETNAME (GxB_COPYSIGN_FP64) ;

    GETNAME (GrB_BOR_INT8) ;
    GETNAME (GrB_BOR_INT16) ;
    GETNAME (GrB_BOR_INT32) ;
    GETNAME (GrB_BOR_INT64) ;
    GETNAME (GrB_BOR_UINT8) ;
    GETNAME (GrB_BOR_UINT16) ;
    GETNAME (GrB_BOR_UINT32) ;
    GETNAME (GrB_BOR_UINT64) ;

    GETNAME (GrB_BAND_INT8) ;
    GETNAME (GrB_BAND_INT16) ;
    GETNAME (GrB_BAND_INT32) ;
    GETNAME (GrB_BAND_INT64) ;
    GETNAME (GrB_BAND_UINT8) ;
    GETNAME (GrB_BAND_UINT16) ;
    GETNAME (GrB_BAND_UINT32) ;
    GETNAME (GrB_BAND_UINT64) ;

    GETNAME (GrB_BXOR_INT8) ;
    GETNAME (GrB_BXOR_INT16) ;
    GETNAME (GrB_BXOR_INT32) ;
    GETNAME (GrB_BXOR_INT64) ;
    GETNAME (GrB_BXOR_UINT8) ;
    GETNAME (GrB_BXOR_UINT16) ;
    GETNAME (GrB_BXOR_UINT32) ;
    GETNAME (GrB_BXOR_UINT64) ;

    GETNAME (GrB_BXNOR_INT8) ;
    GETNAME (GrB_BXNOR_INT16) ;
    GETNAME (GrB_BXNOR_INT32) ;
    GETNAME (GrB_BXNOR_INT64) ;
    GETNAME (GrB_BXNOR_UINT8) ;
    GETNAME (GrB_BXNOR_UINT16) ;
    GETNAME (GrB_BXNOR_UINT32) ;
    GETNAME (GrB_BXNOR_UINT64) ;

    GETNAME (GxB_BGET_INT8) ;
    GETNAME (GxB_BGET_INT16) ;
    GETNAME (GxB_BGET_INT32) ;
    GETNAME (GxB_BGET_INT64) ;
    GETNAME (GxB_BGET_UINT8) ;
    GETNAME (GxB_BGET_UINT16) ;
    GETNAME (GxB_BGET_UINT32) ;
    GETNAME (GxB_BGET_UINT64) ;

    GETNAME (GxB_BSET_INT8) ;
    GETNAME (GxB_BSET_INT16) ;
    GETNAME (GxB_BSET_INT32) ;
    GETNAME (GxB_BSET_INT64) ;
    GETNAME (GxB_BSET_UINT8) ;
    GETNAME (GxB_BSET_UINT16) ;
    GETNAME (GxB_BSET_UINT32) ;
    GETNAME (GxB_BSET_UINT64) ;

    GETNAME (GxB_BCLR_INT8) ;
    GETNAME (GxB_BCLR_INT16) ;
    GETNAME (GxB_BCLR_INT32) ;
    GETNAME (GxB_BCLR_INT64) ;
    GETNAME (GxB_BCLR_UINT8) ;
    GETNAME (GxB_BCLR_UINT16) ;
    GETNAME (GxB_BCLR_UINT32) ;
    GETNAME (GxB_BCLR_UINT64) ;

    GETNAME (GxB_BSHIFT_INT8) ;
    GETNAME (GxB_BSHIFT_INT16) ;
    GETNAME (GxB_BSHIFT_INT32) ;
    GETNAME (GxB_BSHIFT_INT64) ;
    GETNAME (GxB_BSHIFT_UINT8) ;
    GETNAME (GxB_BSHIFT_UINT16) ;
    GETNAME (GxB_BSHIFT_UINT32) ;
    GETNAME (GxB_BSHIFT_UINT64) ;

    GETNAM2 (GrB_EQ_BOOL,       "GrB_LXNOR") ;
    GETNAME (GrB_EQ_INT8) ;
    GETNAME (GrB_EQ_INT16) ;
    GETNAME (GrB_EQ_INT32) ;
    GETNAME (GrB_EQ_INT64) ;
    GETNAME (GrB_EQ_UINT8) ;
    GETNAME (GrB_EQ_UINT16) ;
    GETNAME (GrB_EQ_UINT32) ;
    GETNAME (GrB_EQ_UINT64) ;
    GETNAME (GrB_EQ_FP32) ;
    GETNAME (GrB_EQ_FP64) ;
    GETNAME (GxB_EQ_FC32) ;
    GETNAME (GxB_EQ_FC64) ;

    GETNAME (GrB_NE_BOOL) ;
    GETNAME (GrB_NE_INT8) ;
    GETNAME (GrB_NE_INT16) ;
    GETNAME (GrB_NE_INT32) ;
    GETNAME (GrB_NE_INT64) ;
    GETNAME (GrB_NE_UINT8) ;
    GETNAME (GrB_NE_UINT16) ;
    GETNAME (GrB_NE_UINT32) ;
    GETNAME (GrB_NE_UINT64) ;
    GETNAME (GrB_NE_FP32) ;
    GETNAME (GrB_NE_FP64) ;
    GETNAME (GxB_NE_FC32) ;
    GETNAME (GxB_NE_FC64) ;

    GETNAME (GrB_GT_BOOL) ;
    GETNAME (GrB_GT_INT8) ;
    GETNAME (GrB_GT_INT16) ;
    GETNAME (GrB_GT_INT32) ;
    GETNAME (GrB_GT_INT64) ;
    GETNAME (GrB_GT_UINT8) ;
    GETNAME (GrB_GT_UINT16) ;
    GETNAME (GrB_GT_UINT32) ;
    GETNAME (GrB_GT_UINT64) ;
    GETNAME (GrB_GT_FP32) ;
    GETNAME (GrB_GT_FP64) ;

    GETNAME (GrB_LT_BOOL) ;
    GETNAME (GrB_LT_INT8) ;
    GETNAME (GrB_LT_INT16) ;
    GETNAME (GrB_LT_INT32) ;
    GETNAME (GrB_LT_INT64) ;
    GETNAME (GrB_LT_UINT8) ;
    GETNAME (GrB_LT_UINT16) ;
    GETNAME (GrB_LT_UINT32) ;
    GETNAME (GrB_LT_UINT64) ;
    GETNAME (GrB_LT_FP32) ;
    GETNAME (GrB_LT_FP64) ;

    GETNAME (GrB_GE_BOOL) ;
    GETNAME (GrB_GE_INT8) ;
    GETNAME (GrB_GE_INT16) ;
    GETNAME (GrB_GE_INT32) ;
    GETNAME (GrB_GE_INT64) ;
    GETNAME (GrB_GE_UINT8) ;
    GETNAME (GrB_GE_UINT16) ;
    GETNAME (GrB_GE_UINT32) ;
    GETNAME (GrB_GE_UINT64) ;
    GETNAME (GrB_GE_FP32) ;
    GETNAME (GrB_GE_FP64) ;

    GETNAME (GrB_LE_BOOL) ;
    GETNAME (GrB_LE_INT8) ;
    GETNAME (GrB_LE_INT16) ;
    GETNAME (GrB_LE_INT32) ;
    GETNAME (GrB_LE_INT64) ;
    GETNAME (GrB_LE_UINT8) ;
    GETNAME (GrB_LE_UINT16) ;
    GETNAME (GrB_LE_UINT32) ;
    GETNAME (GrB_LE_UINT64) ;
    GETNAME (GrB_LE_FP32) ;
    GETNAME (GrB_LE_FP64) ;

    GETNAME (GxB_CMPLX_FP32) ;
    GETNAME (GxB_CMPLX_FP64) ;

    GETNAME (GxB_FIRSTI_INT32) ;    GETNAME (GxB_FIRSTI_INT64) ;
    GETNAME (GxB_FIRSTI1_INT32) ;   GETNAME (GxB_FIRSTI1_INT64) ;
    GETNAME (GxB_FIRSTJ_INT32) ;    GETNAME (GxB_FIRSTJ_INT64) ;
    GETNAME (GxB_FIRSTJ1_INT32) ;   GETNAME (GxB_FIRSTJ1_INT64) ;
    GETNAME (GxB_SECONDI_INT32) ;   GETNAME (GxB_SECONDI_INT64) ;
    GETNAME (GxB_SECONDI1_INT32) ;  GETNAME (GxB_SECONDI1_INT64) ;
    GETNAME (GxB_SECONDJ_INT32) ;   GETNAME (GxB_SECONDJ_INT64) ;
    GETNAME (GxB_SECONDJ1_INT32) ;  GETNAME (GxB_SECONDJ1_INT64) ;

    GETNAME (GxB_IGNORE_DUP) ;

    //--------------------------------------------------------------------------
    // other get/set methods for GrB_BinaryOp
    //--------------------------------------------------------------------------

    OK (GrB_BinaryOp_get_INT32_(GrB_MAX_FP32, &code, GrB_INP0_TYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    OK (GrB_BinaryOp_get_SIZE_(GrB_MAX_FP32, &size, GrB_INP0_TYPE_STRING)) ;
    CHECK (size == strlen ("GrB_FP32") + 1) ;

    OK (GrB_BinaryOp_get_String_(GrB_MAX_FP32, name, GrB_INP0_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_BinaryOp_get_SIZE_(GrB_MAX_INT32, &size, GrB_INP1_TYPE_STRING)) ;
    CHECK (size == strlen ("GrB_INT32") + 1) ;

    OK (GrB_BinaryOp_get_String_(GrB_MAX_INT32, name, GrB_INP1_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_INT32")) ;

    OK (GrB_BinaryOp_get_INT32_(GrB_MAX_FP64, &code, GrB_OUTP_TYPE_CODE)) ;
    CHECK (code == GrB_FP64_CODE) ;

    OK (GrB_BinaryOp_get_SIZE_(GrB_MAX_FP64, &size, GrB_OUTP_TYPE_STRING)) ;
    CHECK (size == strlen ("GrB_FP64") + 1) ;

    OK (GrB_BinaryOp_get_String_(GrB_MAX_FP64, name, GrB_OUTP_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP64")) ;

    OK (GrB_BinaryOp_get_Scalar_(GrB_MAX_FP32, s_int32, GrB_INP0_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&code, s_int32)) ;
    CHECK (code == GrB_FP32_CODE) ;

    OK (GrB_BinaryOp_get_Scalar_(GrB_LAND, s_int32, GrB_OUTP_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&code, s_int32)) ;
    CHECK (code == GrB_BOOL_CODE) ;

    OK (GrB_BinaryOp_get_INT32_(GrB_PLUS_FP64, &code, GrB_INP1_TYPE_CODE)) ;
    CHECK (code == GrB_FP64_CODE) ;

    OK (GrB_BinaryOp_get_Scalar_(GrB_LAND, s_int32, GrB_INP1_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&code, s_int32)) ;
    CHECK (code == GrB_BOOL_CODE) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_BinaryOp_get_INT32_(GrB_LAND, &code, GrB_NAME)) ;
    ERR (GrB_BinaryOp_get_String_(GrB_MAX_INT32, name, 999)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_BinaryOp_get_VOID_(GrB_LAND, nothing, 0)) ;

    OK (GrB_BinaryOp_new (&binop, myfunc, GrB_FP32, GrB_FP32, GrB_FP32)) ;
    OK (GrB_BinaryOp_get_SIZE_(binop, &size, GrB_NAME)) ;
    CHECK (size == 1) ;
    OK (GrB_BinaryOp_get_SIZE_(binop, &size, GxB_JIT_C_NAME)) ;
    CHECK (size == 1) ;
    OK (GrB_BinaryOp_get_SIZE_(binop, &size, GxB_JIT_C_DEFINITION)) ;
    CHECK (size == 1) ;
    OK (GrB_BinaryOp_set_String_(binop, "myfunc", GxB_JIT_C_NAME)) ;
    OK (GrB_BinaryOp_get_String_(binop, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "myfunc")) ;
    CHECK (binop->hash == UINT64_MAX) ;
    METHOD (GrB_BinaryOp_set_String (binop, MYFUNC_DEFN, GxB_JIT_C_DEFINITION)) ;
    OK (GrB_BinaryOp_get_String_(binop, defn, GxB_JIT_C_DEFINITION)) ;
    CHECK (MATCH (defn, MYFUNC_DEFN)) ;
    CHECK (binop->hash != UINT64_MAX) ;
    OK (GxB_print (binop, 3)) ;

    OK (GrB_BinaryOp_set_String_(binop, "user name for myfunc", GrB_NAME)) ;
    OK (GrB_BinaryOp_get_String_(binop, name, GrB_NAME)) ;
    CHECK (MATCH (name, "user name for myfunc")) ;
    expected = GrB_ALREADY_SET ;
    ERR (GrB_BinaryOp_set_String_(binop, "another user name", GrB_NAME)) ;
    printf ("    test GrB_ALREADY_SET: ok\n") ;

    OK (GrB_BinaryOp_get_INT32_(binop, &code, GrB_INP1_TYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_BinaryOp_set_Scalar_(binop, s_int32, 0)) ;
    ERR (GrB_BinaryOp_set_INT32_(binop, 0, 0)) ;
    ERR (GrB_BinaryOp_set_VOID_(binop, nothing, 0, 0)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&s_fp64) ;
    GrB_free (&s_fp32) ;
    GrB_free (&s_int32) ;
    GrB_free (&binop) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test28:  all tests passed\n\n") ;
}

