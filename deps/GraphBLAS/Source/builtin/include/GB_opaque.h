//------------------------------------------------------------------------------
// GB_opaque.h: definitions of opaque objects
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_OPAQUE_H
#define GB_OPAQUE_H

#define GB_OPAQUE(x) GB (GB_EVAL2 (_opaque__, x))

//------------------------------------------------------------------------------
// GB_void: like void, but valid for pointer arithmetic
//------------------------------------------------------------------------------

typedef unsigned char GB_void ;

//------------------------------------------------------------------------------
// type codes for GrB_Type
//------------------------------------------------------------------------------

typedef enum
{
    // the 14 scalar types: 13 built-in types, and one user-defined type code
    GB_ignore_code  = 0,
    GB_BOOL_code    = 1,        // 'logical' in @GrB interface
    GB_INT8_code    = 2,
    GB_UINT8_code   = 3,
    GB_INT16_code   = 4,
    GB_UINT16_code  = 5,
    GB_INT32_code   = 6,
    GB_UINT32_code  = 7,
    GB_INT64_code   = 8,
    GB_UINT64_code  = 9,
    GB_FP32_code    = 10,       // float ('single' in @GrB interface)
    GB_FP64_code    = 11,       // double
    GB_FC32_code    = 12,       // float complex ('single complex' in @GrB)
    GB_FC64_code    = 13,       // double complex
    GB_UDT_code     = 14        // void *, user-defined type
}
GB_Type_code ;                  // enumerated type code

//------------------------------------------------------------------------------
// opcodes for all operators
//------------------------------------------------------------------------------

typedef enum
{

    GB_NOP_code = 0,    // no operation

    //==========================================================================
    // unary operators
    //==========================================================================

    //--------------------------------------------------------------------------
    // primary unary operators x=f(x)
    //--------------------------------------------------------------------------

    GB_ONE_unop_code       = 1,    // z = 1
    GB_IDENTITY_unop_code  = 2,    // z = x
    GB_AINV_unop_code      = 3,    // z = -x
    GB_ABS_unop_code       = 4,    // z = abs(x) ; z is real if x is complex
    GB_MINV_unop_code      = 5,    // z = 1/x ; special cases for bool and ints
    GB_LNOT_unop_code      = 6,    // z = !x
    GB_BNOT_unop_code      = 7,    // z = ~x (bitwise complement)

    //--------------------------------------------------------------------------
    // unary operators for floating-point types (real and complex)
    //--------------------------------------------------------------------------

    GB_SQRT_unop_code      = 8,    // z = sqrt (x)
    GB_LOG_unop_code       = 9,    // z = log (x)
    GB_EXP_unop_code       = 10,   // z = exp (x)
    GB_SIN_unop_code       = 11,   // z = sin (x)
    GB_COS_unop_code       = 12,   // z = cos (x)
    GB_TAN_unop_code       = 13,   // z = tan (x)
    GB_ASIN_unop_code      = 14,   // z = asin (x)
    GB_ACOS_unop_code      = 15,   // z = acos (x)
    GB_ATAN_unop_code      = 16,   // z = atan (x)
    GB_SINH_unop_code      = 17,   // z = sinh (x)
    GB_COSH_unop_code      = 18,   // z = cosh (x)
    GB_TANH_unop_code      = 19,   // z = tanh (x)
    GB_ASINH_unop_code     = 20,   // z = asinh (x)
    GB_ACOSH_unop_code     = 21,   // z = acosh (x)
    GB_ATANH_unop_code     = 22,   // z = atanh (x)
    GB_SIGNUM_unop_code    = 23,   // z = signum (x)
    GB_CEIL_unop_code      = 24,   // z = ceil (x)
    GB_FLOOR_unop_code     = 25,   // z = floor (x)
    GB_ROUND_unop_code     = 26,   // z = round (x)
    GB_TRUNC_unop_code     = 27,   // z = trunc (x)
    GB_EXP2_unop_code      = 28,   // z = exp2 (x)
    GB_EXPM1_unop_code     = 29,   // z = expm1 (x)
    GB_LOG10_unop_code     = 30,   // z = log10 (x)
    GB_LOG1P_unop_code     = 31,   // z = log1P (x)
    GB_LOG2_unop_code      = 32,   // z = log2 (x)

    //--------------------------------------------------------------------------
    // unary operators for real floating-point types
    //--------------------------------------------------------------------------

    GB_LGAMMA_unop_code    = 33,   // z = lgamma (x)
    GB_TGAMMA_unop_code    = 34,   // z = tgamma (x)
    GB_ERF_unop_code       = 35,   // z = erf (x)
    GB_ERFC_unop_code      = 36,   // z = erfc (x)
    GB_CBRT_unop_code      = 37,   // z = cbrt (x)
    GB_FREXPX_unop_code    = 38,   // z = frexpx (x), mantissa of C11 frexp
    GB_FREXPE_unop_code    = 39,   // z = frexpe (x), exponent of C11 frexp

    //--------------------------------------------------------------------------
    // unary operators for complex types only
    //--------------------------------------------------------------------------

    GB_CONJ_unop_code      = 40,   // z = conj (x)

    //--------------------------------------------------------------------------
    // unary operators where z is real and x is complex
    //--------------------------------------------------------------------------

    GB_CREAL_unop_code     = 41,   // z = creal (x)
    GB_CIMAG_unop_code     = 42,   // z = cimag (x)
    GB_CARG_unop_code      = 43,   // z = carg (x)

    //--------------------------------------------------------------------------
    // unary operators where z is bool and x is any floating-point type
    //--------------------------------------------------------------------------

    GB_ISINF_unop_code     = 44,   // z = isinf (x)
    GB_ISNAN_unop_code     = 45,   // z = isnan (x)
    GB_ISFINITE_unop_code  = 46,   // z = isfinite (x)

    //--------------------------------------------------------------------------
    // positional unary operators: z is int32 or int64, x is ignored
    //--------------------------------------------------------------------------

    GB_POSITIONI_unop_code     = 47,   // z = position_i(A(i,j)) == i
    GB_POSITIONI1_unop_code    = 48,   // z = position_i1(A(i,j)) == i+1
    GB_POSITIONJ_unop_code     = 49,   // z = position_j(A(i,j)) == j
    GB_POSITIONJ1_unop_code    = 50,   // z = position_j1(A(i,j)) == j+1

    GB_USER_unop_code = 51,

    // true if opcode is for a GrB_UnaryOp
    #define GB_IS_UNARYOP_CODE(opcode) \
        ((opcode) >= GB_ONE_unop_code && \
         (opcode) <= GB_USER_unop_code)

    // true if opcode is for a GrB_UnaryOp positional operator
    #define GB_IS_BUILTIN_UNOP_CODE_POSITIONAL(opcode) \
        ((opcode) >= GB_POSITIONI_unop_code && \
         (opcode) <= GB_POSITIONJ1_unop_code)

    //==========================================================================
    // index_unary operators
    //==========================================================================

    // operator codes used in GrB_IndexUnaryOp structures

    // Result is INT32 or INT64, depending on i and/or j, and thunk:
    GB_ROWINDEX_idxunop_code  = 52,   // (i+thunk): row index - thunk
    GB_COLINDEX_idxunop_code  = 53,   // (j+thunk): col index - thunk
    GB_DIAGINDEX_idxunop_code = 54,   // (j-(i+thunk)): diag index + thunk
    GB_FLIPDIAGINDEX_idxunop_code = 55,   // (i-(j+thunk)), internal use only

    // Result is BOOL, depending on i and/or j, and thunk:
    GB_TRIL_idxunop_code      = 56,   // (j <= (i+thunk)): tril (A,thunk)
    GB_TRIU_idxunop_code      = 57,   // (j >= (i+thunk)): triu (A,thunk)
    GB_DIAG_idxunop_code      = 58,   // (j == (i+thunk)): diag(A,thunk)
    GB_OFFDIAG_idxunop_code   = 59,   // (j != (i+thunk)): offdiag(A,thunk)
    GB_COLLE_idxunop_code     = 60,   // (j <= thunk): A (:,0:thunk)
    GB_COLGT_idxunop_code     = 61,   // (j > thunk): A (:,thunk+1:ncols-1)
    GB_ROWLE_idxunop_code     = 62,   // (i <= thunk): A (0:thunk,:)
    GB_ROWGT_idxunop_code     = 63,   // (i > thunk): A (thunk+1:nrows-1,:)

    // Result is BOOL, depending on whether or not A(i,j) is a zombie
    GB_NONZOMBIE_idxunop_code = 64,

    // Result is BOOL, depending on the value aij and thunk:
    GB_VALUENE_idxunop_code   = 65,   // (aij != thunk)
    GB_VALUEEQ_idxunop_code   = 66,   // (aij == thunk)
    GB_VALUEGT_idxunop_code   = 67,   // (aij > thunk)
    GB_VALUEGE_idxunop_code   = 68,   // (aij >= thunk)
    GB_VALUELT_idxunop_code   = 69,   // (aij < thunk)
    GB_VALUELE_idxunop_code   = 70,   // (aij <= thunk)

    GB_USER_idxunop_code = 71,

    // true if opcode is for a GrB_IndexUnaryOp
    #define GB_IS_INDEXUNARYOP_CODE(opcode) \
        ((opcode) >= GB_ROWINDEX_idxunop_code && \
         (opcode) <= GB_USER_idxunop_code)

    // true if opcode is for a GrB_IndexUnaryOp positional operator
    #define GB_IS_INDEXUNARYOP_CODE_POSITIONAL(opcode) \
        ((opcode) >= GB_ROWINDEX_idxunop_code && \
         (opcode) <= GB_ROWGT_idxunop_code)

    //==========================================================================
    // binary operators
    //==========================================================================

    //--------------------------------------------------------------------------
    // binary ops for 14 valid monoids, including user-defined (72 to 85):
    //--------------------------------------------------------------------------

    GB_USER_binop_code      = 72,   // user defined binary op
    GB_ANY_binop_code       = 73,   // z = x or y, selected arbitrarily
    GB_MIN_binop_code       = 74,   // z = min(x,y)
    GB_MAX_binop_code       = 75,   // z = max(x,y)
    GB_PLUS_binop_code      = 76,   // z = x + y
    GB_TIMES_binop_code     = 77,   // z = x * y
    GB_LOR_binop_code       = 78,   // z = (x != 0) || (y != 0)
    GB_LAND_binop_code      = 79,   // z = (x != 0) && (y != 0)
    GB_LXOR_binop_code      = 80,   // z = (x != 0) != (y != 0)
    GB_EQ_binop_code        = 81,   // z = (x == y), is LXNOR for bool
    GB_BOR_binop_code       = 82,   // z = (x | y), bitwise or
    GB_BAND_binop_code      = 83,   // z = (x & y), bitwise and
    GB_BXOR_binop_code      = 84,   // z = (x ^ y), bitwise xor
    GB_BXNOR_binop_code     = 85,   // z = ~(x ^ y), bitwise xnor

    //--------------------------------------------------------------------------
    // other binary operators 
    //--------------------------------------------------------------------------

    GB_NE_binop_code        = 86,   // z = (x != y)
    GB_FIRST_binop_code     = 87,   // z = x
    GB_SECOND_binop_code    = 88,   // z = y
    GB_PAIR_binop_code      = 89,   // z = 1
    GB_MINUS_binop_code     = 90,   // z = x - y
    GB_RMINUS_binop_code    = 91,   // z = y - x
    GB_DIV_binop_code       = 92,   // z = x / y
    GB_RDIV_binop_code      = 93,   // z = y / x
    GB_POW_binop_code       = 94,   // z = pow (x,y)
    GB_ISEQ_binop_code      = 95,   // z = (x == y)
    GB_ISNE_binop_code      = 96,   // z = (x != y)
    GB_ISGT_binop_code      = 97,   // z = (x >  y)
    GB_ISLT_binop_code      = 98,   // z = (x <  y)
    GB_ISGE_binop_code      = 99,   // z = (x >= y)
    GB_ISLE_binop_code      = 100,  // z = (x <= y)
    GB_BGET_binop_code      = 101,  // z = bitget (x,y)
    GB_BSET_binop_code      = 102,  // z = bitset (x,y)
    GB_BCLR_binop_code      = 103,  // z = bitclr (x,y)
    GB_BSHIFT_binop_code    = 104,  // z = bitshift (x,y)
    GB_GT_binop_code        = 105,  // z = (x >  y)
    GB_LT_binop_code        = 106,  // z = (x <  y)
    GB_GE_binop_code        = 107,  // z = (x >= y)
    GB_LE_binop_code        = 108,  // z = (x <= y)
    GB_ATAN2_binop_code     = 109,  // z = atan2 (x,y)
    GB_HYPOT_binop_code     = 110,  // z = hypot (x,y)
    GB_FMOD_binop_code      = 111,  // z = fmod (x,y)
    GB_REMAINDER_binop_code = 112,  // z = remainder (x,y)
    GB_COPYSIGN_binop_code  = 113,  // z = copysign (x,y)
    GB_LDEXP_binop_code     = 114,  // z = ldexp (x,y)
    GB_CMPLX_binop_code     = 115,  // z = cmplx (x,y)

    //--------------------------------------------------------------------------
    // built-in positional binary operators: z is int64, x and y are ignored
    //--------------------------------------------------------------------------

    GB_FIRSTI_binop_code    = 116,  // z = first_i(A(i,j),y) == i
    GB_FIRSTI1_binop_code   = 117,  // z = first_i1(A(i,j),y) == i+1
    GB_FIRSTJ_binop_code    = 118,  // z = first_j(A(i,j),y) == j
    GB_FIRSTJ1_binop_code   = 119,  // z = first_j1(A(i,j),y) == j+1
    GB_SECONDI_binop_code   = 120,  // z = second_i(x,B(i,j)) == i
    GB_SECONDI1_binop_code  = 121,  // z = second_i1(x,B(i,j)) == i+1
    GB_SECONDJ_binop_code   = 122,  // z = second_j(x,B(i,j)) == j
    GB_SECONDJ1_binop_code  = 123,  // z = second_j1(x,B(i,j)) == j+1

    // true if opcode is for a GrB_BinaryOp
    #define GB_IS_BINARYOP_CODE(opcode) \
        ((opcode) >= GB_USER_binop_code && \
         (opcode) <= GB_SECONDJ1_binop_code)

    // true if opcode is for a GrB_BinaryOp positional operator
    #define GB_IS_BUILTIN_BINOP_CODE_POSITIONAL(opcode) \
        ((opcode) >= GB_FIRSTI_binop_code && \
         (opcode) <= GB_SECONDJ1_binop_code)

    //--------------------------------------------------------------------------
    // index binary operators:
    //--------------------------------------------------------------------------

    GB_USER_idxbinop_code = 124,

    // true if opcode is for a GxB_IndexBinaryOp
    #define GB_IS_INDEXBINARYOP_CODE(opcode) ((opcode) == GB_USER_idxbinop_code)

    //==========================================================================
    // built-in GxB_SelectOp operators (DEPRECATED: do not use)
    //==========================================================================

    // built-in positional select operators: thunk optional; defaults to zero
    GB_TRIL_selop_code      = 125,
    GB_TRIU_selop_code      = 126,
    GB_DIAG_selop_code      = 127,
    GB_OFFDIAG_selop_code   = 128,

    // built-in select operators, no thunk used
    GB_NONZERO_selop_code   = 129,
    GB_EQ_ZERO_selop_code   = 130,
    GB_GT_ZERO_selop_code   = 131,
    GB_GE_ZERO_selop_code   = 132,
    GB_LT_ZERO_selop_code   = 133,
    GB_LE_ZERO_selop_code   = 134,

    // built-in select operators, thunk optional; defaults to zero
    GB_NE_THUNK_selop_code  = 135,
    GB_EQ_THUNK_selop_code  = 136,
    GB_GT_THUNK_selop_code  = 137,
    GB_GE_THUNK_selop_code  = 138,
    GB_LT_THUNK_selop_code  = 139,
    GB_LE_THUNK_selop_code  = 140

    // true if opcode is for a GxB_SelectOp
    #define GB_IS_SELECTOP_CODE(opcode) \
        ((opcode) >= GB_TRIL_selop_code && (opcode) <= GB_LE_THUNK_selop_code)

    // true if opcode is for a GxB_SelectOp positional operator
    #define GB_IS_SELECTOP_CODE_POSITIONAL(opcode) \
        ((opcode) >= GB_TRIL_selop_code && \
         (opcode) <= GB_OFFDIAG_selop_code)

}
GB_Opcode ;

// true if the opcode is a positional operator of any kind
#define GB_OPCODE_IS_POSITIONAL(opcode)                 \
    (GB_IS_BUILTIN_UNOP_CODE_POSITIONAL (opcode) ||     \
     GB_IS_INDEXUNARYOP_CODE_POSITIONAL (opcode) ||     \
     GB_IS_INDEXBINARYOP_CODE (opcode) ||               \
     GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (opcode) ||    \
     GB_IS_SELECTOP_CODE_POSITIONAL (opcode))

// true if the op is a unary or binary positional operator
#define GB_OP_IS_POSITIONAL(op) \
    (((op) == NULL) ? false : GB_OPCODE_IS_POSITIONAL ((op)->opcode))

//------------------------------------------------------------------------------
// opaque content of GraphBLAS objects
//------------------------------------------------------------------------------

// GB_MAGIC is an arbitrary number that is placed inside each object when it is
// initialized, as a way of detecting uninitialized objects.
#define GB_MAGIC  0x72657473786f62ULL

// The magic number is set to GB_FREED when the object is freed, as a way of
// helping to detect dangling pointers.
#define GB_FREED  0x6c6c756e786f62ULL

// The value is set to GB_MAGIC2 when the object has been allocated but cannot
// yet be used in most methods and operations.  Currently this is used only for
// when A->p array is allocated but not initialized.
#define GB_MAGIC2 0x7265745f786f62ULL

// Nearly all GraphBLAS objects contain the same first 4 items (except for
// GB_Global_opaque, which has just the first 2).

struct GB_Type_opaque       // content of GrB_Type
{
    int64_t magic ;         // for detecting uninitialized objects
    size_t header_size ;    // size of the malloc'd block for this struct, or 0
    // ---------------------//
    char *user_name ;       // user name for GrB_get/GrB_set
    size_t user_name_size ; // allocated size of user_name for GrB_get/GrB_set
    // ---------------------//
    size_t size ;           // size of the type
    GB_Type_code code ;     // the type code
    int32_t name_len ;      // length of JIT C name; 0 for builtin
    char name [GxB_MAX_NAME_LEN] ;  // JIT C name of the type
    char *defn ;            // type definition
    size_t defn_size ;      // allocated size of the definition
    uint64_t hash ;         // if 0, type is builtin.
                            // if UINT64_MAX, the type cannot be JIT'd.
    GxB_print_function print_function ; // for printing user-defined types
} ;

struct GB_UnaryOp_opaque    // content of GrB_UnaryOp
{
    #include "include/GB_Operator_content.h"
} ;

struct GB_IndexUnaryOp_opaque   // content of GrB_IndexUnaryOp
{
    #include "include/GB_Operator_content.h"
} ;

struct GB_BinaryOp_opaque   // content of GrB_BinaryOp
{
    #include "include/GB_Operator_content.h"
} ;

struct GB_IndexBinaryOp_opaque   // content of GxB_IndexBinaryOp
{
    #include "include/GB_Operator_content.h"
} ;

struct GB_SelectOp_opaque   // content of GxB_SelectOp
{
    #include "include/GB_Operator_content.h"
} ;

struct GB_Operator_opaque   // content of GB_Operator
{
    #include "include/GB_Operator_content.h"
} ;

// Any GrB_UnaryOp, GrB_IndexUnaryOp, GrB_BinaryOp, or GxB_SelectOp can be
// typecasted to a generic GB_Operator object, which is only used internally.
typedef struct GB_Operator_opaque *GB_Operator ;

struct GB_Monoid_opaque     // content of GrB_Monoid
{
    int64_t magic ;         // for detecting uninitialized objects
    size_t header_size ;    // size of the malloc'd block for this struct, or 0
    // ---------------------//
    char *user_name ;       // user name for GrB_get/GrB_set
    size_t user_name_size ; // allocated size of user_name for GrB_get/GrB_set
    // ---------------------//
    GrB_BinaryOp op ;       // binary operator of the monoid
    void *identity ;        // identity of the monoid; type is op->ztype
    void *terminal ;        // early-exit (NULL if no value); type is op->ztype
    size_t identity_size ;  // allocated size of identity, or 0
    size_t terminal_size ;  // allocated size of terminal, or 0
    uint64_t hash ;         // if 0, monoid uses only builtin ops and types.
                            // if UINT64_MAX, the monoid cannot be JIT'd.
} ;

struct GB_Semiring_opaque   // content of GrB_Semiring
{
    int64_t magic ;         // for detecting uninitialized objects
    size_t header_size ;    // size of the malloc'd block for this struct, or 0
    // ---------------------//
    char *user_name ;       // user name for GrB_get/GrB_set
    size_t user_name_size ; // allocated size of user_name for GrB_get/GrB_set
    // ---------------------//
    GrB_Monoid add ;        // add operator of the semiring
    GrB_BinaryOp multiply ; // multiply operator of the semiring
    char *name ;            // name of the semiring; NULL for builtin
    int32_t name_len ;      // length of name; 0 for builtin
    size_t name_size ;      // allocated size of the name
    uint64_t hash ;         // if 0, semiring uses only builtin ops and types
} ;

struct GB_Descriptor_opaque // content of GrB_Descriptor
{
    // first 6 items exactly match GrB_Matrix, GrB_Vector, GrB_Scalar structs:
    int64_t magic ;         // for detecting uninitialized objects
    size_t header_size ;    // size of the malloc'd block for this struct, or 0
    // ---------------------//
    char *user_name ;       // user name for GrB_get/GrB_set
    size_t user_name_size ; // allocated size of user_name for GrB_get/GrB_set
    // ---------------------//
    char *logger ;          // error logger string
    size_t logger_size ;    // size of the malloc'd block for logger, or 0
    // ---------------------//
    // specific to the descriptor struct:
    GrB_Desc_Value out ;    // output descriptor
    GrB_Desc_Value mask ;   // mask descriptor
    GrB_Desc_Value in0 ;    // first input descriptor (A for C=A*B, for example)
    GrB_Desc_Value in1 ;    // second input descriptor (B for C=A*B)
    GrB_Desc_Value axb ;    // for selecting the method for C=A*B
    int compression ;       // compression method for GxB_Matrix_serialize
    bool do_sort ;          // if nonzero, do the sort in GrB_mxm
    int import ;            // if zero (default), trust input data
    int row_list ;          // how to use the row index list, I
    int col_list ;          // how to use the col index list, J
    int val_list ;          // how to use the value list, X
} ;

#define GB_MAX_NGPUS 1024

struct GB_Context_opaque    // content of GxB_Context
{
    int64_t magic ;         // for detecting uninitialized objects
    size_t header_size ;    // size of the malloc'd block for this struct, or 0
    // ---------------------//
    char *user_name ;       // user name for GrB_get/GrB_set
    size_t user_name_size ; // allocated size of user_name for GrB_get/GrB_set
    // ---------------------//
    // OpenMP thread(s):
    double chunk ;          // chunk size for # of threads for small problems
    int32_t nthreads_max ;  // max # threads to use in this call to GraphBLAS
    // GPU(s):
    int32_t ngpus ;         // # of GPUs available to use in this context
                            // (in range 0 to GB_MAX_NGPUS)
    uint16_t gpu_ids [GB_MAX_NGPUS] ;   // using GPUs gpu_ids [0..ngpus-1],
                            // or no GPU if ngpus == 0.
} ;

//------------------------------------------------------------------------------
// GB_Pending data structure: for scalars, vectors, and matrices
//------------------------------------------------------------------------------

// Pending tuples are a list of unsorted (i,j,x) tuples that have not yet been
// added to a matrix.  The indices Pending->i and Pending->j are 32/64 bit, as
// determined by A->i_is_32 and A->j_is_32, respectively.

struct GB_Pending_struct    // list of pending tuples for a matrix
{
    size_t header_size ;    // size of the malloc'd block for this struct, or 0
    int64_t n ;         // number of pending tuples to add to matrix
    int64_t nmax ;      // size of i,j,x
    bool sorted ;       // true if pending tuples are in sorted order
    void *i ;           // row indices of pending tuples
    size_t i_size ;
    void *j ;           // col indices of pending tuples; NULL if A->vdim <= 1
    size_t j_size ;
    GB_void *x ;        // values of pending tuples
    size_t x_size ;
    GrB_Type type ;     // the type of x
    size_t size ;       // type->size
    GrB_BinaryOp op ;   // operator to assemble pending tuples
} ;

typedef struct GB_Pending_struct *GB_Pending ;

//------------------------------------------------------------------------------
// scalar, vector, and matrix types
//------------------------------------------------------------------------------

// true if A is bitmap
#define GB_IS_BITMAP(A) ((A) != NULL && ((A)->b != NULL))

// true if A is full (but not bitmap)
#define GB_IS_FULL(A) \
    ((A) != NULL && (A)->h == NULL && (A)->p == NULL && (A)->i == NULL \
        && (A)->b == NULL)

// true if A is hypersparse
#define GB_IS_HYPERSPARSE(A) ((A) != NULL && ((A)->h != NULL))

// true if A is sparse (but not hypersparse)
#define GB_IS_SPARSE(A) ((A) != NULL && ((A)->h == NULL) && (A)->p != NULL)

struct GB_Scalar_opaque     // content of GrB_Scalar: 1-by-1 standard CSC matrix
{
    #include "include/GB_Matrix_content.h"
} ;

struct GB_Vector_opaque     // content of GrB_Vector: m-by-1 standard CSC matrix
{
    #include "include/GB_Matrix_content.h"
} ;

struct GB_Matrix_opaque     // content of GrB_Matrix
{
    #include "include/GB_Matrix_content.h"
} ;

//------------------------------------------------------------------------------
// Accessing the content of a scalar, vector, or matrix
//------------------------------------------------------------------------------

// A GrB_Matrix has three different types of integers:
//
// (1) A->p can be uint32_t or uint64_t, as determined by A->p_is_32.
//
// (2) These types are all determined by A->i_is_32:
// A->i    can be  int32_t or  int64_t (signed, for flagging zombies: default)
// A->i    can be uint32_t or uint64_t (unsigned, if no zombies appear)
//
// (3) These types are all determined by A->j_is_32:
// A->h    can be uint32_t or uint64_t
// A->Y->p can be uint32_t or uint64_t
// A->Y->i can be uint32_t or uint64_t (never has zombies)
// A->Y->x can be uint32_t or uint64_t

// For examples on how these macros expand, see Source/math/include/GB_zombie.h.

// helper macro: declare a 32/64-bit integer array I
#define GB_MDECL(I,const,u)                         \
    const void *I = NULL ;                          \
    const u ## int32_t *restrict I ## 32 = NULL ;   \
    const u ## int64_t *restrict I ## 64 = NULL

// assign to a type-specific pointer from a void pointer, 32/64 bit
#define GB_IPTR(I,is_32)                            \
    I ## 32 = (is_32) ? I : NULL ;                  \
    I ## 64 = (is_32) ? NULL : I

// general method for getting an entry from the Ah array of a matrix; used for
// generic kernels, and JIT kernels for hyperlist arrays created inside the
// kernel (assign JIT kernels only)
#define GBh(Ah,k)                       \
    ((Ah ## 32) ? Ah ## 32 [k] :        \
    ((Ah ## 64) ? Ah ## 64 [k] :        \
    (k)))

#ifndef GB_JIT_KERNEL

    //--------------------------------------------------------------------------
    // for mainline, Factory, and generic kernels
    //--------------------------------------------------------------------------

    // GB_IGET: get I [k] for a 32/64-bit integer array I
    #define GB_IGET(I,k) (I ## 32 ? I ## 32 [k] : I ## 64 [k])

    // GB_ISET: set I [k] for a 32/64-bit integer array I
    #define GB_ISET(I,k,i) \
        { if (I ## 64) { I ## 64 [k] = (i) ; } else { I ## 32 [k] = (i) ; } }

    // GB_IINC: increment I [k] for a 32/64-bit integer array I
    #define GB_IINC(I,k,i) \
        { if (I ## 64) { I ## 64 [k] += (i) ; } else { I ## 32 [k] += (i) ; } }

    // GB_IADDR: &(I [k]) for a 32/64-bit integer array I
    #define GB_IADDR(I,k) (I ## 32 ?   \
        ((void *) (I ## 32 + k)) :  \
        ((void *) (I ## 64 + k)))

    // helper macro: declare a 32/64-bit integer array I
    #define GB_IDECL(I,const,u)                         \
        const u ## int32_t *restrict I ## 32 = NULL ;   \
        const u ## int64_t *restrict I ## 64 = NULL

    // helper macro: get a 32/64-bit pointer from a matrix
    #define GB_GET_MATRIX_PTR(I,A,is_32,component)      \
        I = (A) ? A->component : NULL ;                 \
        I ## 32 = (A) ? (A->is_32 ? I : NULL) : NULL ;  \
        I ## 64 = (A) ? (A->is_32 ? NULL : I) : NULL

    // helper macro: get a 32/64-bit pointer from a matrix hyper_hash.  The
    // integer types of A->Y->[pix] are defined by A->j_is_32.
    #define GB_GET_HYPER_PTR(I,A,pix)                                    \
        I = (A && A->Y) ? A->Y->pix : NULL ;                             \
        I ## 32 = (A && A->Y) ? (A->j_is_32 ? A->Y->pix : NULL) : NULL ; \
        I ## 64 = (A && A->Y) ? (A->j_is_32 ? NULL : A->Y->pix) : NULL

    // helper macros: get 32/64-bit pointers from a matrix Pending object.  The
    // integer types of A->Pending->[ij] are defined by A->i_is_32 and
    // A->j_is_32, respectively.  A->Pending must be non-NULL.
    #define GB_GET_PENDINGi_PTR(I,A)                    \
        I = A->Pending->i ;                             \
        I ## 32 = (A->i_is_32 ? A->Pending->i : NULL) ; \
        I ## 64 = (A->i_is_32 ? NULL : A->Pending->i)
    #define GB_GET_PENDINGj_PTR(I,A)                    \
        I = A->Pending->j ;                             \
        I ## 32 = (A->j_is_32 ? A->Pending->j : NULL) ; \
        I ## 64 = (A->j_is_32 ? NULL : A->Pending->j)

    // general method for getting an entry from the Ap array of a matrix
    #define GBp(Ap,k,vlen)                  \
        ((Ap ## 32) ? Ap ## 32 [k] :        \
        ((Ap ## 64) ? Ap ## 64 [k] :        \
        ((k) * (vlen))))

    // general method for getting an entry from the Ai array of a matrix
    #define GBi(Ai,p,vlen)                  \
        ((Ai ## 32) ? Ai ## 32 [p] :        \
        ((Ai ## 64) ? Ai ## 64 [p] :        \
        ((p) % (vlen))))

    // general method for getting an entry from the Ab array of a matrix
    #define GBb(Ab,p) ((Ab) ? Ab [p] : 1)

    // for declaring pointers for specific matrices (C, M, A, B, S, R, Z):

        // C matrix:
        #define GB_Cp_DECLARE(Cp,const)    GB_MDECL (Cp, const, u)
        #define GB_Ch_DECLARE(Ch,const)    GB_MDECL (Ch, const, u)
        #define GB_Ci_DECLARE(Ci,const)    GB_MDECL (Ci, const,  )
        #define GB_Ci_DECLARE_U(Ci,const)  GB_MDECL (Ci, const, u)
        #define GB_CPendingi_DECLARE(Pending_i) GB_MDECL (Pending_i, , u)
        #define GB_CPendingj_DECLARE(Pending_j) GB_MDECL (Pending_j, , u)

        // M matrix:
        #define GB_Mp_DECLARE(Mp,const)    GB_MDECL (Mp, const, u)
        #define GB_Mh_DECLARE(Mh,const)    GB_MDECL (Mh, const, u)
        #define GB_Mi_DECLARE(Mi,const)    GB_MDECL (Mi, const,  )
        #define GB_Mi_DECLARE_U(Mi,const)  GB_MDECL (Mi, const, u)

        // A matrix:
        #define GB_Ap_DECLARE(Ap,const)    GB_MDECL (Ap, const, u)
        #define GB_Ah_DECLARE(Ah,const)    GB_MDECL (Ah, const, u)
        #define GB_Ai_DECLARE(Ai,const)    GB_MDECL (Ai, const,  )
        #define GB_Ai_DECLARE_U(Ai,const)  GB_MDECL (Ai, const, u)

        // B matrix:
        #define GB_Bp_DECLARE(Bp,const)    GB_MDECL (Bp, const, u)
        #define GB_Bh_DECLARE(Bh,const)    GB_MDECL (Bh, const, u)
        #define GB_Bi_DECLARE(Bi,const)    GB_MDECL (Bi, const,  )
        #define GB_Bi_DECLARE_U(Bi,const)  GB_MDECL (Bi, const, u)

        // S matrix:
        #define GB_Sp_DECLARE(Sp,const)    GB_MDECL (Sp, const, u)
        #define GB_Sh_DECLARE(Sh,const)    GB_MDECL (Sh, const, u)
        #define GB_Si_DECLARE(Si,const)    GB_MDECL (Si, const,  )
        #define GB_Si_DECLARE_U(Si,const)  GB_MDECL (Si, const, u)

        // R matrix:
        #define GB_Rp_DECLARE(Rp,const)    GB_MDECL (Rp, const, u)
        #define GB_Rh_DECLARE(Rh,const)    GB_MDECL (Rh, const, u)
        #define GB_Ri_DECLARE(Ri,const)    GB_MDECL (Ri, const,  )
        #define GB_Ri_DECLARE_U(Ri,const)  GB_MDECL (Ri, const, u)

        // Z matrix:
        #define GB_Zp_DECLARE(Zp,const)    GB_MDECL (Zp, const, u)
        #define GB_Zh_DECLARE(Zh,const)    GB_MDECL (Zh, const, u)
        #define GB_Zi_DECLARE(Zi,const)    GB_MDECL (Zi, const,  )
        #define GB_Zi_DECLARE_U(Zi,const)  GB_MDECL (Zi, const, u)

    // for getting pointers from specific matrices:

        // C matrix:
        #define GB_Cp_PTR(Cp,C)    GB_GET_MATRIX_PTR (Cp, C, p_is_32, p)
        #define GB_Ch_PTR(Ch,C)    GB_GET_MATRIX_PTR (Ch, C, j_is_32, h)
        #define GB_Ci_PTR(Ci,C)    GB_GET_MATRIX_PTR (Ci, C, i_is_32, i)
        #define GB_CPendingi_PTR(Pending_i,C) GB_GET_PENDINGi_PTR (Pending_i, C)
        #define GB_CPendingj_PTR(Pending_j,C) GB_GET_PENDINGj_PTR (Pending_j, C)

        // M matrix:
        #define GB_Mp_PTR(Mp,M)    GB_GET_MATRIX_PTR (Mp, M, p_is_32, p)
        #define GB_Mh_PTR(Mh,M)    GB_GET_MATRIX_PTR (Mh, M, j_is_32, h)
        #define GB_Mi_PTR(Mi,M)    GB_GET_MATRIX_PTR (Mi, M, i_is_32, i)

        // A matrix:
        #define GB_Ap_PTR(Ap,A)    GB_GET_MATRIX_PTR (Ap, A, p_is_32, p)
        #define GB_Ah_PTR(Ah,A)    GB_GET_MATRIX_PTR (Ah, A, j_is_32, h)
        #define GB_Ai_PTR(Ai,A)    GB_GET_MATRIX_PTR (Ai, A, i_is_32, i)

        // B matrix:
        #define GB_Bp_PTR(Bp,B)    GB_GET_MATRIX_PTR (Bp, B, p_is_32, p)
        #define GB_Bh_PTR(Bh,B)    GB_GET_MATRIX_PTR (Bh, B, j_is_32, h)
        #define GB_Bi_PTR(Bi,B)    GB_GET_MATRIX_PTR (Bi, B, i_is_32, i)

        // S matrix:
        #define GB_Sp_PTR(Sp,S)    GB_GET_MATRIX_PTR (Sp, S, p_is_32, p)
        #define GB_Sh_PTR(Sh,S)    GB_GET_MATRIX_PTR (Sh, S, j_is_32, h)
        #define GB_Si_PTR(Si,S)    GB_GET_MATRIX_PTR (Si, S, i_is_32, i)

        // R matrix:
        #define GB_Rp_PTR(Rp,R)    GB_GET_MATRIX_PTR (Rp, R, p_is_32, p)
        #define GB_Rh_PTR(Rh,R)    GB_GET_MATRIX_PTR (Rh, R, j_is_32, h)
        #define GB_Ri_PTR(Ri,R)    GB_GET_MATRIX_PTR (Ri, R, i_is_32, i)

        // Z matrix:
        #define GB_Zp_PTR(Zp,Z)    GB_GET_MATRIX_PTR (Zp, Z, p_is_32, p)
        #define GB_Zh_PTR(Zh,Z)    GB_GET_MATRIX_PTR (Zh, Z, j_is_32, h)
        #define GB_Zi_PTR(Zi,Z)    GB_GET_MATRIX_PTR (Zi, Z, i_is_32, i)

    // for getting entries from Ap, Ah, Ai for specific matrices:

        // C matrix:
        #define GBp_C(Cp,k,vlen) GBp (Cp, k, vlen)
        #define GBh_C(Ch,k)      GBh (Ch, k)
        #define GBi_C(Ci,p,vlen) GBi (Ci, p, vlen)
        #define GBb_C(Cb,p)      GBb (Cb, p)
        #define GB_C_NVALS(e)    int64_t e = GB_nnz (C)
        #define GB_C_NHELD(e)    int64_t e = GB_nnz_held (C)

        // M matrix:
        #define GBp_M(Mp,k,vlen) GBp (Mp, k, vlen)
        #define GBh_M(Mh,k)      GBh (Mh, k)
        #define GBi_M(Mi,p,vlen) GBi (Mi, p, vlen)
        #define GBb_M(Mb,p)      GBb (Mb, p)
        #define GB_M_NVALS(e)    int64_t e = GB_nnz (M)
        #define GB_M_NHELD(e)    int64_t e = GB_nnz_held (M)

        // A matrix:
        #define GBp_A(Ap,k,vlen) GBp (Ap, k, vlen)
        #define GBh_A(Ah,k)      GBh (Ah, k)
        #define GBi_A(Ai,p,vlen) GBi (Ai, p, vlen)
        #define GBb_A(Ab,p)      GBb (Ab, p)
        #define GB_A_NVALS(e)    int64_t e = GB_nnz (A)
        #define GB_A_NHELD(e)    int64_t e = GB_nnz_held (A)

        // B matrix:
        #define GBp_B(Bp,k,vlen) GBp (Bp, k, vlen)
        #define GBh_B(Bh,k)      GBh (Bh, k)
        #define GBi_B(Bi,p,vlen) GBi (Bi, p, vlen)
        #define GBb_B(Bb,p)      GBb (Bb, p)
        #define GB_B_NVALS(e)    int64_t e = GB_nnz (B)
        #define GB_B_NHELD(e)    int64_t e = GB_nnz_held (B)

        // S matrix:
        #define GBp_S(Sp,k,vlen) GBp (Sp, k, vlen)
        #define GBh_S(Sh,k)      GBh (Sh, k)
        #define GBi_S(Si,p,vlen) GBi (Si, p, vlen)
        #define GBb_S(Sb,p)      GBb (Sb, p)
        #define GB_S_NVALS(e)    int64_t e = GB_nnz (S)
        #define GB_S_NHELD(e)    int64_t e = GB_nnz_held (S)

        // R matrix:
        #define GBp_R(Rp,k,vlen) GBp (Rp, k, vlen)
        #define GBh_R(Rh,k)      GBh (Rh, k)
        #define GBi_R(Ri,p,vlen) GBi (Ri, p, vlen)
        #define GBb_R(Rb,p)      GBb (Rb, p)
        #define GB_R_NVALS(e)    int64_t e = GB_nnz (R)
        #define GB_R_NHELD(e)    int64_t e = GB_nnz_held (R)

        // Z matrix:
        #define GBp_Z(Zp,k,vlen) GBp (Zp, k, vlen)
        #define GBh_Z(Zh,k)      GBh (Zh, k)
        #define GBi_Z(Zi,p,vlen) GBi (Zi, p, vlen)
        #define GBb_Z(Zb,p)      GBb (Zb, p)
        #define GB_Z_NVALS(e)    int64_t e = GB_nnz (Z)
        #define GB_Z_NHELD(e)    int64_t e = GB_nnz_held (Z)

#else

    //--------------------------------------------------------------------------
    // for JIT and PreJIT kernels
    //--------------------------------------------------------------------------

    // The JIT kernels only need to define GB_Ap_BITS, GB_Aj_BITS, and
    // GB_Ai_BITS for each matrix, as 32 or 64.

    // GB_IGET: get I [k] for a 32/64-bit integer array I
    #define GB_IGET(I,k) I [k]

    // GB_ISET: set I [k] for a 32/64-bit integer array I
    #define GB_ISET(I,k,i) I [k] = (i)

    // GB_IINC: increment I [k] for a 32/64-bit integer array I
    #define GB_IINC(I,k,i) I [k] += (i)

    // JIT helper macro
    #ifdef GB_CUDA_KERNEL
        #define GB_JDECL(I,const,u,bits) \
            const GB_EVAL4 (u,int,bits,_t) *__restrict__ I = NULL
    #else
        #define GB_JDECL(I,const,u,bits) \
            const GB_EVAL4 (u,int,bits,_t) *restrict I = NULL
    #endif

    // helper macro: get a 32/64-bit pointer from a matrix
    #define GB_GET_MATRIX_PTR(I,A,component) \
        I = (A) ? (A->component) : NULL

    // helper macro: get a 32/64-bit pointer from a matrix hyper_hash.
    #define GB_GET_HYPER_PTR(I,A,component) \
        I = (A && A->Y) ? (A->Y->component) : NULL

    // for declaring pointers for specific matrices:

        // C matrix:
        #define GB_Cp_DECLARE(Cp,const)    GB_JDECL (Cp, const, u, GB_Cp_BITS)
        #define GB_Ch_DECLARE(Ch,const)    GB_JDECL (Ch, const, u, GB_Cj_BITS)
        #define GB_Ci_DECLARE(Ci,const)    GB_JDECL (Ci, const,  , GB_Ci_BITS)
        #define GB_Ci_DECLARE_U(Ci,const)  GB_JDECL (Ci, const, u, GB_Ci_BITS)
        #define GB_CPendingi_DECLARE(Pending_i) \
                GB_JDECL (Pending_i, , u, GB_Ci_BITS)
        #define GB_CPendingj_DECLARE(Pending_j) \
                GB_JDECL (Pending_j, , u, GB_Cj_BITS)
        #define GB_Cp_IS_32 (GB_Cp_BITS == 32)
        #define GB_Cj_IS_32 (GB_Cj_BITS == 32)
        #define GB_Ci_IS_32 (GB_Ci_BITS == 32)

        // M matrix:
        #define GB_Mp_DECLARE(Mp,const)    GB_JDECL (Mp, const, u, GB_Mp_BITS)
        #define GB_Mh_DECLARE(Mh,const)    GB_JDECL (Mh, const, u, GB_Mj_BITS)
        #define GB_Mi_DECLARE(Mi,const)    GB_JDECL (Mi, const,  , GB_Mi_BITS)
        #define GB_Mi_DECLARE_U(Mi,const)  GB_JDECL (Mi, const, u, GB_Mi_BITS)
        #define GB_Mp_IS_32 (GB_Mp_BITS == 32)
        #define GB_Mj_IS_32 (GB_Mj_BITS == 32)
        #define GB_Mi_IS_32 (GB_Mi_BITS == 32)

        // A matrix:
        #define GB_Ap_DECLARE(Ap,const)    GB_JDECL (Ap, const, u, GB_Ap_BITS)
        #define GB_Ah_DECLARE(Ah,const)    GB_JDECL (Ah, const, u, GB_Aj_BITS)
        #define GB_Ai_DECLARE(Ai,const)    GB_JDECL (Ai, const,  , GB_Ai_BITS)
        #define GB_Ai_DECLARE_U(Ai,const)  GB_JDECL (Ai, const, u, GB_Ai_BITS)
        #define GB_Ap_IS_32 (GB_Ap_BITS == 32)
        #define GB_Aj_IS_32 (GB_Aj_BITS == 32)
        #define GB_Ai_IS_32 (GB_Ai_BITS == 32)

        // B matrix:
        #define GB_Bp_DECLARE(Bp,const)    GB_JDECL (Bp, const, u, GB_Bp_BITS)
        #define GB_Bh_DECLARE(Bh,const)    GB_JDECL (Bh, const, u, GB_Bj_BITS)
        #define GB_Bi_DECLARE(Bi,const)    GB_JDECL (Bi, const,  , GB_Bi_BITS)
        #define GB_Bi_DECLARE_U(Bi,const)  GB_JDECL (Bi, const, u, GB_Bi_BITS)
        #define GB_Bp_IS_32 (GB_Bp_BITS == 32)
        #define GB_Bj_IS_32 (GB_Bj_BITS == 32)
        #define GB_Bi_IS_32 (GB_Bi_BITS == 32)

        // S matrix:
        #define GB_Sp_DECLARE(Sp,const)    GB_JDECL (Sp, const, u, GB_Sp_BITS)
        #define GB_Sh_DECLARE(Sh,const)    GB_JDECL (Sh, const, u, GB_Sj_BITS)
        #define GB_Si_DECLARE(Si,const)    GB_JDECL (Si, const,  , GB_Si_BITS)
        #define GB_Si_DECLARE_U(Si,const)  GB_JDECL (Si, const, u, GB_Si_BITS)
        #define GB_Sp_IS_32 (GB_Sp_BITS == 32)
        #define GB_Sj_IS_32 (GB_Sj_BITS == 32)
        #define GB_Si_IS_32 (GB_Si_BITS == 32)

        // R matrix:
        #define GB_Rp_DECLARE(Rp,const)    GB_JDECL (Rp, const, u, GB_Rp_BITS)
        #define GB_Rh_DECLARE(Rh,const)    GB_JDECL (Rh, const, u, GB_Rj_BITS)
        #define GB_Ri_DECLARE(Ri,const)    GB_JDECL (Ri, const,  , GB_Ri_BITS)
        #define GB_Ri_DECLARE_U(Ri,const)  GB_JDECL (Ri, const, u, GB_Ri_BITS)
        #define GB_Rp_IS_32 (GB_Rp_BITS == 32)
        #define GB_Rj_IS_32 (GB_Rj_BITS == 32)
        #define GB_Ri_IS_32 (GB_Ri_BITS == 32)

        // Z matrix:
        #define GB_Zp_DECLARE(Zp,const)    GB_JDECL (Zp, const, u, GB_Zp_BITS)
        #define GB_Zh_DECLARE(Zh,const)    GB_JDECL (Zh, const, u, GB_Zj_BITS)
        #define GB_Zi_DECLARE(Zi,const)    GB_JDECL (Zi, const,  , GB_Zi_BITS)
        #define GB_Zi_DECLARE_U(Zi,const)  GB_JDECL (Zi, const, u, GB_Zi_BITS)
        #define GB_Zp_IS_32 (GB_Zp_BITS == 32)
        #define GB_Zj_IS_32 (GB_Zj_BITS == 32)
        #define GB_Zi_IS_32 (GB_Zi_BITS == 32)

    // for getting pointers from specific matrices:

        // C matrix:
        #define GB_Cp_PTR(Cp,C)    GB_GET_MATRIX_PTR (Cp, C, p)
        #define GB_Ch_PTR(Ch,C)    GB_GET_MATRIX_PTR (Ch, C, h)
        #define GB_Ci_PTR(Ci,C)    GB_GET_MATRIX_PTR (Ci, C, i)
        #define GB_CPendingi_PTR(Pending_i,C) Pending_i = C->Pending->i
        #define GB_CPendingj_PTR(Pending_j,C) Pending_j = C->Pending->j

        // M matrix:
        #define GB_Mp_PTR(Mp,M)    GB_GET_MATRIX_PTR (Mp, M, p)
        #define GB_Mh_PTR(Mh,M)    GB_GET_MATRIX_PTR (Mh, M, h)
        #define GB_Mi_PTR(Mi,M)    GB_GET_MATRIX_PTR (Mi, M, i)

        // A matrix:
        #define GB_Ap_PTR(Ap,A)    GB_GET_MATRIX_PTR (Ap, A, p)
        #define GB_Ah_PTR(Ah,A)    GB_GET_MATRIX_PTR (Ah, A, h)
        #define GB_Ai_PTR(Ai,A)    GB_GET_MATRIX_PTR (Ai, A, i)

        // B matrix:
        #define GB_Bp_PTR(Bp,B)    GB_GET_MATRIX_PTR (Bp, B, p)
        #define GB_Bh_PTR(Bh,B)    GB_GET_MATRIX_PTR (Bh, B, h)
        #define GB_Bi_PTR(Bi,B)    GB_GET_MATRIX_PTR (Bi, B, i)

        // S matrix:
        #define GB_Sp_PTR(Sp,S)    GB_GET_MATRIX_PTR (Sp, S, p)
        #define GB_Sh_PTR(Sh,S)    GB_GET_MATRIX_PTR (Sh, S, h)
        #define GB_Si_PTR(Si,S)    GB_GET_MATRIX_PTR (Si, S, i)

        // R matrix:
        #define GB_Rp_PTR(Rp,R)    GB_GET_MATRIX_PTR (Rp, R, p)
        #define GB_Rh_PTR(Rh,R)    GB_GET_MATRIX_PTR (Rh, R, h)
        #define GB_Ri_PTR(Ri,R)    GB_GET_MATRIX_PTR (Ri, R, i)

        // Z matrix:
        #define GB_Zp_PTR(Zp,Z)    GB_GET_MATRIX_PTR (Zp, Z, p)
        #define GB_Zh_PTR(Zh,Z)    GB_GET_MATRIX_PTR (Zh, Z, h)
        #define GB_Zi_PTR(Zi,Z)    GB_GET_MATRIX_PTR (Zi, Z, i)

    // for getting entries from Ap, Ah, Ai for specific matrices:

        // These must be #define'd in each JIT kernel, via GB_macrofy_sparsity
        // and GB_macrofy_nvals.

#endif

#endif

