//------------------------------------------------------------------------------
// gb_interface.h: the SuiteSparse:GraphBLAS MATLAB/Octave interface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This interface depends heavily on internal details of the
// SuiteSparse:GraphBLAS library.  Thus, GB.h is #include'd (via GB_helper.h),
// not just GraphBLAS.h.

// The gb_wrapup function accesses GB_methods inside GraphBLAS.

#ifndef GB_INTERFACE_H
#define GB_INTERFACE_H

#define NHISTORICAL
#undef GRAPHBLAS_VANILLA
#include "GraphBLAS.h"
#include "GB_helper.h"
#include "mex.h"
#include <ctype.h>

//------------------------------------------------------------------------------
// error handling and test coverage
//------------------------------------------------------------------------------

#ifdef GBCOV
#define GBCOV_MAX 1000
extern int64_t gbcov [GBCOV_MAX] ;
extern int gbcov_max ;
void gbcov_get (void) ;
void gbcov_put (void) ;
#define GBCOV_PUT gbcov_put ( )
#else
#define GBCOV_PUT
#endif

static inline void gb_wrapup (void)
{
    GBCOV_PUT ;
    if (GB_Global_memtable_n ( ) != 0)
    { 
        printf ("GrB memory leak!\n") ;
        GB_Global_memtable_dump ( ) ;
        mexErrMsgIdAndTxt ("GrB:error", "memory leak") ;
    }
}

#define ERROR2(errmsg, arg)                                 \
{                                                           \
    GBCOV_PUT ;                                             \
    mexErrMsgIdAndTxt ("GrB:error", errmsg, arg) ;          \
}

#define ERROR(errmsg)                                       \
{                                                           \
    GBCOV_PUT ;                                             \
    mexErrMsgIdAndTxt ("GrB:error", errmsg) ;               \
}

#define CHECK_ERROR(error,errmsg) if (error) ERROR (errmsg) ;

#define OK(method)                                          \
{                                                           \
    GrB_Info this_info = method ;                           \
    if (this_info != GrB_SUCCESS)                           \
    {                                                       \
        ERROR (gb_error_string (this_info)) ;               \
    }                                                       \
}

#define OK0(method)                                                 \
{                                                                   \
    GrB_Info this_info = method ;                                   \
    if (!(this_info == GrB_SUCCESS || this_info == GrB_NO_VALUE))   \
    {                                                               \
        ERROR (gb_error_string (this_info)) ;                       \
    }                                                               \
}

#define OK1(C,method)                                               \
{                                                                   \
    GrB_Info this_info = method ;                                   \
    if (this_info != GrB_SUCCESS)                                   \
    {                                                               \
        const char *err1 = gb_error_string (this_info) ;            \
        printf ("%s\n", err1) ;                                     \
        const char *err2 ;                                          \
        GrB_Matrix_error (&err2, C) ;                               \
        ERROR ((err2 == NULL || err2 [0] == '\0') ? err1 : err2) ;  \
    }                                                               \
}

//------------------------------------------------------------------------------
// basic macros
//------------------------------------------------------------------------------

// MATCH(s,t) compares two strings and returns true if equal
#define MATCH(s,t) (strcmp(s,t) == 0)

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(x)   (((x) >= 0) ? (x) : (-(x)))

// largest integer representable as a double
#define FLINTMAX (((int64_t) 1) << 53)

//------------------------------------------------------------------------------
// typedefs
//------------------------------------------------------------------------------

typedef enum            // output of GrB.methods
{
    KIND_GRB = 0,       // return a struct containing a GrB_Matrix
    KIND_SPARSE = 1,    // return a built-in sparse matrix
    KIND_FULL = 2,      // return a built-in full matrix
    KIND_BUILTIN = 3    // return a built-in sparse or full matrix (full if all
                        // entries present, sparse otherwise)
}
kind_enum_t ;

// [I,J,X] = GrB.extracttuples (A, desc) can return I and J in three ways:
//
//      one-based double:   just like [I,J,X] = find (A)
//      one-based int64:    I and J are one-based, as built-in but int64.
//      zero-based int64:   I and J are zero-based, and int64.  This is meant
//                          for internal use in GrB methods, but it is also
//                          the
//
// The descriptor is also used for GrB.build, GrB.extract, GrB.assign, and
// GrB.subassign.  In that case, the type is determined by the input arrays I
// and J.
//
// desc.base can be one of several strings:
//
//      'default'           the default is used (one-based int)
//      'zero-based'        zero-based uint32/uint64
//      'zero-based int'    zero-based uint32/uint64
//      'one-based'         one-based uint32/uint64
//      'one-based int'     one-based uint32/uint64
//      'one-based double'  the type is double, and one-based
//      'double'            the type is double, and one-based
//
// Note that there is no option for zero-based double.

typedef enum            // type of indices
{
    BASE_DEFAULT = 0,   // one-based integers (int32/int64)
    BASE_0_INT = 1,     // indices are returned as zero-based int32/int64
    BASE_1_INT = 2,     // indices are returned as one-based int32/int64
    BASE_1_DOUBLE = 3   // one-based double, unless the dimensions are too big
                        // for a flint (max(size(A)) > flintmax).  In that
                        // case, BASE_1_INT is used.
}
base_enum_t ;

//------------------------------------------------------------------------------
// gb_double_to_integer: convert a double to int64_t and check conversion
//------------------------------------------------------------------------------

static inline int64_t gb_double_to_integer (double x)
{
    int64_t i = (int64_t) x ;
    CHECK_ERROR (x != (double) i, "index must be integer") ;
    return (i) ;
}

//------------------------------------------------------------------------------
// function prototypes
//------------------------------------------------------------------------------

GrB_Type gb_mxarray_type        // return the GrB_Type of a built-in matrix
(
    const mxArray *X
) ;

GrB_Type gb_mxstring_to_type    // return the GrB_Type from a built-in string
(
    const mxArray *S        // built-in mxArray containing a string
) ;

GrB_Type gb_code_to_type    // return the GrB_Type from a GrB_Type_Code
(
    GrB_Type_Code code
) ;

GrB_Type gb_binaryop_ztype  // return the GrB_Type of the output of a binary op
(
    GrB_BinaryOp op
) ;

GrB_Type gb_monoid_type     // return the GrB_Type of a monoid
(
    GrB_Monoid op
) ;

void gb_mxstring_to_string  // copy a built-in string into a C string
(
    char *string,           // size at least maxlen+1
    const size_t maxlen,    // length of string
    const mxArray *S,       // built-in mxArray containing a string
    const char *name        // name of the mxArray
) ;

GrB_Matrix gb_get_shallow   // return a shallow copy of built-in sparse matrix
(
    const mxArray *X
) ;

GrB_Matrix gb_get_deep      // return a deep GrB_Matrix copy of a built-in X
(
    const mxArray *X        // input built-in matrix (sparse or struct)
) ;

mxArray * gb_type_to_mxstring    // return the built-in string from a GrB_Type
(
    const GrB_Type type
) ;

GrB_Matrix gb_typecast  // C = (type) A, where C is deep
(
    GrB_Matrix A,       // may be shallow
    GrB_Type type,      // if NULL, copy but do not typecast
    int fmt,            // format of C
    int sparsity        // sparsity control for C, if 0 use A
) ;

GrB_Matrix gb_new       // create and empty matrix C
(
    GrB_Type type,      // type of C
    GrB_Index nrows,    // # of rows
    GrB_Index ncols,    // # of rows
    int fmt,            // requested format
    int sparsity        // sparsity control for C, 0 for default
) ;

void gb_abort ( void ) ;    // failure

int gb_flush ( void ) ;     // flush mexPrintf output to Command Window

void gb_usage       // check usage and make sure GxB_init has been called
(
    bool ok,                // if false, then usage is not correct
    const char *message     // error message if usage is not correct
) ;

const char *gb_error_string     // return an error string from a GrB_Info value
(
    GrB_Info info
) ;

void gb_find_dot            // find 1st and 2nd dot ('.') in a string
(
    int32_t position [2],   // positions of one or two dots
    const char *s           // null-terminated string to search
) ;

GrB_Type gb_string_to_type      // return the GrB_Type from a string
(
    const char *classname
) ;

GrB_UnaryOp gb_mxstring_to_unop         // return unary operator from a string
(
    const mxArray *mxstring,            // built-in string
    const GrB_Type default_type         // default type if not in the string
) ;

GrB_UnaryOp gb_string_to_unop           // return unary operator from a string
(
    char *opstring,                     // string defining the operator
    const GrB_Type default_type         // default type if not in the string
) ;

GrB_UnaryOp gb_string_and_type_to_unop  // return op from string and type
(
    const char *op_name,        // name of the operator, as a string
    const GrB_Type type,        // type of the x,y inputs to the operator
    const bool type_not_given   // true if no type present in the string
) ;

GrB_BinaryOp gb_mxstring_to_binop       // return binary operator from a string
(
    const mxArray *mxstring,            // built-in string
    const GrB_Type atype,               // type of A
    const GrB_Type btype                // type of B
) ;

void gb_mxstring_to_binop_or_idxunop    // binop or idxunop from a string
(
    const mxArray *mxstring,            // built-in string
    const GrB_Type atype,               // type of A
    const GrB_Type btype,               // type of B
    // output:
    GrB_BinaryOp *op2,                  // binary op
    GrB_IndexUnaryOp *idxunop,          // idxunop
    int64_t *ithunk                     // thunk for idxunop
) ;

GrB_BinaryOp gb_string_to_binop_or_idxunop
(
    char *opstring,                     // string defining the operator
    const GrB_Type atype,               // type of A
    const GrB_Type btype,               // type of B
    GrB_IndexUnaryOp *idxunop,          // idxunop from the string
    int64_t *ithunk                     // thunk for idxunop
) ;

GrB_BinaryOp gb_string_and_type_to_binop_or_idxunop
(
    const char *op_name,        // name of the operator, as a string
    const GrB_Type type,        // type of the x,y inputs to the operator
    const bool type_not_given,  // true if no type present in the string
    GrB_IndexUnaryOp *idxunop,          // idxunop from the string
    int64_t *ithunk                     // thunk for idxunop
) ;

GrB_Semiring gb_mxstring_to_semiring    // return semiring from a string
(
    const mxArray *mxstring,            // built-in string
    const GrB_Type atype,               // type of A
    const GrB_Type btype                // type of B
) ;

GrB_Semiring gb_string_to_semiring      // return a semiring from a string
(
    char *semiring_string,              // string defining the semiring
    const GrB_Type atype,               // type of A
    const GrB_Type btype                // type of B
) ;

GrB_Semiring gb_semiring            // built-in semiring, or NULL if error
(
    const GrB_BinaryOp add,         // add operator
    const GrB_BinaryOp mult         // multiply operator
) ;

GrB_Descriptor gb_mxarray_to_descriptor // new descriptor, or NULL if none
(
    const mxArray *desc_builtin,// built-in struct with possible descriptor
    kind_enum_t *kind,          // GrB, sparse, or full
    int *fmt,                   // by row or by col
    int *sparsity,              // hypersparse/sparse/bitmap/full
    base_enum_t *base           // 0-based int, 1-based int, or 1-based double
) ;

GrB_Matrix gb_expand_to_full    // C = full (A), and typecast
(
    const GrB_Matrix A,         // input matrix to expand to full
    GrB_Type type,              // type of C, if NULL use the type of A
    int fmt,                    // format of C
    GrB_Matrix id               // identity value, use zero if NULL
) ;

mxArray *gb_export_to_mxstruct  // return exported built-in struct G
(
    GrB_Matrix *A_handle        // matrix to export; freed on output
) ;

mxArray *gb_export_to_mxsparse  // return exported built-in sparse matrix S
(
    GrB_Matrix *A_handle        // matrix to export; freed on output
) ;

mxArray *gb_export_to_mxfull    // return exported built-in full matrix F
(
    void **X_handle,            // pointer to array to export
    const GrB_Index nrows,      // dimensions of F
    const GrB_Index ncols,
    GrB_Type type               // type of the array
) ;

mxArray *gb_export              // return the exported built-in matrix or struct
(
    GrB_Matrix *C_handle,       // GrB_Matrix to export and free
    kind_enum_t kind            // GrB, sparse, or full
) ;

void gb_string_to_idxunop
(
    // outputs: one of the outputs is non-NULL and the other NULL
    GrB_IndexUnaryOp *op,       // GrB_IndexUnaryOp, if found
    bool *thunk_zero,           // true if op requires a thunk zero
    bool *op_is_positional,     // true if op is positional
    // input/output:
    int64_t *ithunk,
    // inputs:
    char *opstring,             // string defining the operator
    const GrB_Type atype        // type of A, or NULL if not present
) ;

void gb_mxstring_to_idxunop
(
    // outputs: one of the outputs is non-NULL and the other NULL
    GrB_IndexUnaryOp *op,       // GrB_IndexUnaryOp, if found
    bool *thunk_zero,           // true if op requires a thunk zero
    bool *op_is_positional,     // true if op is positional
    // input/output:
    int64_t *ithunk,
    // inputs:
    const mxArray *mxstring,    // built-in string
    const GrB_Type atype        // type of A, or NULL if not present
) ;

bool gb_mxarray_is_scalar   // true if built-in array is a scalar
(
    const mxArray *S
) ;

uint64_t gb_mxget_uint64_scalar // return uint64 value of a MATLAB scalar
(
    const mxArray *mxscalar,    // MATLAB scalar to extract
    char *name                  // name of the scalar
) ;

bool gb_mxarray_is_empty    // true if built-in array is NULL, or 2D and 0-by-0
(
    const mxArray *S
) ;

void gb_mxfree              // mxFree wrapper
(
    void **p_handle         // handle to pointer to be freed
) ;

GrB_Vector gb_mxarray_to_list   // list of indices or values
(
    const mxArray *X,       // MATLAB input matrix or struct with GrB content
    const int base_offset   // 1 or 0
) ;

GrB_Vector gb_mxcell_to_list    // return index list I
(
    // input
    const mxArray *Cell,        // built-in MATLAB cell array
    const int base_offset,      // 1 or 0
    const uint64_t n,           // dimension of the matrix
    // output
    uint64_t *nI,               // # of items in the list
    int64_t *I_max              // largest item in the list
) ;

GrB_BinaryOp gb_first_binop         // return GrB_FIRST_[type] operator
(
    const GrB_Type type
) ;

GrB_Monoid gb_binop_to_monoid           // return monoid from a binary op
(
    GrB_BinaryOp op
) ;

GrB_Monoid gb_string_to_monoid          // return monoid from a string
(
    char *opstring,                     // string defining the operator
    const GrB_Type type                 // default type if not in the string
) ;

GrB_Monoid gb_mxstring_to_monoid        // return monoid from a string
(
    const mxArray *mxstring,            // built-in string
    const GrB_Type type                 // default type if not in the string
) ;

bool gb_mxstring_to_format      // true if a valid format is found
(
    // input
    const mxArray *mxformat,    // built-in string, 'by row' or 'by col'
    // output
    int *fmt,
    int *sparsity
) ;

void gb_assign                  // gbassign or gbsubassign mexFunctions
(
    int nargout,                // # output arguments for mexFunction
    mxArray *pargout [ ],       // output arguments for mexFunction
    int nargin,                 // # input arguments for mexFunction
    const mxArray *pargin [ ],  // input arguments for mexFunction
    bool do_subassign,          // true: do subassign, false: do assign
    const char *usage           // usage string to print if error
) ;

GrB_Matrix gb_by_col            // return the matrix by column
(
    GrB_Matrix *A_copy_handle,  // copy made of A, stored by column, or NULL
    GrB_Matrix A_input          // input matrix, by row or column
) ;

int gb_default_format       // GxB_BY_ROW or GxB_BY_COL
(
    GrB_Index nrows,        // row vectors are stored by row
    GrB_Index ncols         // column vectors are stored by column
) ;

bool gb_is_vector               // true if A is a row or column vector
(
    GrB_Matrix A                // GrB_Matrix to query
) ;

bool gb_is_column_vector        // true if A is a column vector
(
    GrB_Matrix A                // GrB_matrix to query
) ;

int gb_get_format           // GxB_BY_ROW or GxB_BY_COL
(
    GrB_Index cnrows,       // C is cnrows-by-cncols
    GrB_Index cncols,
    GrB_Matrix A,           // may be NULL
    GrB_Matrix B,           // may be NULL
    int fmt_descriptor      // may be GxB_NO_FORMAT
) ;

int gb_get_sparsity         // 1 to 15
(
    GrB_Matrix A,                       // may be NULL
    GrB_Matrix B,                       // may be NULL
    int sparsity_default                // may be 0
) ;

bool gb_is_equal            // true if A == B, false if A ~= B
(
    GrB_Matrix A,
    GrB_Matrix B
) ;

bool gb_is_all              // true if op (A,B) is all true, false otherwise
(
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp op
) ;

void gb_isnan32 (bool *z, const float *aij,
                 int64_t i, int64_t j, const void *thunk) ;
void gb_isnan64 (bool *z, const double *aij,
                 int64_t i, int64_t j, const void *thunk) ;
void gb_isnotnan32 (bool *z, const float *aij,
                    int64_t i, int64_t j, const void *thunk) ;
void gb_isnotnan64 (bool *z, const double *aij,
                    int64_t i, int64_t j, const void *thunk) ;
void gb_isnanfc32 (bool *z, const GxB_FC32_t *x,
                   int64_t i, int64_t j, const void *thunk) ;
void gb_isnanfc64 (bool *z, const GxB_FC64_t *aij,
                   int64_t i, int64_t j, const void *thunk) ;
void gb_isnotnanfc32 (bool *z, const GxB_FC32_t *aij,
                      int64_t i, int64_t j, const void *thunk) ;
void gb_isnotnanfc64 (bool *z, const GxB_FC64_t *aij,
                      int64_t i, int64_t j, const void *thunk) ;

void gb_get_mxargs
(
    // input:
    int nargin,                 // # input arguments for mexFunction
    const mxArray *pargin [ ],  // input arguments for mexFunction
    const char *usage,          // usage to print, if too many args appear
    // output:
    mxArray *Matrix [4],        // matrix arguments
    int *nmatrices,             // # of matrix arguments
    mxArray *String [2],        // string arguments
    int *nstrings,              // # of string arguments
    mxArray *Cell [2],          // cell array arguments
    int *ncells,                // # of cell array arguments
    GrB_Descriptor *desc,       // last argument is always the descriptor
    base_enum_t *base,          // desc.base
    kind_enum_t *kind,          // desc.kind
    int *fmt,                   // desc.format : by row or by col
    int *sparsity               // desc.format : hypersparse/sparse/bitmap/full
) ;

int64_t gb_norm_kind (const mxArray *arg) ;

double gb_norm              // compute norm (A,kind)
(
    GrB_Matrix A,
    int64_t norm_kind       // 0, 1, 2, INT64_MAX, or INT64_MIN
) ;

GrB_Type gb_default_type        // return the default type to use
(
    const GrB_Type atype,       // type of the A matrix
    const GrB_Type btype        // type of the B matrix
) ;

bool gb_is_integer (const GrB_Type type) ;

bool gb_is_float (const GrB_Type type) ;

bool gb_is_dense                // true if A is dense
(
    GrB_Matrix A                // GrB_Matrix to query
) ;

bool gb_is_readonly             // true if A has any readonly components
(
    GrB_Matrix A                // GrB_matrix to query
) ;

GrB_UnaryOp gb_round_op (const GrB_Type type) ;

mxArray *gb_mxclass_to_mxstring (mxClassID class, bool is_complex) ;

void gb_defaults (void) ;   // set global GraphBLAS defaults for MATLAB

void gb_at_exit ( void ) ;  // call GrB_finalize

//------------------------------------------------------------------------------
// GraphBLAS polymorphic methods
//------------------------------------------------------------------------------

// The @GrB MATLAB interface does not use these macros since they require a
// C11, and thus they cannot be used for MATLAB on Windows.

#undef GrB_Monoid_new
#undef GxB_Monoid_terminal_new
#undef GrB_Scalar_setElement
#undef GrB_Scalar_extractElement
#undef GrB_Vector_build
#undef GrB_Vector_setElement
#undef GrB_Vector_extractElement
#undef GrB_Vector_extractTuples
#undef GrB_Matrix_build
#undef GrB_Matrix_setElement
#undef GrB_Matrix_extractElement
#undef GrB_Matrix_extractTuples
#undef GrB_get
#undef GrB_set
#undef GrB_wait
#undef GrB_error
#undef GrB_eWiseMult
#undef GrB_eWiseAdd
#undef GxB_eWiseUnion
#undef GrB_extract
#undef GxB_subassign
#undef GrB_assign
#undef GrB_apply
#undef GrB_select
#undef GrB_reduce
#undef GrB_kronecker
#undef GxB_resize
#undef GxB_fprint
#undef GxB_print
#undef GrB_Matrix_import
#undef GrB_Matrix_export
#undef GxB_sort
#undef GrB_free
#undef GxB_Scalar_setElement
#undef GxB_Scalar_extractElement
#undef GxB_set
#undef GxB_get
#undef GxB_select

#endif

