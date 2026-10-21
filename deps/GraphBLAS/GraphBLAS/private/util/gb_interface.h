//------------------------------------------------------------------------------
// gb_interface.h: the SuiteSparse:GraphBLAS MATLAB/Octave interface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This interface depends heavily on internal details of the
// SuiteSparse:GraphBLAS library.  Thus, GB.h is #include'd (via GB_helper.h),
// not just GraphBLAS.h.

#undef GRAPHBLAS_VANILLA
#include "GraphBLAS.h"
#include "GB_helper.h"
#include "mex.h"
#include <ctype.h>

//------------------------------------------------------------------------------
// error handling and test coverage
//------------------------------------------------------------------------------

#ifdef GBCOV

    //--------------------------------------------------------------------------
    // test coverage only, not used in production
    //--------------------------------------------------------------------------

    #define GBCOV_MAX 1000
    extern int64_t gbcov [GBCOV_MAX] ;
    extern int gbcov_max ;
    void gbcov_get (void) ;
    void gbcov_put (void) ;
    static inline void gb_wrapup (void)
    {
        gbcov_put ( ) ;
    }

#else

    //--------------------------------------------------------------------------
    // no test coverage in production
    //--------------------------------------------------------------------------

    #define gbcov_get()
    #define gbcov_put()
    #define gb_wrapup()

#endif

//------------------------------------------------------------------------------
// basic error handling for gb_* utilities
//------------------------------------------------------------------------------

#define FREE_WORK
#define FREE_ALL

#define ERRLEN (GB_LOGGER_LEN+128)

// error handling for gb_* utilities
#define ERROR2(errmsg,arg,info)                             \
{                                                           \
    if (err [0] == '\0')                                    \
    {                                                       \
        snprintf (err, ERRLEN, errmsg, arg) ;               \
        err [ERRLEN-1] = '\0' ;                             \
    }                                                       \
    FREE_ALL ;                                              \
    return (info) ;                                         \
}

#define ERROR(errmsg,info)                                  \
{                                                           \
    if (err [0] == '\0')                                    \
    {                                                       \
        GB_string_copy (err, errmsg, ERRLEN) ;              \
        err [ERRLEN-1] = '\0' ;                             \
    }                                                       \
    FREE_ALL ;                                              \
    return (info) ;                                         \
}

#define CHECK_ERROR(error,errmsg)                           \
    if (error) ERROR (errmsg, GrB_INVALID_VALUE) ;

#define OK(method)                                          \
{                                                           \
    GrB_Info this_info = method ;                           \
    if (this_info != GrB_SUCCESS)                           \
    {                                                       \
        const char *errmsg = (err [0] != '\0') ? err :      \
            gb_error_string (this_info) ;                   \
        ERROR (errmsg, this_info) ;                         \
    }                                                       \
}

#define OK0(method)                                                 \
{                                                                   \
    GrB_Info this_info = method ;                                   \
    if (!(this_info == GrB_SUCCESS || this_info == GrB_NO_VALUE))   \
    {                                                               \
        const char *errmsg = (err [0] != '\0') ? err :              \
            gb_error_string (this_info) ;                           \
        ERROR (errmsg, this_info) ;                                 \
    }                                                               \
}

#define OK1(C,method)                                               \
{                                                                   \
    GrB_Info this_info = method ;                                   \
    if (this_info != GrB_SUCCESS)                                   \
    {                                                               \
        const char *err2 ;                                          \
        GrB_Matrix_error (&err2, C) ;                               \
        if (err2 != NULL && err2 [0] != '\0')                       \
        {                                                           \
            /* copy the err2 string into err since err2 is freed */ \
            /* when C is freed */                                   \
            GB_string_copy (err, err2, ERRLEN) ;                    \
            err [ERRLEN-1] = '\0' ;                                 \
            ERROR (err, this_info) ;                                \
        }                                                           \
        else                                                        \
        {                                                           \
            ERROR (gb_error_string (this_info), this_info) ;        \
        }                                                           \
    }                                                               \
}

#define CHECK_NULL(p)                                               \
{                                                                   \
    if ((p) == NULL)                                                \
    {                                                               \
        ERROR ("out of memory", GrB_OUT_OF_MEMORY) ;                \
    }                                                               \
}

// for test coverage only:
#define GOTCHA                                                      \
    mexErrMsgIdAndTxt ("GraphBLAS:gotcha", "gotcha! %s line %d",    \
        __FILE__, __LINE__) ;

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

// default maximum string length
#define LEN 256

//------------------------------------------------------------------------------
// typedefs
//------------------------------------------------------------------------------

typedef enum            // output of GrB.methods
{
    KIND_GRB = 0,       // return G.opaque containing a GrB_Matrix
    KIND_GHB = -1,      // same as KIND_GRB, except for display
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

// gb_descriptor_struct: a plain struct, so that it can be statically allocated
// in the mx* portion of a mexFunction, and filled with values from the MATLAB
// desc struct.

struct gb_descriptor_struct
{
    int nondefault ;    // 0: all GrB_Descriptor options are default;
                        // so the GrB_Descriptor can be NULL
    int is_present ;    // 1: MATLAB descriptor struct is present on input;
                        // 0: not present

    // these appear in the GraphBLAS GrB_Descriptor:
    int out ;           // output descriptor
    int mask ;          // mask descriptor
    int in0 ;           // first input descriptor (A for C=A*B, for example)
    int in1 ;           // second input descriptor (B for C=A*B)
    int axb ;           // for selecting the method for C=A*B

    // these are only in the gb_descriptor:
    kind_enum_t kind ;  // how to return the output
    int fmt ;           // by row or by col
    int sparsity ;      // hypersparse/sparse/bitmap/full
    base_enum_t base ;  // 0-based-int, 1-based int, or 1-based double

// these appear in the GraphBLAS descriptor but are not needed here:
//  int compression ;   // compression method for GxB_Matrix_serialize
//  int do_sort ;       // if nonzero, do the sort in GrB_mxm
//  int import ;        // if zero (default), trust input data
//  int row_list ;      // how to use the row index list, I
//  int col_list ;      // how to use the col index list, J
//  int val_list ;      // how to use the value list, X
} ;

typedef struct gb_descriptor_struct *gb_descriptor ;

// gb_matrix_struct: a plain struct that can be statically allocated, which
// holds either a GraphBLAS GrB or GhB matrix or the contents of a MATLAB
// sparse or full matrix.

struct gb_matrix_struct
{
    //--------------------------------------------------------------------------
    // content for all matrices, GraphBLAS or MATLAB
    //--------------------------------------------------------------------------

    uint64_t nvals ;    // # of entries (for a GraphBLAS matrix, includes
                        // zombies but excludes pending tuples). 
    GrB_Type type ;     // type of the MATLAB matrix, as a GrB_Type
    uint64_t nrows ;
    uint64_t ncols ;
    size_t typesize ;   // size of the data type

    //--------------------------------------------------------------------------

    // Only one of the two sections are present.  This struct is memset to all
    // zero, and then only one of the two sections are filled.  If the matrix
    // is a GhB GraphBLAS matrix, then G is non-NULL.  Otherwise, the matrix
    // is a built-in MATLAB sparse or full matrix, or a GrB value matrix.

        //----------------------------------------------------------------------
        // (1) GhB handle matrix; NULL if MATLAB or GrB matrix
        //----------------------------------------------------------------------

        GrB_Matrix G ;

        //----------------------------------------------------------------------
        // (2) GrB value matrix or MATLAB matrix: populated if G is non-NULL
        //----------------------------------------------------------------------

        // If the input is a 0-by-0 MATLAB matrix, the [p,i,x] content below is
        // NULL, and sparsity is GxB_FULL.

        void *p ;
        void *h ;
        void *b ;
        void *i ;
        void *x ;
        void *Yp ;
        void *Yi ;
        void *Yx ;
        int64_t plen ;      // p has size plen+1 and h has size plen
        int64_t nvec ;      // size of Yi and Yx and # entries in Y
        int64_t nvec_nonempty ;
        int64_t yncols ;
        int64_t ynrows ;
        int sparsity ;      // sparse/hyper/bitmap/full
        bool by_col ;       // true if held by column, false if by row
        bool p_is_32 ;      // type of p (32 bit or 64 bit)
        bool j_is_32 ;      // type of h, Yp, Yi, and Yx (32 bit or 64 bit)
        bool i_is_32 ;      // type of i (32 bit or 64 bit)
        bool iso ;          // true if iso-valued
        bool is_empty ;     // true for an empty MATLAB matrix

    //--------------------------------------------------------------------------
    // bool content for a GhB matrix; not needed for other 
    //--------------------------------------------------------------------------

    bool will_wait ;    // true if G has any pending work; always false for a
                        // MATLAB matrix or GrB matrix

    kind_enum_t kind ;  // for display only
} ;

typedef struct gb_matrix_struct *gb_matrix ;

//------------------------------------------------------------------------------
// function prototypes
//------------------------------------------------------------------------------

void gb_at_exit ( void ) ;  // call GrB_finalize

GrB_Info gb_binaryop_ztype
(
    // output
    GrB_Type *ztype,    // the GrB_Type of the output of a binary op
    // input
    GrB_BinaryOp op,
    char err [ERRLEN]
) ;

GrB_Info gb_binop_to_monoid         // return monoid from a binary op
(
    // output
    GrB_Monoid *monoid,
    // input
    GrB_BinaryOp op,
    char err [ERRLEN]
) ;

GrB_Info gb_by_col
(
    // output
    GrB_Matrix *A_handle,       // return the matrix by column
    GrB_Matrix *A_copy_handle,  // copy made of A, stored by column, or NULL
    // input
    GrB_Matrix A_input,         // input matrix, by row or column
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_cell_to_list
(
    // output
    GrB_Vector *I_handle,
    GrB_Vector *I_to_free_handle,
    uint64_t *nI,               // # of items in the list
    int64_t *I_max,             // largest item in the list (NULL if not needed)
    // input
    struct gb_matrix_struct Cell_Matrix [3],    // contents of the Cell
    const int len,              // # of items in Cell_Matrix
    const int base_offset,      // 1 or 0
    const uint64_t n,           // dimension of the matrix
    const int arena,
    char err [ERRLEN]
) ;

GrB_Type gb_code_to_type    // return the GrB_Type from a GrB_Type_Code
(
    GrB_Type_Code code
) ;

GrB_Info gb_default_format
(
    // output
    int *fmt,               // GxB_BY_ROW or GxB_BY_COL
    // input
    uint64_t nrows,        // row vectors are stored by row
    uint64_t ncols,        // column vectors are stored by column
    char err [ERRLEN]
) ;

GrB_Type gb_default_type        // return the default type to use
(
    const GrB_Type atype,       // type of the A matrix
    const GrB_Type btype        // type of the B matrix
) ;

GrB_Info gb_dup             // copy a matrix
(
    // output:
    GrB_Matrix *C_handle,   // copy of the input matrix
    // input:
    GrB_Matrix Cin,         // matrix to copy
    const int arena,
    char err [ERRLEN]
) ;

const char *gb_error_string     // return an error string from a GrB_Info value
(
    GrB_Info info
) ;

GrB_Info gb_expand_scalar_to_vector
(
    // output
    GrB_Vector *V,
    // input
    GrB_Vector W,
    GrB_Type type,
    uint64_t nvals,
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_expand_to_full      // C = full (A), and typecast
(
    // output
    GrB_Matrix *C_handle,
    // inputs
    const GrB_Matrix A,         // input matrix to expand to full
    GrB_Type type,              // type of C, if NULL use the type of A
    int fmt,                    // format of C
    GrB_Matrix id,              // identity value, use zero if NULL
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_export              // export a GrB_Matrix to MATLAB
(
    // output:
    GrB_Matrix *C_opaque,       // matrix for export as GhB;
                                // NULL if in-place
    // input/output:
    GrB_Matrix *C_handle,       // GrB_Matrix to export
    // input:
    kind_enum_t kind,           // GrB, sparse, full, or built-in
    const bool ghb,
    char err [ERRLEN]
) ;

GrB_Info gb_export_to_full
(
    GrB_Matrix *C_handle,   // GraphBLAS matrix to modify for export to MATLAB
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_export_to_sparse
(
    // input/output
    GrB_Matrix *C_handle,   // GraphBLAS matrix to modify for export to MATLAB
    // intput
    const int arena,
    char err [ERRLEN]
) ;

void gb_find_dot            // find 1st and 2nd dot ('.') in a string
(
    int32_t position [2],   // positions of one or two dots
    const char *s           // null-terminated string to search
) ;

GrB_Info gb_first_binop     // construct GrB_FIRST_[type] operator
(
    // output
    GrB_BinaryOp *op,       // return GrB_FIRST_[type] operator
    // input
    const GrB_Type type,
    char err [ERRLEN]
) ;

GrB_Info gb_get_deep        // get the input/output matrix C
(
    // output:
    GrB_Matrix *C_handle,   // matrix C: deep copy if in-place
    // input:
    bool inplace,           // if true, C is modified in-place (C is Cin)
    gb_matrix matrix,       // input MATLAB, GrB, or GhB matrix
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_get_descriptor
(
    // output:
    GrB_Descriptor *desc_handle,    // GraphBLAS descriptor
    // input:
    gb_descriptor gbdesc,           // gb_descriptor, pointer to static struct
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_get_descriptor_mxm
(
    // output:
    GrB_Descriptor *desc_handle,    // GraphBLAS descriptor
    // input:
    gb_descriptor gbdesc,           // gb_descriptor, pointer to static struct
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_get_first_scalar
(
    // output:
    GrB_Scalar *x,          // x = find (V, 'first')
    // input:
    GrB_Vector V,
    GrB_Type type,
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_get_format      // get the format (by row or by col)
(
    // input:
    GrB_Index cnrows,       // C is cnrows-by-cncols
    GrB_Index cncols,
    GrB_Matrix A,           // may be NULL
    GrB_Matrix B,           // may be NULL
    // input/output:
    int *fmt,               // may be GxB_NO_FORMAT on input
    char err [ERRLEN]
) ;

GrB_Info gb_get_matlab_or_grb_matrix   // shallow copy of MATLAB or GrB matrix
(
    // output
    GrB_Matrix *A_handle,   // content of A is tagged GxB_IS_READONLY
    // input
    gb_matrix matrix,       // contents of a MATLAB matrix
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_get_matrix
(
    // output
    GrB_Matrix *A_handle,   // output matrix
    GrB_Matrix *A_to_free,  // must be freed by the caller if not NULL
    // input
    gb_matrix X,            // input MATLAB, GrB, or GhB matrix
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_get_sparsity    // determine the sparsity of C for C = method(A,B)
(
    // input:
    GrB_Matrix A,           // may be NULL
    GrB_Matrix B,           // may be NULL
    // input/output:
    int *sparsity,          // may be 0 on input
    char err [ERRLEN]
) ;

GrB_Info gb_is_all          // check two matrices for equality, given an op
(
    // output:
    bool *result,           // true if op (A,B) is all true, false otherwise
    // input:
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp op,
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_is_column_vector    // determine if A is a column vector
(
    // output:
    bool *is_column_vector,
    // input:
    GrB_Matrix A,               // GrB_matrix to query
    char err [ERRLEN]
) ;

GrB_Info gb_is_dense            // determine if A is dense
(
    // output:
    bool *is_dense,
    // input:
    GrB_Matrix A,               // GrB_Matrix to query
    char err [ERRLEN]
) ;

GrB_Info gb_is_equal
(
    // output:
    bool *is_equal,             // true if A == B, false if A ~= B
    // input:
    GrB_Matrix A,
    GrB_Matrix B,
    const int arena,
    char err [ERRLEN]
) ;

bool gb_is_float (const GrB_Type type) ;

bool gb_is_integer (const GrB_Type type) ;

GrB_Info gb_is_scalar
(
    // output:
    bool *is_scalar,    // true if A is a 1-by-1 GrB_Matrix with 1 entry
    // input
    GrB_Matrix A,
    char err [ERRLEN]
) ;

GrB_Info gb_is_vector
(
    bool *is_vector,            // true if A is a row or column vector
    GrB_Matrix A,               // GrB_Matrix to query
    char err [ERRLEN]
) ;

GrB_Info gb_matrix_to_list
(
    // outputs:
    GrB_Vector *V_handle,   // list of indices or values; caller must not free
    GrB_Vector *V_to_free_handle,  // must be freed by the caller
    // inputs:
    gb_matrix matrix,
    const int base_offset,  // 1 or 0
    const int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_monoid_type
(
    // output:
    GrB_Type *type,
    // input:
    GrB_Monoid op,
    char err [ERRLEN]
) ;

GrB_Info gb_new       // create and empty matrix C
(
    // output
    GrB_Matrix *C_handle,
    // input
    GrB_Type type,      // type of C
    uint64_t nrows,     // # of rows
    uint64_t ncols,     // # of rows
    int fmt,            // requested format, if < 0 use default
    int sparsity,       // sparsity control for C, 0 for default
    int arena,
    char err [ERRLEN]
) ;

GrB_Info gb_norm            // compute norm (A,kind)
(
    // output:
    double *s,              // norm of A
    // inputs:
    GrB_Matrix A,
    int64_t norm_kind,      // 0, 1, 2, INT64_MAX, or INT64_MIN
    const int arena,
    char err [ERRLEN]
) ;

GrB_UnaryOp gb_round_op (const GrB_Type type) ;

GrB_Info gb_semiring                // find semiring from (add,mult) ops
(
    // output:
    GrB_Semiring *semiring,
    // inputs:
    const GrB_BinaryOp add,         // add operator
    const GrB_BinaryOp mult,        // multiply operator
    char err [ERRLEN]
) ;

GrB_Info gb_string_and_type_to_binop_or_idxunop
(
    // output:
    GrB_BinaryOp *binop,        // binary op, or NULL if idxunop
    GrB_IndexUnaryOp *idxunop,          // idxunop from the string
    // input/output:
    int64_t *ithunk,                    // thunk for idxunop
    // input:
    const char *op_name,        // name of the operator, as a string
    const GrB_Type type,        // type of the x,y inputs to the operator
    const bool type_not_given,  // true if no type present in the string
    char err [ERRLEN]
) ;

GrB_Info gb_string_and_type_to_unop  // return op from string and type
(
    // output
    GrB_UnaryOp *unop,
    // input
    const char *op_name,        // name of the operator, as a string
    const GrB_Type type,        // type of the input to the operator
    const bool type_not_given,  // true if no type present in the string
    char err [ERRLEN]
) ;

GrB_Info gb_string_to_binop // return binary operator from a string
(
    // output
    GrB_BinaryOp *binop,        // binary op determined from the string
    // input/output:
    char *opstring,             // string that defines the binary operator
    // input:
    const GrB_Type atype,       // type of A
    const GrB_Type btype,       // type of B
    char err [ERRLEN]
) ;

GrB_Info gb_string_to_binop_or_idxunop
(
    // output:
    GrB_BinaryOp *binop,        // binary op, or NULL if idxunop
    GrB_IndexUnaryOp *idxunop,          // idxunop from the string
    // input/output:
    int64_t *ithunk,                    // thunk for idxunop
    char *opstring,                     // string defining the operator
    // input:
    const GrB_Type atype,               // type of A
    const GrB_Type btype,               // type of B
    char err [ERRLEN]
) ;

bool gb_string_to_format        // true if a valid format is found
(
    // input
    char *format_string,
    // output
    int *fmt,
    bool *fmt_present,          // true if 'by row' or 'by col' is explicit
    int *sparsity,
    bool *sparsity_present      // true if sparse/hyper/bitmap/full is explicit
) ;

GrB_Info gb_string_to_idxunop
(
    // outputs: one of the outputs is non-NULL and the other NULL
    GrB_IndexUnaryOp *op,       // GrB_IndexUnaryOp, if found
    bool *thunk_zero,           // true if op requires a thunk zero
    bool *op_is_positional,     // true if op is positional
    // input/output:
    int64_t *ithunk,
    // inputs:
    char *opstring,             // string defining the operator
    const GrB_Type atype,       // type of A, or NULL if not present
    char err [ERRLEN]
) ;

GrB_Info gb_string_to_monoid            // return monoid from a string
(
    // output
    GrB_Monoid *monoid,
    // input
    char *opstring,                     // string defining the operator
    const GrB_Type type,                // default type if not in the string
    char err [ERRLEN]
) ;

GrB_Info gb_string_to_semiring          // return a GrB semiring from a string
(
    // output:
    GrB_Semiring *semiring,
    // input/output:
    char *semiring_string,              // string defining the semiring
    // inputs:
    const GrB_Type atype,               // type of A
    const GrB_Type btype,               // type of B
    char err [ERRLEN]
) ;

GrB_Type gb_string_to_type      // return the GrB_Type from a string
(
    const char *classname
) ;

GrB_Info gb_string_to_unop              // return unary operator from a string
(
    // output
    GrB_UnaryOp *unop,                  // unary op determined by the string
    // input
    char *opstring,                     // string defining the operator
    const GrB_Type default_type,        // default type if not in the string
    char err [ERRLEN]
) ;

GrB_Info gb_typecast  // C = (type) A, where C is deep
(
    // output:
    GrB_Matrix *C_handle,
    // inputs:
    GrB_Matrix A,       // may be shallow
    GrB_Type type,      // if NULL, use the type of A
    int fmt,            // format of C
    int sparsity,       // sparsity control for C, if 0 use A
    const int arena,
    char err [ERRLEN]
) ;

// allocate/free memory space in each arena:
void *gb_malloc (size_t n, int arena) ;
void gb_free (void **p, int arena) ;

// the arena for mxMalloc/mxFree:
#define MXARENA 1

//------------------------------------------------------------------------------
// remove access to GraphBLAS polymorphic methods
//------------------------------------------------------------------------------

// The GrB MATLAB interface does not use these macros since they require a
// C11 compiler, and thus they cannot be used for MATLAB on Windows.

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

//------------------------------------------------------------------------------
// gb_* source
//------------------------------------------------------------------------------

// These files are available in all mexFunctions:
#ifndef NO_UTIL_SOURCE
#include "gb_at_exit.c"
#include "gb_binaryop_ztype.c"
#include "gb_by_col.c"
#include "gb_code_to_type.c"
#include "gb_default_format.c"
#include "gb_default_type.c"
#include "gb_dup.c"
#include "gb_error_string.c"
#include "gb_expand_scalar_to_vector.c"
#include "gb_expand_to_full.c"
#include "gb_export.c"
#include "gb_export_to_full.c"
#include "gb_export_to_sparse.c"
#include "gb_find_dot.c"
#include "gb_first_binop.c"
#include "gb_get_deep.c"
#include "gb_get_descriptor.c"
#include "gb_get_first_scalar.c"
#include "gb_get_format.c"
#include "gb_get_matlab_or_grb_matrix.c"
#include "gb_get_matrix.c"
#include "gb_get_sparsity.c"
#include "gb_is_all.c"
#include "gb_is_column_vector.c"
#include "gb_is_dense.c"
#include "gb_is_equal.c"
#include "gb_is_float.c"
#include "gb_is_integer.c"
#include "gb_is_scalar.c"
#include "gb_is_vector.c"
#include "gb_monoid_type.c"
#include "gb_new.c"
#include "gb_round_op.c"
#include "gb_string_to_format.c"
#include "gb_string_to_type.c"
#include "gb_typecast.c"
#include "gb_malloc.c"
#include "gb_free.c"
#endif

// These files are not included in all mexFunctions:
// #include "gb_binop_to_monoid.c"
// #include "gb_get_descriptor_mxm.c"
// #include "gb_string_and_type_to_binop_or_idxunop.c"
// #include "gb_string_to_binop.c"
// #include "gb_string_to_binop_or_idxunop.c"
// #include "gb_norm.c"
// #include "gb_matrix_to_list.c"
// #include "gb_cell_to_list.c"
// #include "gb_string_to_monoid.c"
// #include "gb_string_and_type_to_unop.c"
// #include "gb_string_to_unop.c"
// #include "gb_string_to_idxunop.c"
// #include "gb_semiring.c"
// #include "gb_string_to_semiring.c"

